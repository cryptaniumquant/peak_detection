import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta, timezone
import asyncio
from typing import List, Optional, Dict, Any

import config
from .state_store import StateStore
from .visualization_service import visualize_last_7_days, visualize_last_7_days_df

# Import existing processors (in-memory versions)
from strategy_data_processor import process_strategy_df_hours_async, process_strategy_df_async, process_strategy_df, process_strategy_df_hours
from calculate_peak import process_file as calc_process_file, process_df as calc_process_df

logger = logging.getLogger(__name__)


async def list_strategies_async(whitelist: List[str] | None) -> List[str]:
    """Get list of strategies from database analysts or whitelist (async version)"""
    if whitelist:
        return whitelist
    try:
        # Get analysts from database
        analysts = await config.get_analysts_from_database()
        if analysts:
            logger.debug(f"Found {len(analysts)} strategies from database")
            return analysts
        else:
            logger.error("No analysts found in database")
            return []
    except Exception as e:
        logger.error(f"Failed to read strategies from database: {e}")
        return []

def list_strategies(whitelist: List[str] | None) -> List[str]:
    """Get list of strategies from database analysts or whitelist (sync wrapper)"""
    if whitelist:
        return whitelist
    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, we can't run async code synchronously
            logger.error("Cannot run sync list_strategies in event loop context. Use list_strategies_async instead.")
            return []
        except RuntimeError:
            # No running loop, we can create one
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            strategies = loop.run_until_complete(list_strategies_async(whitelist))
            loop.close()
            return strategies
    except Exception as e:
        logger.error(f"Failed to read strategies: {e}")
        return []


async def build_viz_df_for_strategy_async(strategy: str, viz_window_days: int = 7) -> pd.DataFrame | None:
    """
    Build a visualization DataFrame for a strategy over the specified window (async version).
    This fetches data for the full visualization window and applies peak detection with
    per-strategy absolute thresholds if configured.
    """
    try:
        base_df = await process_strategy_df_hours_async(strategy, hours=24 * viz_window_days)
        if base_df is None or base_df.empty:
            return None
        if not isinstance(base_df.index, pd.DatetimeIndex):
            base_df.index = pd.to_datetime(base_df.index)
        if base_df.index.tz is None:
            base_df.index = base_df.index.tz_localize('UTC')
        # Always reload thresholds in async context to ensure DB query is executed
        config._cached_thresholds = await config.load_strategy_thresholds_async()
        thresholds = config._cached_thresholds
        abs_thr = thresholds.get(strategy)
        out_df = calc_process_df(base_df, absolute_threshold=abs_thr)
        if out_df is None or out_df.empty:
            return None
        if out_df.index.tz is None:
            out_df.index = out_df.index.tz_localize('UTC')
        return out_df
    except Exception as e:
        logger.error(f"build_viz_df_for_strategy error for {strategy}: {e}")
        return None

def build_viz_df_for_strategy(strategy: str, viz_window_days: int = 7) -> pd.DataFrame | None:
    """
    Build a visualization DataFrame for a strategy over the specified window.
    This fetches data for the full visualization window and applies peak detection with
    per-strategy absolute thresholds if configured.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, build_viz_df_for_strategy_async(strategy, viz_window_days))
                return future.result()
        else:
            return asyncio.run(build_viz_df_for_strategy_async(strategy, viz_window_days))
    except RuntimeError:
        return asyncio.run(build_viz_df_for_strategy_async(strategy, viz_window_days))


async def run_realtime_cycle_async(strategies: List[str], state: StateStore, detect_hours: int = 25, viz_window_days: int = 7) -> list[dict]:
    """
    Real mode: process strategies and notify only on a NEW rebalance that appears at the latest timestamp (async version).
    Prevent historical backfill: on the first run (no state), initialize last_notified to the latest historical rebalance and skip sending.
    Returns list of events: {strategy, joined_path, last_signal_ts}
    """
    events: list[dict] = []
    # Always reload thresholds in async context to ensure DB query is executed
    config._cached_thresholds = await config.load_strategy_thresholds_async()
    thresholds = config._cached_thresholds
    for s in strategies:
        try:
            # In-memory processing: fetch only last <detect_hours> for detection
            hourly_df = await process_strategy_df_hours_async(s, hours=detect_hours)
            if hourly_df is None or hourly_df.empty:
                continue
            if not isinstance(hourly_df.index, pd.DatetimeIndex):
                hourly_df.index = pd.to_datetime(hourly_df.index)
            if hourly_df.index.tz is None:
                hourly_df.index = hourly_df.index.tz_localize('UTC')
            # Detect on last <detect_hours> hours only
            latest_ts = hourly_df.index.max()
            detect_start = latest_ts - timedelta(hours=detect_hours)
            detect_df = hourly_df.loc[(hourly_df.index >= detect_start) & (hourly_df.index <= latest_ts)].copy()
            if detect_df.empty:
                continue
            abs_thr = thresholds.get(s)
            dff = calc_process_df(detect_df, absolute_threshold=abs_thr)
            if dff is None or dff.empty or 'rebalance_point' not in dff.columns:
                continue
            if dff.index.tz is None:
                dff.index = dff.index.tz_localize('UTC')

            last_notified = state.get_last_notified(s)
            latest_mask = dff['rebalance_point'] == True
            latest_signal_ts = None
            if latest_mask.any():
                latest_signal_ts = dff.index[latest_mask].max()
            if last_notified is None and latest_signal_ts is not None:
                state.set_last_notified(s, latest_signal_ts)
                continue
            # Check if there's a valid rebalance signal at latest_ts
            # Only trigger if we have valid derivative data (not NaN)
            is_rebalance_now = False
            if (latest_ts in dff.index and 
                not pd.isna(dff.loc[latest_ts, 'derivative']) and
                not pd.isna(dff.loc[latest_ts, 'second_derivative']) and
                not pd.isna(dff.loc[latest_ts, 'quantile_threshold'])):
                is_rebalance_now = bool(dff.loc[latest_ts, 'rebalance_point'])
            
            if is_rebalance_now and ((last_notified is None) or (latest_ts > last_notified)):
                # Use the actual latest signal timestamp, not latest_ts
                actual_signal_ts = latest_signal_ts if latest_signal_ts is not None else latest_ts
                
                # Build visualization DF up to latest_ts (fetch only last viz_window_days)
                viz_base = await process_strategy_df_hours_async(s, hours=24 * viz_window_days)
                if viz_base is not None and not viz_base.empty and viz_base.index.tz is None:
                    viz_base.index = viz_base.index.tz_localize('UTC')
                if viz_base is not None and not viz_base.empty:
                    viz_base = viz_base.loc[(viz_base.index <= latest_ts)]
                viz_df = calc_process_df(viz_base, absolute_threshold=abs_thr) if not viz_base.empty else None
                if viz_df is not None and not viz_df.empty:
                    events.append({'strategy': s, 'df': viz_df, 'last_signal_ts': actual_signal_ts})
        except Exception as e:
            logger.error(f"Realtime cycle error for {s}: {e}")
            continue
    return events

def run_realtime_cycle(strategies: List[str], state: StateStore, detect_hours: int = 25, viz_window_days: int = 7) -> list[dict]:
    """
    Real mode: process strategies and notify only on a NEW rebalance that appears at the latest timestamp (sync wrapper).
    Prevent historical backfill: on the first run (no state), initialize last_notified to the latest historical rebalance and skip sending.
    Returns list of events: {strategy, joined_path, last_signal_ts}
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_realtime_cycle_async(strategies, state, detect_hours, viz_window_days))
                return future.result()
        else:
            return asyncio.run(run_realtime_cycle_async(strategies, state, detect_hours, viz_window_days))
    except RuntimeError:
        return asyncio.run(run_realtime_cycle_async(strategies, state, detect_hours, viz_window_days))


async def run_simulation_cycle_async(strategies: List[str], state: StateStore, window_hours: int, step_hours: int) -> tuple[list[dict], datetime]:
    """
    Simulation mode: maintain a moving 'now' and only allow data up to that now (async version).
    We process strategy -> merged csv; then clip to [now - window, now] and run peak detection on the clipped temp file.
    Returns (events, new_now)
    """
    now = state.get_simulation_now()
    if now is None:
        now = datetime.now(timezone.utc)
    new_now = now + timedelta(hours=step_hours)

    events: list[dict] = []
    # Always reload thresholds in async context to ensure DB query is executed
    config._cached_thresholds = await config.load_strategy_thresholds_async()
    thresholds = config._cached_thresholds
    for s in strategies:
        try:
            hourly_df = await process_strategy_df_async(s, days=max(30, window_hours // 24 + 2))
            if hourly_df is None or hourly_df.empty:
                continue
            if hourly_df.index.tz is None:
                hourly_df.index = hourly_df.index.tz_localize('UTC')
            start_ts = new_now - timedelta(hours=window_hours)
            dff = hourly_df.loc[(hourly_df.index >= start_ts) & (hourly_df.index <= new_now)].copy()
            if dff.empty:
                continue
            abs_thr = thresholds.get(s)
            out_df = calc_process_df(dff, absolute_threshold=abs_thr)
            if out_df is None or out_df.empty or 'rebalance_point' not in out_df.columns:
                continue
            if out_df.index.tz is None:
                out_df.index = out_df.index.tz_localize('UTC')
            last_notified = state.get_last_notified(s)
            mask = out_df['rebalance_point'] == True
            if last_notified is not None:
                mask &= (out_df.index > last_notified)
            new_points = out_df[mask]
            if not new_points.empty:
                last_signal_ts = new_points.index.max()
                events.append({'strategy': s, 'df': out_df, 'last_signal_ts': last_signal_ts})
        except Exception as e:
            logger.error(f"Simulation cycle error for {s}: {e}")
            continue

    return events, new_now

def run_simulation_cycle(strategies: List[str], state: StateStore, window_hours: int, step_hours: int) -> tuple[list[dict], datetime]:
    """
    Simulation mode: maintain a moving 'now' and only allow data up to that now (sync wrapper).
    We process strategy -> merged csv; then clip to [now - window, now] and run peak detection on the clipped temp file.
    Returns (events, new_now)
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_simulation_cycle_async(strategies, state, window_hours, step_hours))
                return future.result()
        else:
            return asyncio.run(run_simulation_cycle_async(strategies, state, window_hours, step_hours))
    except RuntimeError:
        return asyncio.run(run_simulation_cycle_async(strategies, state, window_hours, step_hours))


def build_notification_payload(event: dict, upto_ts: Optional[datetime] = None) -> tuple[str, Optional[str]]:
    s = event['strategy']
    ts = event['last_signal_ts']
    
    # Check if we have a valid signal timestamp
    if ts is None or pd.isna(ts):
        title = f"Rebalance signal: {s}\nTime: No signals detected"
    else:
        title = f"Rebalance signal: {s}\nTime: {ts.strftime('%Y-%m-%d %H:%M UTC')}"
    df = event.get('df')
    if df is not None:
        image_path = visualize_last_7_days_df(df, s, upto_ts=upto_ts or ts)
    else:
        # backward-compatibility if event carries a path
        joined_path = event.get('joined_path')
        image_path = visualize_last_7_days(joined_path, upto_ts=upto_ts or ts) if joined_path else None
    return title, image_path


async def run_simulation_at_async(strategy: str, at_dt: datetime, window_hours: int, detect_hours: int = 25, viz_window_days: int = 7) -> Optional[dict]:
    """
    Simulate a single strategy at a specific datetime (UTC-aware) - async version.
    Process data, clip to [at_dt - window_hours, at_dt], run peak detection, and
    return an event if the last row (at_dt) has rebalance_point == True.
    Returns: {strategy, joined_path, last_signal_ts} or None.
    """
    try:
        # Always reload thresholds in async context to ensure DB query is executed
        config._cached_thresholds = await config.load_strategy_thresholds_async()
        thresholds = config._cached_thresholds
        abs_thr = thresholds.get(strategy)
        
        hourly_df = await process_strategy_df_async(strategy, days=max(30, window_hours // 24 + 2))
        if hourly_df is None or hourly_df.empty:
            return None
        if hourly_df.index.tz is None:
            hourly_df.index = hourly_df.index.tz_localize('UTC')
        # Always detect on last <detect_hours> for consistency with realtime
        start_ts = at_dt - timedelta(hours=detect_hours)
        dff = hourly_df.loc[(hourly_df.index >= start_ts) & (hourly_df.index <= at_dt)].copy()
        if dff.empty:
            return None
        out_df = calc_process_df(dff, absolute_threshold=abs_thr)
        if out_df is None or out_df.empty or 'rebalance_point' not in out_df.columns:
            return None
        if out_df.index.tz is None:
            out_df.index = out_df.index.tz_localize('UTC')
        last_ts = out_df.index.max()
        if last_ts != at_dt:
            return None
        if bool(out_df.loc[last_ts, 'rebalance_point']):
            # Build viz df up to at_dt
            viz_start = at_dt - timedelta(days=viz_window_days)
            viz_base = hourly_df.loc[(hourly_df.index >= viz_start) & (hourly_df.index <= at_dt)].copy()
            viz_df = calc_process_df(viz_base, absolute_threshold=abs_thr) if not viz_base.empty else None
            if viz_df is None or viz_df.empty:
                return None
            return {'strategy': strategy, 'df': viz_df, 'last_signal_ts': last_ts}
        return None
    except Exception as e:
        logger.error(f"Simulation-at error for {strategy} @ {at_dt}: {e}")
        return None

def run_simulation_at(strategy: str, at_dt: datetime, window_hours: int, detect_hours: int = 25, viz_window_days: int = 7) -> Optional[dict]:
    """
    Simulate a single strategy at a specific datetime (UTC-aware) - sync wrapper.
    Process data, clip to [at_dt - window_hours, at_dt], run peak detection, and
    return an event if the last row (at_dt) has rebalance_point == True.
    Returns: {strategy, joined_path, last_signal_ts} or None.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_simulation_at_async(strategy, at_dt, window_hours, detect_hours, viz_window_days))
                return future.result()
        else:
            return asyncio.run(run_simulation_at_async(strategy, at_dt, window_hours, detect_hours, viz_window_days))
    except RuntimeError:
        return asyncio.run(run_simulation_at_async(strategy, at_dt, window_hours, detect_hours, viz_window_days))
