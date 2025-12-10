"""
Dummy mode peak detection - simple approach based on PnL difference
Triggers rebalance signal when unrealized_pnl exceeds realized_pnl by threshold
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def process_dummy_df(df_in: pd.DataFrame, pnl_threshold: float = 10000.0, 
                     last_rebalance_ts: Optional[datetime] = None,
                     cooldown_hours: int = 24) -> pd.DataFrame | None:
    """
    Process DataFrame for dummy mode detection.
    
    Args:
        df_in: DataFrame with column 'unrealized_pnl' (and optional 'realized_pnl')
        pnl_threshold: Threshold for unrealized_pnl to trigger signal (e.g., 10000)
        last_rebalance_ts: Timestamp of last rebalance (for cooldown)
        cooldown_hours: Hours to wait after rebalance before next signal
        
    Returns:
        DataFrame with additional columns:
        - pnl_difference: unrealized_pnl - realized_pnl (if realized_pnl exists, for visualization)
        - rebalance_point: True if unrealized_pnl >= threshold and not in cooldown
        - in_cooldown: True if within cooldown period
    """
    if df_in is None or df_in.empty:
        logger.warning("Empty input DataFrame for dummy processing")
        return None
    
    df = df_in.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}")
            return None
    
    # Check required columns
    if 'unrealized_pnl' not in df.columns:
        logger.error("Missing required column: unrealized_pnl")
        return None
    
    # Calculate PnL difference for visualization (optional, if realized_pnl exists)
    if 'realized_pnl' in df.columns:
        df['pnl_difference'] = df['unrealized_pnl'] - df['realized_pnl']
    else:
        df['pnl_difference'] = np.nan
    
    # Initialize rebalance and cooldown columns
    df['rebalance_point'] = False
    df['in_cooldown'] = False
    
    # Track cooldown state
    cooldown_until = None
    if last_rebalance_ts is not None:
        if last_rebalance_ts.tzinfo is None:
            last_rebalance_ts = last_rebalance_ts.replace(tzinfo=timezone.utc)
        cooldown_until = last_rebalance_ts + timedelta(hours=cooldown_hours)
    
    # Process each row to detect rebalance points
    for idx in df.index:
        current_time = idx
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        # Check if we're in cooldown period
        if cooldown_until is not None and current_time <= cooldown_until:
            df.loc[idx, 'in_cooldown'] = True
            continue
        
        # Check if unrealized_pnl exceeds threshold
        unrealized = df.loc[idx, 'unrealized_pnl']
        if not pd.isna(unrealized) and unrealized >= pnl_threshold:
            df.loc[idx, 'rebalance_point'] = True
            # Update cooldown for subsequent rows
            cooldown_until = current_time + timedelta(hours=cooldown_hours)
            logger.debug(f"Rebalance signal at {current_time}: unrealized_pnl={unrealized:.2f}, cooldown until {cooldown_until}")
    
    return df


def check_latest_signal(df: pd.DataFrame, detect_hours: int = 28) -> tuple[bool, Optional[datetime]]:
    """
    Check if there's a rebalance signal at the latest timestamp.
    
    Args:
        df: Processed DataFrame with rebalance_point column
        detect_hours: Hours to look back for validation
        
    Returns:
        (has_signal, signal_timestamp) tuple
    """
    if df is None or df.empty or 'rebalance_point' not in df.columns:
        return False, None
    
    # Get latest timestamp
    latest_ts = df.index.max()
    
    # Check if latest point has rebalance signal
    if df.loc[latest_ts, 'rebalance_point']:
        return True, latest_ts
    
    return False, None


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for dummy mode DataFrame.
    
    Returns:
        Dictionary with stats: current_diff, threshold_exceeded, signals_count
    """
    if df is None or df.empty:
        return {}
    
    stats = {
        'latest_realized_pnl': None,
        'latest_unrealized_pnl': None,
        'latest_pnl_difference': None,
        'signals_count': 0,
        'cooldown_hours_count': 0
    }
    
    if 'realized_pnl' in df.columns and not df['realized_pnl'].empty:
        stats['latest_realized_pnl'] = df['realized_pnl'].iloc[-1]
    
    if 'unrealized_pnl' in df.columns and not df['unrealized_pnl'].empty:
        stats['latest_unrealized_pnl'] = df['unrealized_pnl'].iloc[-1]
    
    if 'pnl_difference' in df.columns and not df['pnl_difference'].empty:
        stats['latest_pnl_difference'] = df['pnl_difference'].iloc[-1]
    
    if 'rebalance_point' in df.columns:
        stats['signals_count'] = int(df['rebalance_point'].sum())
    
    if 'in_cooldown' in df.columns:
        stats['cooldown_hours_count'] = int(df['in_cooldown'].sum())
    
    return stats
