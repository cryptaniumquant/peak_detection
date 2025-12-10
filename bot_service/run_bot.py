from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update, Bot

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import asyncio
from asyncio import sleep
import logging
from functools import wraps
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from logging_config import setup_logging
from .services.notifier import Notifier
from .services.state_store import StateStore
from .services.data_pipeline import (
    list_strategies,
    list_strategies_async,
    run_realtime_cycle,
    run_realtime_cycle_async,
    run_simulation_cycle_async,
    run_dummy_cycle_async,
    simulate_dummy_week_async,
    build_notification_payload,
    build_dummy_notification_payload,
    build_dummy_simulation_payload,
    build_viz_df_for_strategy_async,
)
from strategy_data_processor import get_strategy_data_async

logger = logging.getLogger(__name__)


def authorized_chat_only(func):
    """Decorator to check if command comes from authorized chat_id"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = context.application.bot_data['settings']
        authorized_chat_id = int(settings.telegram_chat_id)
        
        if update.effective_chat.id != authorized_chat_id:
            # Silently ignore commands from unauthorized chats
            logger.warning(f"Unauthorized command attempt from chat_id: {update.effective_chat.id}")
            return
        
        return await func(update, context)
    return wrapper


async def run_cycle(context: ContextTypes.DEFAULT_TYPE, announce_no_signals: bool = False) -> int:
    settings = context.application.bot_data['settings']
    notifier: Notifier = context.application.bot_data['notifier']
    state: StateStore = context.application.bot_data['state']

    strategies = await list_strategies_async(settings.strategy_whitelist)
    if not strategies:
        await notifier.send_text("No strategies to process.")
        return 0

    sent = 0
    if settings.mode == 'simulate':
        events, new_now = run_simulation_cycle(
            strategies, state, settings.sim_window_hours, settings.sim_step_hours
        )
        upto_ts = new_now
        if events:
            for ev in events:
                title, image_path = build_notification_payload(ev, upto_ts=upto_ts)
                if image_path:
                    await notifier.send_photo(image_path, caption=title)
                else:
                    await notifier.send_text(title)
                state.set_last_notified(ev['strategy'], ev['last_signal_ts'])
                sent += 1
        state.set_simulation_now(new_now)
    elif settings.mode == 'dummy':
        # Dummy mode: simple PnL difference detection
        events = await run_dummy_cycle_async(
            strategies,
            state,
            detect_hours=settings.real_detect_hours,
            viz_window_days=settings.viz_window_days,
        )
        if events:
            for ev in events:
                title, image_path = build_dummy_notification_payload(ev)
                if image_path:
                    await notifier.send_photo(image_path, caption=title)
                else:
                    await notifier.send_text(title)
                sent += 1
    else:
        events = run_realtime_cycle(
            strategies,
            state,
            detect_hours=settings.real_detect_hours,
            viz_window_days=settings.viz_window_days,
        )
        if events:
            for ev in events:
                title, image_path = build_notification_payload(ev)
                if image_path:
                    await notifier.send_photo(image_path, caption=title)
                else:
                    await notifier.send_text(title)
                state.set_last_notified(ev['strategy'], ev['last_signal_ts'])
                sent += 1

    if announce_no_signals and sent == 0:
        await notifier.send_text("No new signals on the latest candle.")
    return sent


# Global variables for APScheduler access
_app_data = None

async def scheduled_run_cycle():
    """Scheduled function for APScheduler - runs at the beginning of each hour"""
    if _app_data is None:
        logger.error("App data not initialized for scheduled run")
        return
    
    try:
        settings = _app_data['settings']
        notifier = _app_data['notifier']
        state = _app_data['state']
        
        strategies = await list_strategies_async(settings.strategy_whitelist)
        if not strategies:
            await notifier.send_text("No strategies to process.")
            return

        sent = 0
        if settings.mode == 'simulate':
            events, new_now = await run_simulation_cycle_async(
                strategies, state, settings.sim_window_hours, settings.sim_step_hours
            )
            upto_ts = new_now
            if events:
                for ev in events:
                    title, image_path = build_notification_payload(ev, upto_ts=upto_ts)
                    if image_path:
                        await notifier.send_photo(image_path, caption=title)
                    else:
                        await notifier.send_text(title)
                    state.set_last_notified(ev['strategy'], ev['last_signal_ts'])
                    sent += 1
            state.set_simulation_now(new_now)
        elif settings.mode == 'dummy':
            # Dummy mode: simple PnL difference detection
            events = await run_dummy_cycle_async(
                strategies,
                state,
                detect_hours=settings.real_detect_hours,
                viz_window_days=settings.viz_window_days,
            )
            if events:
                for ev in events:
                    title, image_path = build_dummy_notification_payload(ev)
                    if image_path:
                        await notifier.send_photo(image_path, caption=title)
                    else:
                        await notifier.send_text(title)
                    sent += 1
        else:
            events = await run_realtime_cycle_async(
                strategies,
                state,
                detect_hours=settings.real_detect_hours,
                viz_window_days=settings.viz_window_days,
            )
            if events:
                for ev in events:
                    title, image_path = build_notification_payload(ev)
                    if image_path:
                        await notifier.send_photo(image_path, caption=title)
                    else:
                        await notifier.send_text(title)
                    state.set_last_notified(ev['strategy'], ev['last_signal_ts'])
                    sent += 1

        logger.info(f"Scheduled cycle completed. Sent {sent} notifications.")
        
    except Exception as e:
        logger.error(f"Error in scheduled run cycle: {e}")
        if _app_data and 'notifier' in _app_data:
            try:
                await _app_data['notifier'].send_text(f"Error in scheduled cycle: {e}")
            except:
                pass


@authorized_chat_only
async def cmd_run_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Running job now...")
    await run_cycle(context, announce_no_signals=True)
    await update.message.reply_text("Done.")


def _parse_datetime_in_tz(dt_str: str, tz_name: str) -> datetime | None:
    fmts = ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"]
    for fmt in fmts:
        try:
            naive = datetime.strptime(dt_str, fmt)
            return naive.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
        except Exception:
            continue
    # try fromisoformat as last resort
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@authorized_chat_only
async def cmd_simulate_at(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = context.application.bot_data['settings']
    notifier: Notifier = context.application.bot_data['notifier']

    if settings.mode != 'simulate':
        await update.message.reply_text("This command works in simulate mode only. Set MODE=simulate in .env and restart bot.")
        return
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /simulate_at <strategy> <YYYY-MM-DD HH:MM>")
        return
    strategy = context.args[0]
    dt_str = " ".join(context.args[1:])
    at_dt_utc = _parse_datetime_in_tz(dt_str, settings.timezone)
    if at_dt_utc is None:
        await update.message.reply_text("Invalid datetime. Use YYYY-MM-DD HH:MM or YYYY-MM-DDTHH:MM")
        return

    await update.message.reply_text(f"Simulating {strategy} at {at_dt_utc.strftime('%Y-%m-%d %H:%M UTC')}...")
    ev = await run_simulation_at_async(
        strategy,
        at_dt_utc,
        settings.sim_window_hours,
        detect_hours=settings.real_detect_hours,
        viz_window_days=settings.viz_window_days,
    )
    if ev:
        title, image_path = build_notification_payload(ev, upto_ts=at_dt_utc)
        if image_path:
            await notifier.send_photo(image_path, caption=title)
        else:
            await notifier.send_text(title)
    else:
        await notifier.send_text("No signal at the specified time.")


@authorized_chat_only
async def cmd_dummy_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test dummy mode for past week - shows all potential signals"""
    settings = context.application.bot_data['settings']
    notifier: Notifier = context.application.bot_data['notifier']

    if settings.mode != 'dummy':
        await update.message.reply_text("This command works in dummy mode only. Set MODE=dummy and restart bot.")
        return
    
    # Get optional parameters: strategy name and days back
    strategy_filter = None
    days_back = 7
    
    if context.args:
        if len(context.args) >= 1:
            strategy_filter = context.args[0]
        if len(context.args) >= 2:
            try:
                days_back = int(context.args[1])
            except ValueError:
                await update.message.reply_text("Days must be a number. Using default: 7 days")
                days_back = 7
    
    strategies = await list_strategies_async(settings.strategy_whitelist)
    if not strategies:
        await notifier.send_text("No strategies to test.")
        return
    
    # Filter to specific strategy if requested
    if strategy_filter:
        strategies = [s for s in strategies if strategy_filter.lower() in s.lower()]
        if not strategies:
            await notifier.send_text(f"No strategies matching '{strategy_filter}'")
            return
    
    await update.message.reply_text(f"Testing dummy mode for last {days_back} days...")
    sent = 0
    
    for s in strategies:
        try:
            result = await simulate_dummy_week_async(s, days_back=days_back)
            if result is None:
                continue
            
            title, image_path = build_dummy_simulation_payload(result)
            if image_path:
                await notifier.send_photo(image_path, caption=title)
            else:
                await notifier.send_text(title)
            sent += 1
            
            # Add delay to prevent Telegram flood control
            await sleep(3)
            
        except Exception as e:
            logger.error(f"Failed dummy test for {s}: {e}")
            await notifier.send_text(f"Failed to test {s}: {e}")
            continue
    
    if sent == 0:
        await notifier.send_text("No test results produced.")
    else:
        await notifier.send_text(f"✅ Dummy test completed for {sent} strategies")


@authorized_chat_only
async def cmd_all_viz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = context.application.bot_data['settings']
    notifier: Notifier = context.application.bot_data['notifier']

    if settings.mode != 'real':
        await update.message.reply_text("This command works in real mode only. Set MODE=real in .env and restart bot.")
        return
    strategies = await list_strategies_async(settings.strategy_whitelist)
    if not strategies:
        await notifier.send_text("No strategies to visualize.")
        return
    await update.message.reply_text("Building 7-day charts for all strategies...")
    sent = 0
    for s in strategies:
        try:
            df = await build_viz_df_for_strategy_async(s, settings.viz_window_days)
            if df is None or df.empty:
                continue
            # Find the actual last signal timestamp, not just the last data point
            rebalance_mask = df['rebalance_point'] == True
            if rebalance_mask.any():
                last_signal_ts = df.index[rebalance_mask].max()
            else:
                last_signal_ts = None
            
            # Use upto_ts as last index to render right edge consistent
            upto_ts = df.index.max()
            title, image_path = build_notification_payload({'strategy': s, 'df': df, 'last_signal_ts': last_signal_ts}, upto_ts=upto_ts)
            if image_path:
                await notifier.send_photo(image_path, caption=title)
            else:
                await notifier.send_text(title)
            sent += 1
            # Add delay to prevent Telegram flood control errors
            await sleep(3)
        except Exception as e:
            await notifier.send_text(f"Failed to visualize {s}: {e}")
            continue
    if sent == 0:
        await notifier.send_text("No charts produced.")


def main():
    global _app_data
    
    settings = config.load_settings()
    
    # Setup logging with Telegram error notifications
    setup_logging(
        log_level=config.LOG_LEVEL,
        telegram_token=settings.telegram_token,
        telegram_chat_id=settings.telegram_chat_id
    )
    
    logger.info("Trading Signal Bot starting...")
    
    app = Application.builder().token(settings.telegram_token).build()

    bot = app.bot  # type: Bot
    notifier = Notifier(bot, settings.telegram_chat_id)
    state = StateStore()

    app.bot_data['settings'] = settings
    app.bot_data['notifier'] = notifier
    app.bot_data['state'] = state
    
    # Store app data globally for APScheduler access
    _app_data = app.bot_data

    app.add_handler(CommandHandler('run_now', cmd_run_now))
    app.add_handler(CommandHandler('simulate_at', cmd_simulate_at))
    app.add_handler(CommandHandler('all_viz', cmd_all_viz))
    app.add_handler(CommandHandler('dummy_test', cmd_dummy_test))

    # Setup APScheduler for cron-based scheduling
    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    
    # Schedule to run using configuration parameters with hardcoded safety settings
    scheduler.add_job(
        scheduled_run_cycle,
        timezone=settings.timezone,
        max_instances=1,    # Prevent overlapping runs (hardcoded)
        coalesce=True,      # If multiple runs are queued, run only the latest (hardcoded)
        **config.SCHEDULER_JOB_CONFIG
    )

    # Notify start and run polling
    async def on_startup(app_inner: Application):
        scheduler.start()
        logger.info(f"Scheduler started - will run at the beginning of each hour ({settings.timezone})")
        
        # Build startup message with mode info
        mode_name = settings.mode.upper()
        startup_msg = f"Bot started in {mode_name} mode.\n"
        startup_msg += "Commands: /run_now"
        
        if settings.mode == 'simulate':
            startup_msg += ", /simulate_at"
        elif settings.mode == 'real':
            startup_msg += ", /all_viz"
        elif settings.mode == 'dummy':
            startup_msg += ", /dummy_test"
        
        startup_msg += "\nScheduled to run at the beginning of each hour.\n"
        
        if settings.mode == 'dummy':
            startup_msg += f"\n💡 Dummy Mode Settings:\n"
            startup_msg += f"  • Threshold: ${config.DUMMY_PNL_THRESHOLD:,.0f}\n"
            startup_msg += f"  • Cooldown: {config.DUMMY_COOLDOWN_HOURS}h\n"
            startup_msg += f"  • Use /dummy_test to simulate past week"
        
        await notifier.send_text(startup_msg)

    app.post_init = on_startup
    
    try:
        app.run_polling()
    finally:
        scheduler.shutdown()


if __name__ == '__main__':
    main()
