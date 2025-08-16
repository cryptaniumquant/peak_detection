from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update, Bot

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from config import load_settings
from .services.notifier import Notifier
from .services.state_store import StateStore
from .services.data_pipeline import (
    list_strategies,
    run_realtime_cycle,
    run_realtime_cycle_async,
    run_simulation_cycle_async,
    build_notification_payload,
    run_simulation_at_async,
    build_viz_df_for_strategy_async,
)


async def run_cycle(context: ContextTypes.DEFAULT_TYPE, announce_no_signals: bool = False) -> int:
    settings = context.application.bot_data['settings']
    notifier: Notifier = context.application.bot_data['notifier']
    state: StateStore = context.application.bot_data['state']

    strategies = list_strategies(settings.strategy_whitelist)
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
        print("ERROR: App data not initialized for scheduled run")
        return
    
    try:
        settings = _app_data['settings']
        notifier = _app_data['notifier']
        state = _app_data['state']
        
        strategies = list_strategies(settings.strategy_whitelist)
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

        print(f"Scheduled cycle completed. Sent {sent} notifications.")
        
    except Exception as e:
        print(f"Error in scheduled run cycle: {e}")
        if _app_data and 'notifier' in _app_data:
            try:
                await _app_data['notifier'].send_text(f"Error in scheduled cycle: {e}")
            except:
                pass


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


async def cmd_all_viz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = context.application.bot_data['settings']
    notifier: Notifier = context.application.bot_data['notifier']

    if settings.mode != 'real':
        await update.message.reply_text("This command works in real mode only. Set MODE=real in .env and restart bot.")
        return
    strategies = list_strategies(settings.strategy_whitelist)
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
            # Use upto_ts as last index to render right edge consistent
            upto_ts = df.index.max()
            title, image_path = build_notification_payload({'strategy': s, 'df': df, 'last_signal_ts': upto_ts}, upto_ts=upto_ts)
            if image_path:
                await notifier.send_photo(image_path, caption=title)
            else:
                await notifier.send_text(title)
            sent += 1
        except Exception as e:
            await notifier.send_text(f"Failed to visualize {s}: {e}")
            continue
    if sent == 0:
        await notifier.send_text("No charts produced.")


def main():
    global _app_data
    
    settings = load_settings()
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

    # Setup APScheduler for cron-based scheduling
    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    
    # Schedule to run at the beginning of each hour (minute=0)
    # Using cron trigger: every hour at 0 minutes
    scheduler.add_job(
        scheduled_run_cycle,
        trigger="cron",
        minute=0,  # Run at the beginning of each hour
        max_instances=1,  # Prevent overlapping runs
        coalesce=True,    # If multiple runs are queued, run only the latest
        timezone=settings.timezone
    )

    # Notify start and run polling
    async def on_startup(app_inner: Application):
        scheduler.start()
        print(f"Scheduler started - will run at the beginning of each hour ({settings.timezone})")
        await notifier.send_text("Bot started. Commands: /run_now, /simulate_at, /all_viz\nScheduled to run at the beginning of each hour.")

    app.post_init = on_startup
    
    try:
        app.run_polling()
    finally:
        scheduler.shutdown()


if __name__ == '__main__':
    main()
