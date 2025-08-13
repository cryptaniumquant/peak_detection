# Peak Detection Telegram Bot Service

This service provides real-time trading signal detection and visualization through a Telegram bot that:

- **Optimized Data Fetching**: Fetches only the last 25 hours of data for signal detection (aligned with Savitzky-Golay filter window), reducing database load and memory usage.
- **Configurable Absolute Thresholds**: Uses per-strategy absolute threshold values from `strategy_thresholds.json` instead of dynamic rolling quantiles for consistent and predictable peak detection.
- **Real-time Signal Detection**: Detects new rebalance points and sends immediate Telegram notifications with 7-day visualization charts.
- **Simulation Mode**: Supports historical signal reproduction at specific timestamps for backtesting and analysis.
- **Flexible Configuration**: All time windows and thresholds are configurable via environment variables and JSON config files.

## Features

### Real-time Mode (`MODE=real`)
- Monitors strategies continuously with configurable intervals
- Detects peaks using absolute thresholds on the second derivative of smoothed PnL
- Sends notifications only for new signals (prevents spam from historical data)
- Generates 7-day visualization charts automatically when signals are detected

### Simulation Mode (`MODE=simulate`)
- Sliding window simulation for historical analysis
- Reproduces exact signal detection behavior at any past timestamp
- Consistent with real-time detection logic and visualization

### Telegram Commands
- `/run_now` — Trigger detection cycle immediately
- `/simulate_at <strategy> <YYYY-MM-DD HH:MM>` — Simulate signal detection at specific time (simulation mode only)
- `/all_viz` — Generate 7-day charts for all strategies regardless of signals (real mode only)

## Quick Start

1) **Configure Environment**: Create `.env` file with your settings:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
MODE=real
REAL_DETECT_HOURS=25
VIZ_WINDOW_DAYS=7
SCHEDULE_MINUTES=60
TIMEZONE=Europe/Moscow
```

2) **Configure Thresholds**: Edit `../strategy_thresholds.json` with per-strategy absolute thresholds:
```json
{
  "pcatr_eth_30m_1": -18.0,
  "fluger_sol_30m_1": -25.0
}
```

3) **Install Dependencies**:
```bash
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4) **Run the Bot**:
```bash
python run_bot.py
```

## Configuration

### Environment Variables (`.env`)
- `TELEGRAM_BOT_TOKEN` — Your Telegram bot token
- `TELEGRAM_CHAT_ID` — Chat ID for notifications
- `MODE` — `real` or `simulate`
- `REAL_DETECT_HOURS` — Hours of data to fetch for detection (default: 25)
- `VIZ_WINDOW_DAYS` — Days of data for visualization (default: 7)
- `SCHEDULE_MINUTES` — Detection cycle interval in minutes (default: 60)
- `TIMEZONE` — Timezone for datetime parsing (default: Europe/Moscow)
- `STRATEGY_WHITELIST` — Comma-separated strategy names (empty = all strategies)

### Strategy Thresholds (`../strategy_thresholds.json`)
Per-strategy absolute thresholds for the second derivative of smoothed PnL:
- Negative values detect downward peaks (typical for rebalance signals)
- Values are in the same units as the second derivative of your PnL data
- Fallback to `../strategy_quantile_values.csv` if JSON is missing

## Architecture

### Core Components
- `run_bot.py` — Main bot application with command handlers and scheduler
- `config.py` — Configuration loader for environment variables and strategy thresholds
- `services/data_pipeline.py` — Core pipeline orchestrating data fetch, processing, and detection
- `services/visualization_service.py` — Chart generation for 7-day analysis
- `services/notifier.py` — Telegram notification utilities
- `services/state_store.py` — Persistent state management for last notification timestamps

### Data Processing Pipeline
1. **Fetch**: Get limited hours of data from database via `strategy_data_processor.py`
2. **Process**: Apply Savitzky-Golay smoothing and compute derivatives via `calculate_peak.py`
3. **Detect**: Compare second derivative against absolute thresholds
4. **Visualize**: Generate comprehensive 7-day charts showing PnL, derivatives, thresholds, and signals
5. **Notify**: Send Telegram messages with charts for new signals only

### Signal Detection Logic
- Uses 25-hour window for detection (aligned with Savitzky-Golay filter requirements)
- Peak detected when first derivative changes from positive to negative AND second derivative falls below threshold
- Rebalance point marked on the candle following the detected peak
- Weight logic applies cooldown and recovery periods to avoid frequent signals

## Usage Examples

### Real-time Monitoring
```bash
# Set MODE=real in .env
python run_bot.py
# Bot runs every hour, sends notifications for new signals
```

### Historical Analysis
```bash
# Set MODE=simulate in .env
python run_bot.py
# In Telegram: /simulate_at pcatr_eth_30m_1 2025-01-15 14:30
```

### Generate All Charts
```bash
# In Telegram (real mode): /all_viz
# Generates 7-day charts for all strategies
```

## Notes

- **Efficient Data Usage**: Only fetches necessary data (25h for detection, 7d for visualization when needed)
- **Consistent Behavior**: Real-time and simulation modes use identical detection logic
- **Threshold Flexibility**: Easy to update thresholds via JSON without code changes
- **State Persistence**: Remembers last notifications to prevent duplicate alerts
- **Timezone Aware**: All timestamps handled with proper timezone conversion
