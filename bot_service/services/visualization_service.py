import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

try:
    # When imported as part of the package (e.g., from peak_detection)
    from bot_service.config import VIZ_DIR
except ImportError:  # pragma: no cover - fallback for running from bot_service directory
    # When running `python run_bot.py` from within bot_service/
    from config import VIZ_DIR


def visualize_last_7_days(joined_csv_path: str, upto_ts: datetime | None = None) -> str | None:
    """
    Create a 7-day visualization (similar to visualize_peak_detection.py) from a *_joined_with_peak.csv
    Returns: path to saved PNG or None on error
    """
    try:
        df = pd.read_csv(joined_csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        # Ensure timezone-aware index (UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        # Normalize upto_ts to timezone-aware
        if upto_ts is None:
            upto_ts = df.index.max()
        elif upto_ts.tzinfo is None:
            upto_ts = upto_ts.replace(tzinfo=timezone.utc)
        # Clip to last 7 days up to upto_ts
        start_ts = upto_ts - timedelta(days=7)
        dff = df.loc[(df.index >= start_ts) & (df.index <= upto_ts)].copy()
        if dff.empty:
            return None

        # Ensure required columns exist
        required = ['backtest_Value', 'smoothed_pnl', 'derivative', 'second_derivative',
                    'quantile_threshold', 'peak_detected', 'rebalance_point', 'weight']
        for col in required:
            if col not in dff.columns:
                # fill missing with safe defaults
                if col in ('smoothed_pnl', 'derivative', 'second_derivative', 'quantile_threshold'):
                    dff[col] = 0.0
                elif col in ('peak_detected', 'rebalance_point'):
                    dff[col] = False
                elif col == 'weight':
                    dff[col] = 1
        # Extract strategy name
        fname = os.path.basename(joined_csv_path)
        strategy_name = fname.replace('merged_', '').replace('_joined_with_peak.csv', '')

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        if 'backtest_Value' in dff.columns:
            ax1.plot(dff.index, dff['backtest_Value'], label='PnL', color='blue', alpha=0.6)
        if 'smoothed_pnl' in dff.columns:
            ax1.plot(dff.index, dff['smoothed_pnl'], label='Smoothed PnL', color='darkblue', linewidth=2)
        peak_points = dff[dff['peak_detected'] == True].index
        if not peak_points.empty:
            ax1.scatter(peak_points, dff.loc[peak_points, 'backtest_Value'], color='red', s=100, marker='^', label='Peak Detected')
        rebalance_points = dff[dff['rebalance_point'] == True].index
        if not rebalance_points.empty:
            ax1.scatter(rebalance_points, dff.loc[rebalance_points, 'backtest_Value'], color='purple', s=100, marker='o', label='Rebalance Point')
        for i in range(len(dff)-1):
            if dff['weight'].iloc[i] == 0:
                ax1.axvspan(dff.index[i], dff.index[i+1], alpha=0.2, color='red')
        ax1.set_title(f'Strategy: {strategy_name} — Last 7 days', fontsize=16)
        ax1.set_ylabel('PnL Value', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(dff.index, dff['derivative'], label='First Derivative', color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('First Derivative', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(dff.index, dff['second_derivative'], label='Second Derivative', color='orange')
        ax3.plot(dff.index, dff['quantile_threshold'], label='Quantile Threshold', color='red', linestyle='--')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Second Derivative', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True)

        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.step(dff.index, dff['weight'], label='Weight', color='purple', where='post')
        ax4.set_ylim(-0.1, 1.1)
        ax4.set_ylabel('Weight', fontsize=12)
        ax4.set_xlabel('Time', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.grid(True)

        for ax in (ax1, ax2, ax3, ax4):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        fig.autofmt_xdate()
        os.makedirs(VIZ_DIR, exist_ok=True)
        out_path = os.path.join(VIZ_DIR, f"{strategy_name}_last7.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"Visualization error for {joined_csv_path}: {e}")
        return None


def visualize_last_7_days_df(df: pd.DataFrame, strategy_name: str, upto_ts: datetime | None = None) -> str | None:
    """
    Create a 7-day visualization directly from an in-memory DataFrame with columns like
    'combined_Value', 'derivative', 'second_derivative', 'rebalance_point', 'weight'.
    Saves PNG into VIZ_DIR and returns the path.
    """
    try:
        if df is None or df.empty:
            return None
        # Ensure timezone-aware index (UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        if upto_ts is None:
            upto_ts = df.index.max()
        elif upto_ts.tzinfo is None:
            upto_ts = upto_ts.replace(tzinfo=timezone.utc)
        start_ts = upto_ts - timedelta(days=7)
        dff = df.loc[(df.index >= start_ts) & (df.index <= upto_ts)].copy()
        if dff.empty:
            return None

        fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        y = dff['combined_Value'] if 'combined_Value' in dff.columns else (
            dff['backtest_Value'] if 'backtest_Value' in dff.columns else None
        )
        if y is None:
            return None
        ax1.plot(dff.index, y, label='Value', color='tab:blue')
        if 'smoothed_pnl' in dff.columns:
            ax1.plot(dff.index, dff['smoothed_pnl'], label='Smoothed', color='tab:orange', alpha=0.8)
        # Mark rebalance points
        if 'rebalance_point' in dff.columns:
            rp_idx = dff.index[dff['rebalance_point'] == True]
            ax1.scatter(rp_idx, y.loc[rp_idx], color='red', marker='o', s=30, label='Rebalance')
        ax1.set_title(f"{strategy_name} — last 7 days")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if 'derivative' in dff.columns:
            ax2.plot(dff.index, dff['derivative'], color='tab:green', label='Derivative')
        if 'second_derivative' in dff.columns:
            ax2.plot(dff.index, dff['second_derivative'], color='tab:purple', label='Second Derivative', alpha=0.7)
        if 'quantile_threshold' in dff.columns:
            ax2.plot(dff.index, dff['quantile_threshold'], color='tab:brown', linestyle='--', label='Quantile thr')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        if 'weight' in dff.columns:
            ax3.step(dff.index, dff['weight'], where='post', label='Weight', color='tab:gray')
            ax3.set_ylim(-0.1, 1.1)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        os.makedirs(VIZ_DIR, exist_ok=True)
        out_path = os.path.join(VIZ_DIR, f"{strategy_name}_last7.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"Visualization (DF) error for {strategy_name}: {e}")
        return None
