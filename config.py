"""
Unified configuration file for Peak Detection system.
Contains all constants and default values.
User-specific credentials are imported from local_settings.py
"""
import os
import json
import csv
from dataclasses import dataclass
from typing import Optional, List, Dict

# SQLAlchemy async database URI - constructed from individual params or override in local_settings.py
SQLALCHEMY_DATABASE_URI = None

# =============================================================================
# TELEGRAM BOT CONFIGURATION
# =============================================================================
# Telegram bot credentials - override in local_settings.py
TELEGRAM_BOT_TOKEN = None
TELEGRAM_CHAT_ID = None

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
# Bot operation mode
MODE = 'real'  # 'real' or 'simulate'

# Strategy filtering
STRATEGY_WHITELIST = []  # Empty list means all strategies

# Time windows
REAL_DETECT_HOURS = 25  # Hours of data to fetch for signal detection
VIZ_WINDOW_DAYS = 7     # Days of data for visualization

# Scheduling (APScheduler cron-based)
SCHEDULE_MINUTES = 60   # Legacy setting - now uses cron scheduling
SCHEDULER_CRON_MINUTE = 0  # Run at minute 0 of each hour (beginning of hour)
SCHEDULER_CRON_HOUR = "*"  # Every hour

# Timezone
TIMEZONE = 'Europe/Moscow'

# Simulation parameters
SIM_WINDOW_HOURS = 168  # 7 days
SIM_STEP_HOURS = 1

# =============================================================================
# FILE PATHS
# =============================================================================
# Base directory (peak_detection folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')
STATE_DIR = os.path.join(BASE_DIR, 'bot_service', 'state')

# Configuration files
STRATEGY_THRESHOLDS_JSON = os.path.join(BASE_DIR, 'strategy_thresholds.json')
STRATEGY_QUANTILE_CSV = os.path.join(BASE_DIR, 'strategy_quantile_values.csv')

# =============================================================================
# DATACLASS FOR SETTINGS
# =============================================================================
@dataclass
class Settings:
    """Settings dataclass for bot configuration"""
    telegram_token: str
    telegram_chat_id: int
    mode: str
    strategy_whitelist: List[str]
    real_detect_hours: int
    viz_window_days: int
    schedule_minutes: int
    timezone: str
    sim_window_hours: int
    sim_step_hours: int

# =============================================================================
# CONFIGURATION LOADERS
# =============================================================================
def load_strategy_thresholds() -> Dict[str, float]:
    """
    Load per-strategy absolute thresholds from JSON file.
    Falls back to CSV if JSON is not found.
    """
    # Try JSON first
    if os.path.exists(STRATEGY_THRESHOLDS_JSON):
        try:
            with open(STRATEGY_THRESHOLDS_JSON, 'r') as f:
                thresholds = json.load(f)
                print(f"Loaded {len(thresholds)} strategy thresholds from JSON")
                return thresholds
        except Exception as e:
            print(f"Error loading strategy thresholds from JSON: {e}")
    
    # Fallback to CSV
    if os.path.exists(STRATEGY_QUANTILE_CSV):
        try:
            thresholds = {}
            with open(STRATEGY_QUANTILE_CSV, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    strategy = row.get('strategy')
                    # Use quantile_value (negative) as absolute threshold
                    threshold = float(row.get('quantile_value', 0))
                    if strategy:
                        thresholds[strategy] = threshold
            print(f"Loaded {len(thresholds)} strategy thresholds from CSV fallback")
            return thresholds
        except Exception as e:
            print(f"Error loading strategy thresholds from CSV: {e}")
    
    print("No strategy thresholds found, using dynamic quantiles")
    return {}

def load_settings() -> Settings:
    """Load all settings, combining defaults with local_settings overrides"""
    # Create settings with current values
    settings = Settings(
        telegram_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        mode=MODE,
        strategy_whitelist=STRATEGY_WHITELIST,
        real_detect_hours=REAL_DETECT_HOURS,
        viz_window_days=VIZ_WINDOW_DAYS,
        schedule_minutes=SCHEDULE_MINUTES,
        timezone=TIMEZONE,
        sim_window_hours=SIM_WINDOW_HOURS,
        sim_step_hours=SIM_STEP_HOURS
    )
    
    # Validate required settings
    if not settings.telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required in local_settings.py")
    if not settings.telegram_chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is required in local_settings.py")
    
    return settings

def get_async_database_uri():
    """Get async database URI for SQLAlchemy"""
    assert SQLALCHEMY_DATABASE_URI

    # Use provided URI, replace pymysql with asyncmy for async support
    return SQLALCHEMY_DATABASE_URI.replace('+pymysql', '+asyncmy')

# =============================================================================
# IMPORT LOCAL SETTINGS (MUST BE AT THE END)
# =============================================================================
# Import user-specific settings to override the None values above
try:
    # Try to import from current directory first
    from local_settings import *
    print("Local settings imported successfully")
except ImportError:
    # If not found, try to import from the peak_detection directory
    import sys
    import os
    import importlib.util
    
    # Get the peak_detection directory path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    local_settings_path = os.path.join(current_file_dir, 'local_settings.py')
    
    if os.path.exists(local_settings_path):
        try:
            spec = importlib.util.spec_from_file_location("local_settings", local_settings_path)
            local_settings_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_settings_module)
            
            # Import all variables from local_settings
            for attr_name in dir(local_settings_module):
                if not attr_name.startswith('_'):
                    globals()[attr_name] = getattr(local_settings_module, attr_name)
            
            print("Local settings imported successfully")
        except Exception as e:
            print(f"ERROR importing local_settings: {e}")
    else:
        print("WARNING: local_settings.py not found. Please create it with your credentials.")
except Exception as e:
    print(f"ERROR importing local_settings: {e}")
