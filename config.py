"""
Unified configuration file for Peak Detection system.
Contains all constants and default values.
User-specific credentials are imported from local_settings.py
"""
import os
import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# SQLAlchemy async database URI - set in local_settings.py
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

# Scheduler job configuration
SCHEDULER_JOB_CONFIG = {
    'trigger': 'cron',
    'minute': 0,           # Run at minute 0 of each hour (beginning of hour)
    'hour': '*',           # Every hour
    'max_instances': 1,    # Prevent overlapping runs
    'coalesce': True,      # If multiple runs are queued, run only the latest
}

# Timezone
TIMEZONE = 'Europe/Moscow'

# Logging configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

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
                logger.info(f"Loaded {len(thresholds)} strategy thresholds from JSON")
                return thresholds
        except Exception as e:
            logger.error(f"Error loading strategy thresholds from JSON: {e}")
    
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
            logger.info(f"Loaded {len(thresholds)} strategy thresholds from CSV fallback")
            return thresholds
        except Exception as e:
            logger.error(f"Error loading strategy thresholds from CSV: {e}")
    
    logger.warning("No strategy thresholds found, using dynamic quantiles")
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
    if not SQLALCHEMY_DATABASE_URI:
        raise ValueError("SQLALCHEMY_DATABASE_URI is required in local_settings.py")
    
    # Ensure async driver is used
    if '+pymysql' in SQLALCHEMY_DATABASE_URI:
        return SQLALCHEMY_DATABASE_URI.replace('+pymysql', '+asyncmy')
    elif '+asyncmy' not in SQLALCHEMY_DATABASE_URI and 'mysql://' in SQLALCHEMY_DATABASE_URI:
        return SQLALCHEMY_DATABASE_URI.replace('mysql://', 'mysql+asyncmy://')
    
    return SQLALCHEMY_DATABASE_URI

# =============================================================================
# IMPORT LOCAL SETTINGS (MUST BE AT THE END)
# =============================================================================
# Import user-specific settings to override the None values above
try:
    # Try to import from current directory first
    from local_settings import *
    logger.info("Local settings imported successfully")
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
            
            logger.info("Local settings imported successfully")
        except Exception as e:
            logger.error(f"ERROR importing local_settings: {e}")
    else:
        logger.warning("local_settings.py not found. Please create it with your credentials.")
except Exception as e:
    logger.error(f"ERROR importing local_settings: {e}")
