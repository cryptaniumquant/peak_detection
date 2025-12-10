"""
Unified configuration file for Peak Detection system.
Contains all constants and default values.
User-specific credentials are imported from local_settings.py
"""
import os
import os
import json
import logging
import csv
import asyncio
from dataclasses import dataclass
from typing import List, Dict
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

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
MODE = 'dummy'  # 'real', 'simulate', or 'dummy'

# Strategy filtering
STRATEGY_WHITELIST = []  # Empty list means all strategies

# Time windows
REAL_DETECT_HOURS = 28  # Hours of data to fetch for signal detection
VIZ_WINDOW_DAYS = 7     # Days of data for visualization

# Dummy mode configuration
DUMMY_PNL_THRESHOLD = 10000.0  # Dollar threshold for unrealized - realized PnL difference
DUMMY_COOLDOWN_HOURS = 24      # Hours to wait after rebalance before next signal

# Scheduling (APScheduler cron-based)
SCHEDULE_MINUTES = 60   # Legacy setting - now uses cron scheduling

# Scheduler job configuration (only schedule parameters - others are hardcoded)
SCHEDULER_JOB_CONFIG = {
    'trigger': 'cron',
    'minute': 15,           # Run at minute 0 of each hour (beginning of hour)
    'hour': '*',           # Every hour
}

# Timezone
TIMEZONE = 'Europe/Moscow'

# Logging configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Simulation parameters
SIM_WINDOW_HOURS = 168  # 7 days
SIM_STEP_HOURS = 1

# Default threshold for analysts from database
DEFAULT_ABSOLUTE_THRESHOLD = -100.0

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
# Пока предлагаю систему в 3 уровня:
# 1. Частое определение пиков с квантилем 100
# 2. Среднее определение пиков с квантилем в 250
# 3. Редкое определение пиков с квантилем в 500
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
async def get_analysts_from_database() -> List[str]:
    """
    Query database to get all enabled analyst codes from management table.
    """
    try:
        db_uri = get_async_database_uri()
        engine = create_async_engine(db_uri)
        async_session = sessionmaker(engine, class_=AsyncSession)
        
        async with async_session() as session:
            query = sa.text("""
                select a.code as analyst 
                from analyst a
                where a.state = 'enabled' and a.typ='plan'
                and exists (select 'x' from management m where m.analyst=a.code and m.state='enabled')
            """)
            result = await session.execute(query)
            analysts = [row.analyst for row in result.fetchall()]
            logger.debug(f"Found {len(analysts)} enabled analysts from database")
            return analysts
    except Exception as e:
        logger.error(f"Error querying analysts from database: {e}")
        return []

async def load_strategy_thresholds_async() -> Dict[str, float]:
    """
    Async version of load_strategy_thresholds for use within existing event loops.
    """
    thresholds = {}
    
    # Step 1: Get analysts from database and set default threshold
    try:
        analysts = await get_analysts_from_database()
        # Set default threshold for all analysts
        for analyst in analysts:
            thresholds[analyst] = DEFAULT_ABSOLUTE_THRESHOLD
        logger.debug(f"Set default threshold {DEFAULT_ABSOLUTE_THRESHOLD} for {len(analysts)} analysts")
    except Exception as e:
        logger.error(f"Error loading analysts from database: {e}")
    
    # Step 2: Override with JSON values if file exists
    if os.path.exists(STRATEGY_THRESHOLDS_JSON):
        try:
            with open(STRATEGY_THRESHOLDS_JSON, 'r') as f:
                json_thresholds = json.load(f)
                for strategy, threshold in json_thresholds.items():
                    thresholds[strategy] = threshold
                logger.debug(f"Applied {len(json_thresholds)} overrides from JSON file")
        except Exception as e:
            logger.error(f"Error loading strategy thresholds from JSON: {e}")
    
    # Step 3: Override with CSV values if file exists
    if os.path.exists(STRATEGY_QUANTILE_CSV):
        try:
            with open(STRATEGY_QUANTILE_CSV, 'r') as f:
                reader = csv.DictReader(f)
                csv_count = 0
                for row in reader:
                    strategy = row.get('strategy')
                    # Use quantile_value (negative) as absolute threshold
                    threshold = float(row.get('quantile_value', 0))
                    if strategy:
                        thresholds[strategy] = threshold
                        csv_count += 1
                logger.debug(f"Applied {csv_count} overrides from CSV file")
        except Exception as e:
            logger.error(f"Error loading strategy thresholds from CSV: {e}")
    
    logger.debug(f"Final threshold configuration: {len(thresholds)} strategies")
    return thresholds

def load_strategy_thresholds() -> Dict[str, float]:
    """
    Load per-strategy absolute thresholds with new logic:
    1. Query database for all enabled analysts and set them to DEFAULT_ABSOLUTE_THRESHOLD
    2. Override with values from JSON file if they exist
    3. Override with values from CSV file if they exist
    """
    thresholds = {}
    
    # Step 1: Try to get analysts from database
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, we can't run async code synchronously
            logger.warning("Already in event loop, skipping database query for analysts")
        except RuntimeError:
            # No running loop, we can create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysts = loop.run_until_complete(get_analysts_from_database())
            loop.close()
            
            # Set default threshold for all analysts
            for analyst in analysts:
                thresholds[analyst] = DEFAULT_ABSOLUTE_THRESHOLD
            logger.debug(f"Set default threshold {DEFAULT_ABSOLUTE_THRESHOLD} for {len(analysts)} analysts")
    except Exception as e:
        logger.error(f"Error loading analysts from database: {e}")
    
    # Step 2: Override with JSON values if file exists
    if os.path.exists(STRATEGY_THRESHOLDS_JSON):
        try:
            with open(STRATEGY_THRESHOLDS_JSON, 'r') as f:
                json_thresholds = json.load(f)
                for strategy, threshold in json_thresholds.items():
                    thresholds[strategy] = threshold
                logger.debug(f"Applied {len(json_thresholds)} overrides from JSON file")
        except Exception as e:
            logger.error(f"Error loading strategy thresholds from JSON: {e}")
    
    # Step 3: Override with CSV values if file exists
    if os.path.exists(STRATEGY_QUANTILE_CSV):
        try:
            with open(STRATEGY_QUANTILE_CSV, 'r') as f:
                reader = csv.DictReader(f)
                csv_count = 0
                for row in reader:
                    strategy = row.get('strategy')
                    # Use quantile_value (negative) as absolute threshold
                    threshold = float(row.get('quantile_value', 0))
                    if strategy:
                        thresholds[strategy] = threshold
                        csv_count += 1
                logger.debug(f"Applied {csv_count} overrides from CSV file")
        except Exception as e:
            logger.error(f"Error loading strategy thresholds from CSV: {e}")
    
    logger.debug(f"Final threshold configuration: {len(thresholds)} strategies")
    return thresholds

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
    logger.debug("Local settings imported successfully")
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
            
            logger.debug("Local settings imported successfully")
        except Exception as e:
            logger.error(f"ERROR importing local_settings: {e}")
    else:
        logger.warning("local_settings.py not found. Please create it with your credentials.")
except Exception as e:
    logger.error(f"ERROR importing local_settings: {e}")
