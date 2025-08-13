"""
Bot service configuration - delegates to unified config system
"""
import os
import sys
import importlib.util

def _get_parent_config():
    """Dynamically import parent config module to avoid circular imports"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_dir, 'config.py')
    
    spec = importlib.util.spec_from_file_location("parent_config", config_path)
    parent_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_config)
    
    return parent_config

# Get parent config module
_parent_config = _get_parent_config()

# Re-export everything from parent config
Settings = _parent_config.Settings
BASE_DIR = _parent_config.BASE_DIR
PROCESSED_DIR = _parent_config.PROCESSED_DIR
VIZ_DIR = _parent_config.VIZ_DIR
STATE_DIR = _parent_config.STATE_DIR
THRESHOLDS_DEFAULT_PATH = _parent_config.STRATEGY_THRESHOLDS_JSON
CSV_FALLBACK_PATH = _parent_config.STRATEGY_QUANTILE_CSV

def load_settings():
    """Load settings using the unified config system"""
    # Create directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    
    return _parent_config.load_settings()

def load_strategy_thresholds(path=None):
    """Load strategy thresholds using the unified config system"""
    return _parent_config.load_strategy_thresholds()
