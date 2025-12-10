"""
Logging configuration for Peak Detection Trading Bot
Provides console logging and Telegram error notifications
"""
import logging
import asyncio
from typing import Optional
from telegram import Bot
from datetime import datetime, timezone


class UTCFormatter(logging.Formatter):
    """Custom formatter that uses UTC timezone for timestamps"""
    
    def formatTime(self, record, datefmt=None):
        """Override formatTime to use UTC timezone"""
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')


class TelegramErrorHandler(logging.Handler):
    """Custom logging handler that sends ERROR+ messages to Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self.setLevel(logging.ERROR)
        
    def emit(self, record):
        """Send log record to Telegram if it's ERROR level or higher"""
        try:            
            # Skip network-related errors from being sent to Telegram
            if self._is_network_error(record):
                return
                
            if not self.bot:
                self.bot = Bot(token=self.bot_token)
            
            # Format the error message
            log_entry = self.format(record)
            message = f"🚨 Bot Error:\n```\n{log_entry}\n```"
            
            # Send to Telegram (async)
            asyncio.create_task(self._send_to_telegram(message))
            
        except Exception:
            # Don't let logging errors break the application
            pass
    def _is_network_error(self, record):
        """Check if the log record is a network-related error that should not be sent to Telegram"""
        error_message = record.getMessage().lower()
        network_error_keywords = [
            'httpx.readerror',
            'httpx.connecterror',
            'httpx.timeouterror',
            'error while getting updates',
            'network error',
            'connection error',
            'timeout error',
            'read timeout',
            'connect timeout',
            'pool timeout'
        ]
        
        # Check if this is from telegram.ext.Updater (the polling mechanism)
        if record.name == 'telegram.ext.Updater':
            return True
            
        # Check for network error keywords in the message
        return any(keyword in error_message for keyword in network_error_keywords)
        
    async def _send_to_telegram(self, message: str):
        """Send message to Telegram asynchronously"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id, 
                text=message, 
                parse_mode='Markdown'
            )
        except Exception:
            # Silently fail if Telegram sending fails
            pass


def setup_logging(log_level: str = 'INFO', telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        telegram_token: Telegram bot token for error notifications
        telegram_chat_id: Telegram chat ID for error notifications
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Console formatter with UTC timezone
    console_formatter = UTCFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Telegram error handler (if credentials provided)
    if telegram_token and telegram_chat_id:
        try:
            telegram_handler = TelegramErrorHandler(telegram_token, telegram_chat_id)
            telegram_formatter = UTCFormatter(
                '%(asctime)s - %(name)s - %(levelname)s\n%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S UTC'
            )
            telegram_handler.setFormatter(telegram_formatter)
            logger.addHandler(telegram_handler)
        except Exception as e:
            # If Telegram handler setup fails, just log to console
            logger.warning(f"Failed to setup Telegram error handler: {e}")
    
    # Set specific loggers to appropriate levels
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    
    logger.debug(f"Logging setup complete - Level: {log_level}")
    if telegram_token and telegram_chat_id:
        logger.info("Telegram error notifications enabled")
