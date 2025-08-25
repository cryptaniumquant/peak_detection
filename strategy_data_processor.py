import pandas as pd
import numpy as np
import logging
from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import csv
from datetime import datetime, timedelta, timezone
import os
import asyncio

# Import database connection parameters from unified config
# Use absolute import to avoid conflicts with bot_service/config.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from the main config module in this directory
import importlib.util
config_path = os.path.join(current_dir, 'config.py')
spec = importlib.util.spec_from_file_location("main_config", config_path)
main_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_config)

get_async_database_uri = main_config.get_async_database_uri

logger = logging.getLogger(__name__)

# Global async engine and session maker
_async_engine = None
_async_session_maker = None

def set_time_zone(dbapi_connection, connection_record):
    """Set MySQL session timezone to UTC"""
    cursor = dbapi_connection.cursor()
    cursor.execute("SET time_zone = 'UTC';")
    cursor.close()

def get_async_engine():
    """Get or create async database engine"""
    global _async_engine, _async_session_maker
    if _async_engine is None:
        database_uri = get_async_database_uri()
        _async_engine = create_async_engine(database_uri, echo=False)
        
        # Set timezone to UTC for all connections
        event.listen(_async_engine.sync_engine, "connect", set_time_zone)
        
        _async_session_maker = async_sessionmaker(_async_engine, expire_on_commit=False, autoflush=False)
    return _async_engine, _async_session_maker


def get_csv_strategies(csv_path):
    """Get strategy names from CSV file"""
    strategies = []
    try:
        with open(csv_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                strategies.append(row['strategy'])
        logger.info(f"Found {len(strategies)} strategies in CSV file")
        return strategies
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None

async def get_strategy_data_async(strategy_name, days=30):
    """
    Get data for a specific strategy from the database (async version)
    Args:
        strategy_name: Name of the strategy
        days: Number of days of data to retrieve (default: 30, to get a month of data for more comprehensive analysis)
    Returns:
        DataFrame with the strategy data
    """
    try:
        _, async_session_maker = get_async_engine()
        
        # Calculate the start time (current time - days) in UTC
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Format times for SQL query (already in UTC)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Query to get data for the strategy
        query = text("""
            SELECT 
                analyst, 
                res_time, 
                virtual_unrealized_pnl
            FROM 
                analyst_result
            WHERE 
                analyst = :strategy AND
                res_time BETWEEN :start_time AND :end_time AND
                virtual_unrealized_pnl != 0
            ORDER BY 
                res_time
        """)
        
        async with async_session_maker() as session:
            # Execute the query with parameters
            result = await session.execute(query, {
                'strategy': strategy_name,
                'start_time': start_time_str,
                'end_time': end_time_str
            })
            
            # Convert to list of mappings
            rows = list(result.mappings().all())
            
        if not rows:
            logger.warning(f"No non-zero data found for strategy {strategy_name}")
            return None
        
        # Convert to DataFrame
        df_data = [{'strategy': row['analyst'], 'timestamp': row['res_time'], 'unrealized_pnl': row['virtual_unrealized_pnl']} for row in rows]
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Retrieved {len(df)} non-zero records for strategy {strategy_name}")
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving data for strategy {strategy_name}: {e}")
        return None

def get_strategy_data(strategy_name, days=30):
    """
    Get data for a specific strategy from the database (sync wrapper for async function)
    Args:
        strategy_name: Name of the strategy
        days: Number of days of data to retrieve (default: 30, to get a month of data for more comprehensive analysis)
    Returns:
        DataFrame with the strategy data
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to create a new task
            # This is a fallback - ideally callers should use the async version directly
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_strategy_data_async(strategy_name, days))
                return future.result()
        else:
            # No running loop, we can use asyncio.run
            return asyncio.run(get_strategy_data_async(strategy_name, days))
    except RuntimeError:
        # No event loop, use asyncio.run
        return asyncio.run(get_strategy_data_async(strategy_name, days))

def aggregate_to_hourly(df, strategy_name=None):
    """
    Aggregate minute data to hourly data by taking the maximum value for each hour
    Args:
        df: DataFrame with minute-level data
        strategy_name: Name of the strategy for logging purposes
    Returns:
        DataFrame with hourly data
    """
    if df is None or df.empty:
        return None
    
    # Resample to hourly data, taking the maximum value for each hour
    hourly_df = df.resample('1h').max()
    
    # Rename column to match the expected format for the peak detection algorithm
    hourly_df.rename(columns={'unrealized_pnl': 'backtest_Value'}, inplace=True)
    
    # Remove hours with NaN values (no data for that hour)
    hourly_df = hourly_df.dropna()
    
    # Check if we have enough consecutive hours of data
    if len(hourly_df) < 25:  # Need at least 25 hours for the Savitzky-Golay filter
        strategy_info = f" for strategy {strategy_name}" if strategy_name else ""
        logger.warning(f"Not enough hourly data points after filtering: {len(hourly_df)}{strategy_info}")
        return None
    
    # Find the longest continuous segment with at least 25 hours of data
    continuous_segments = []
    current_segment = []
    
    # Convert index to list for easier processing
    timestamps = hourly_df.index.tolist()
    
    for i in range(len(timestamps)):
        if i == 0 or (timestamps[i] - timestamps[i-1]).total_seconds() == 3600:  # 1 hour difference
            current_segment.append(i)
        else:
            if len(current_segment) >= 25:  # Only keep segments with at least 25 hours
                continuous_segments.append(current_segment)
            current_segment = [i]
    
    # Add the last segment if it's long enough
    if len(current_segment) >= 25:
        continuous_segments.append(current_segment)
    
    if not continuous_segments:
        strategy_info = f" for strategy {strategy_name}" if strategy_name else ""
        logger.warning(f"No continuous segments with at least 25 hours of data found{strategy_info}")
        return None
    
    # Find the longest segment
    longest_segment = max(continuous_segments, key=len)
    strategy_info = f" for strategy {strategy_name}" if strategy_name else ""
    logger.info(f"Found continuous segment with {len(longest_segment)} hours of data{strategy_info}")
    
    # Extract the longest segment
    segment_indices = [timestamps[i] for i in longest_segment]
    filtered_df = hourly_df.loc[segment_indices]
    
    return filtered_df

def process_strategy(strategy_name, output_dir, days=30):
    """
    Process data for a specific strategy and save it to a CSV file
    Args:
        strategy_name: Name of the strategy
        output_dir: Directory to save the output file
        hours: Number of hours of data to retrieve
    Returns:
        Path to the output file if successful, None otherwise
    """
    # Get data for the strategy
    df = get_strategy_data(strategy_name, days)
    if df is None:
        return None
    
    # Aggregate to hourly data
    hourly_df = aggregate_to_hourly(df, strategy_name)
    if hourly_df is None:
        return None
    
    # Create output file path
    output_file = os.path.join(output_dir, f"merged_{strategy_name}.csv")
    
    # Save to CSV
    hourly_df.to_csv(output_file)
    logger.info(f"Saved data for strategy {strategy_name} to {output_file}")
    
    return output_file

async def process_strategy_df_async(strategy_name, days=30):
    """
    In-memory version: returns hourly dataframe with expected columns for peak detection (async).
    """
    df = await get_strategy_data_async(strategy_name, days)
    if df is None:
        return None
    hourly_df = aggregate_to_hourly(df, strategy_name)
    return hourly_df

def process_strategy_df(strategy_name, days=30):
    """
    In-memory version: returns hourly dataframe with expected columns for peak detection.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, process_strategy_df_async(strategy_name, days))
                return future.result()
        else:
            return asyncio.run(process_strategy_df_async(strategy_name, days))
    except RuntimeError:
        return asyncio.run(process_strategy_df_async(strategy_name, days))

async def get_strategy_data_hours_async(strategy_name: str, hours: int = 25):
    """Get data for a specific strategy limited by hours lookback (async version)."""
    try:
        _, async_session_maker = get_async_engine()
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        query = text("""
            SELECT analyst, res_time, virtual_unrealized_pnl
            FROM analyst_result
            WHERE analyst = :strategy
              AND res_time BETWEEN :start_time AND :end_time
              AND virtual_unrealized_pnl != 0
            ORDER BY res_time
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(query, {
                'strategy': strategy_name,
                'start_time': start_time_str,
                'end_time': end_time_str,
            })
            rows = list(result.mappings().all())
            
        if not rows:
            logger.warning(f"No non-zero data found (last {hours}h) for strategy {strategy_name}")
            return None
            
        df_data = [{'strategy': row['analyst'], 'timestamp': row['res_time'], 'unrealized_pnl': row['virtual_unrealized_pnl']} for row in rows]
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving last {hours}h for {strategy_name}: {e}")
        return None

def get_strategy_data_hours(strategy_name: str, hours: int = 25):
    """Get data for a specific strategy limited by hours lookback (sync wrapper)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_strategy_data_hours_async(strategy_name, hours))
                return future.result()
        else:
            return asyncio.run(get_strategy_data_hours_async(strategy_name, hours))
    except RuntimeError:
        return asyncio.run(get_strategy_data_hours_async(strategy_name, hours))

async def process_strategy_df_hours_async(strategy_name: str, hours: int) -> pd.DataFrame | None:
    """In-memory: fetch last <hours>, aggregate hourly, return DF (async version)."""
    df = await get_strategy_data_hours_async(strategy_name, hours)
    if df is None:
        return None
    return aggregate_to_hourly(df, strategy_name)

def process_strategy_df_hours(strategy_name: str, hours: int) -> pd.DataFrame | None:
    """In-memory: fetch last <hours>, aggregate hourly, return DF."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, process_strategy_df_hours_async(strategy_name, hours))
                return future.result()
        else:
            return asyncio.run(process_strategy_df_hours_async(strategy_name, hours))
    except RuntimeError:
        return asyncio.run(process_strategy_df_hours_async(strategy_name, hours))

def process_all_strategies(csv_path, output_dir):
    """
    Process all strategies from the CSV file
    Args:
        csv_path: Path to the CSV file with strategy names
        output_dir: Directory to save the output files
    """
    # Get strategy names from CSV
    strategies = get_csv_strategies(csv_path)
    if not strategies:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each strategy
    processed_files = []
    for strategy in strategies:
        output_file = process_strategy(strategy, output_dir)
        if output_file:
            processed_files.append(output_file)
    
    logger.info(f"Processed {len(processed_files)} strategies")
    return processed_files

def main():
    # Path to the CSV file with strategy names
    csv_path = r"c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\FULL_FUND_DATA\peak_detection\strategy_quantile_values.csv"
    
    # Output directory for the processed files
    output_dir = r"c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\FULL_FUND_DATA\peak_detection\processed_data"
    
    # Process all strategies
    processed_files = process_all_strategies(csv_path, output_dir)
    
    # Run peak detection on the processed files
    if processed_files:
        logger.info("Running peak detection on processed files...")
        from calculate_peak import process_file
        
        for file_path in processed_files:
            process_file(file_path, output_dir)

if __name__ == "__main__":
    main()
