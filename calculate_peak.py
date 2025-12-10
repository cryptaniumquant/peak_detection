import os
import pandas as pd
import numpy as np
import logging
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob

logger = logging.getLogger(__name__)

def process_df(df_in: pd.DataFrame, absolute_threshold: float | None = None) -> pd.DataFrame | None:
    """
    In-memory version of process_file: takes a DataFrame with datetime index
    and columns including 'backtest_Value' and optional 'real_Value'.
    Resamples to 1h, computes peak/rebalance signals and helper columns,
    returns resulting DataFrame instead of writing CSV.
    """
    if df_in is None or df_in.empty:
        return None
    df = df_in.copy()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
        except Exception:
            return None
    # Resample hourly, use max
    df = df.resample('1h').max()
    columns_to_use = []
    if 'backtest_Value' in df.columns:
        columns_to_use.append('backtest_Value')
    if 'real_Value' in df.columns:
        columns_to_use.append('real_Value')
    if not columns_to_use:
        return None
    df_out = df[columns_to_use].copy()
    if 'backtest_Value' not in df_out.columns:
        return None
    combined_series = df_out['backtest_Value'].copy()
    if 'real_Value' in df_out.columns:
        real_valid_mask = ~df_out['real_Value'].isna()
        combined_series[real_valid_mask] = df_out['real_Value'][real_valid_mask]
    df_out['combined_Value'] = combined_series
    series = combined_series

    peak_detected = pd.Series(False, index=series.index)
    rebalance_point = pd.Series(False, index=series.index)
    window_length = 25
    polyorder = 1
    QUANTILE_LEVEL = 0.01
    if len(series) >= window_length:
        # Initialize output arrays
        smoothed_pnl = np.full(len(series), np.nan)
        derivative = np.full(len(series), np.nan)
        second_derivative = np.full(len(series), np.nan)
        
        # Apply Savitzky-Golay filter point by point using only past data
        # For each point i, use window [i-window_length+1 : i+1] (last 25 hours including current)
        for i in range(window_length - 1, len(series)):
            # Extract window of past data (including current point)
            window_data = series.iloc[i - window_length + 1:i + 1].values
            
            # Apply Savitzky-Golay to this window
            window_smoothed = savgol_filter(window_data, window_length=window_length, polyorder=polyorder)
            
            # Take the last (most recent) value from the smoothed window
            smoothed_pnl[i] = window_smoothed[-1]
        
        # Calculate derivatives using central difference where possible, forward/backward at edges
        for i in range(window_length, len(series)):
            if i == window_length:
                # Forward difference for first point
                if i + 1 < len(smoothed_pnl) and not np.isnan(smoothed_pnl[i]) and not np.isnan(smoothed_pnl[i+1]):
                    derivative[i] = smoothed_pnl[i+1] - smoothed_pnl[i]
            elif i == len(series) - 1:
                # Backward difference for last point
                if not np.isnan(smoothed_pnl[i-1]) and not np.isnan(smoothed_pnl[i]):
                    derivative[i] = smoothed_pnl[i] - smoothed_pnl[i-1]
            else:
                # Central difference for middle points
                if i + 1 < len(smoothed_pnl) and not np.isnan(smoothed_pnl[i-1]) and not np.isnan(smoothed_pnl[i+1]):
                    derivative[i] = (smoothed_pnl[i+1] - smoothed_pnl[i-1]) / 2.0
        
        # Calculate second derivatives using central difference where possible, forward/backward at edges
        for i in range(window_length + 1, len(series)):
            if i == window_length + 1:
                # Forward difference for first point
                if i + 1 < len(derivative) and not np.isnan(derivative[i]) and not np.isnan(derivative[i+1]):
                    second_derivative[i] = derivative[i+1] - derivative[i]
            elif i == len(series) - 1:
                # Backward difference for last point
                if not np.isnan(derivative[i-1]) and not np.isnan(derivative[i]):
                    second_derivative[i] = derivative[i] - derivative[i-1]
            else:
                # Central difference for middle points
                if i + 1 < len(derivative) and not np.isnan(derivative[i-1]) and not np.isnan(derivative[i+1]):
                    second_derivative[i] = (derivative[i+1] - derivative[i-1]) / 2.0
        
        df_out['smoothed_pnl'] = smoothed_pnl
        df_out['derivative'] = derivative
        df_out['second_derivative'] = second_derivative
        # Build threshold series: absolute value if provided, else default -100
        quantile_threshold = pd.Series(np.nan, index=series.index)
        if absolute_threshold is not None:
            # Fill from the point we have derivatives defined
            quantile_threshold.iloc[window_length:] = float(absolute_threshold)
        else:
            # Use default threshold of -100 if no absolute threshold provided
            quantile_threshold.iloc[window_length:] = -100.0
        
        # Peak detection using properly calculated derivatives
        for i in range(window_length + 2, len(series) - 1):
            if (not np.isnan(derivative[i-1]) and 
                not np.isnan(derivative[i+1]) and
                not np.isnan(second_derivative[i]) and
                not np.isnan(quantile_threshold.iloc[i])):
                
                is_peak = derivative[i - 1] > 0 and derivative[i + 1] < 0
                lookback_hours = 24
                start_lookback = max(window_length + 1, i - lookback_hours)
                is_below_quantile = False
                
                for j in range(start_lookback, i + 1):
                    if (not np.isnan(second_derivative[j]) and 
                        not np.isnan(quantile_threshold.iloc[j]) and
                        second_derivative[j] < quantile_threshold.iloc[j]):
                        is_below_quantile = True
                        break
                
                if is_peak and is_below_quantile:
                    peak_detected.iloc[i] = True
                    if i + 1 < len(rebalance_point):
                        rebalance_point.iloc[i + 1] = True
        df_out['quantile_threshold'] = quantile_threshold
    df_out['peak_detected'] = peak_detected
    df_out['rebalance_point'] = rebalance_point
    DEFAULT_COOLDOWN = 24
    RECOVERY_PERIOD = 24
    weight = []
    cooldown = 0
    recovery_period = 0
    for rp in df_out['rebalance_point']:
        if cooldown > 0:
            weight.append(0)
            cooldown -= 1
            if cooldown == 0:
                recovery_period = RECOVERY_PERIOD
        elif recovery_period > 0:
            weight.append(1)
            recovery_period -= 1
            continue
        elif rp:
            cooldown = DEFAULT_COOLDOWN - 1
            weight.append(0)
        else:
            weight.append(1)
    df_out['weight'] = weight
    return df_out
def process_file(filepath, output_dir):

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    # Ресемплируем к часовому таймфрейму, берём максимум за каждый час
    df = df.resample('1h').max()
    # Оставляем только нужные столбцы, если они есть
    columns_to_use = []
    if 'backtest_Value' in df.columns:
        columns_to_use.append('backtest_Value')
    if 'real_Value' in df.columns:
        columns_to_use.append('real_Value')
    if not columns_to_use:
        logger.warning(f"No 'backtest_Value' or 'real_Value' in {filepath}, skipping.")
        return
    df_out = df[columns_to_use].copy()
    
    # Создаем комбинированную серию из бэктеста и реальных торгов
    if 'backtest_Value' not in df_out.columns:
        logger.warning(f"No 'backtest_Value' in {filepath}, skipping.")
        return
    
    # Создаем комбинированную серию
    combined_series = df_out['backtest_Value'].copy()
    
    # Если есть реальные данные, добавляем их в комбинированную серию
    if 'real_Value' in df_out.columns:
        # Заменяем значения бэктеста на реальные там, где реальные данные не NaN
        real_valid_mask = ~df_out['real_Value'].isna()
        combined_series[real_valid_mask] = df_out['real_Value'][real_valid_mask]
    
    # Добавляем комбинированную серию в датафрейм
    df_out['combined_Value'] = combined_series
    
    # Используем комбинированную серию для расчета пиков
    series = combined_series

    
    # Инициализируем серии для отслеживания пиков и точек ребалансировки
    peak_detected = pd.Series(False, index=series.index)
    rebalance_point = pd.Series(False, index=series.index)
    
    # Параметры для фильтра Савицкого-Голея
    window_length = 25  # Должно быть нечетным числом 
    polyorder = 1  # Порядок полинома

    # Применяем фильтр Савицкого-Голея непосредственно к PnL (backtest_Value)
    # Проверяем, достаточно ли данных для применения фильтра
    if len(series) >= window_length:
        # Initialize output arrays
        smoothed_pnl = np.full(len(series), np.nan)
        derivative = np.full(len(series), np.nan)
        second_derivative = np.full(len(series), np.nan)
        
        # Apply Savitzky-Golay filter point by point using only past data
        # For each point i, use window [i-window_length+1 : i+1] (last 25 hours including current)
        for i in range(window_length - 1, len(series)):
            # Extract window of past data (including current point)
            window_data = series.iloc[i - window_length + 1:i + 1].values
            
            # Apply Savitzky-Golay to this window
            window_smoothed = savgol_filter(window_data, window_length=window_length, polyorder=polyorder)
            
            # Take the last (most recent) value from the smoothed window
            smoothed_pnl[i] = window_smoothed[-1]
        
        # Calculate derivatives using central difference where possible, forward/backward at edges
        for i in range(window_length, len(series)):
            if i == window_length:
                # Forward difference for first point
                if i + 1 < len(smoothed_pnl) and not np.isnan(smoothed_pnl[i]) and not np.isnan(smoothed_pnl[i+1]):
                    derivative[i] = smoothed_pnl[i+1] - smoothed_pnl[i]
            elif i == len(series) - 1:
                # Backward difference for last point
                if not np.isnan(smoothed_pnl[i-1]) and not np.isnan(smoothed_pnl[i]):
                    derivative[i] = smoothed_pnl[i] - smoothed_pnl[i-1]
            else:
                # Central difference for middle points
                if i + 1 < len(smoothed_pnl) and not np.isnan(smoothed_pnl[i-1]) and not np.isnan(smoothed_pnl[i+1]):
                    derivative[i] = (smoothed_pnl[i+1] - smoothed_pnl[i-1]) / 2.0
        
        # Calculate second derivatives using central difference where possible, forward/backward at edges
        for i in range(window_length + 1, len(series)):
            if i == window_length + 1:
                # Forward difference for first point
                if i + 1 < len(derivative) and not np.isnan(derivative[i]) and not np.isnan(derivative[i+1]):
                    second_derivative[i] = derivative[i+1] - derivative[i]
            elif i == len(series) - 1:
                # Backward difference for last point
                if not np.isnan(derivative[i-1]) and not np.isnan(derivative[i]):
                    second_derivative[i] = derivative[i] - derivative[i-1]
            else:
                # Central difference for middle points
                if i + 1 < len(derivative) and not np.isnan(derivative[i-1]) and not np.isnan(derivative[i+1]):
                    second_derivative[i] = (derivative[i+1] - derivative[i-1]) / 2.0
        
        # Добавляем отладочную информацию
        df_out['smoothed_pnl'] = smoothed_pnl
        df_out['derivative'] = derivative
        df_out['second_derivative'] = second_derivative
        
        # Создаем серию для хранения порога квантиля
        quantile_threshold = pd.Series(np.nan, index=series.index)
        
        # Build threshold series: absolute value if provided, else default -100
        if absolute_threshold is not None:
            # Fill from the point we have derivatives defined
            quantile_threshold.iloc[window_length:] = float(absolute_threshold)
        else:
            # Use default threshold of -100 if no absolute threshold provided
            quantile_threshold.iloc[window_length:] = -100.0
        
        # Peak detection using properly calculated derivatives
        for i in range(window_length + 2, len(series) - 1):
            if (not np.isnan(derivative[i-1]) and 
                not np.isnan(derivative[i+1]) and
                not np.isnan(second_derivative[i]) and
                not np.isnan(quantile_threshold.iloc[i])):
                
                is_peak = derivative[i - 1] > 0 and derivative[i + 1] < 0
                lookback_hours = 24
                start_lookback = max(window_length + 1, i - lookback_hours)
                is_below_quantile = False
                
                for j in range(start_lookback, i + 1):
                    if (not np.isnan(second_derivative[j]) and 
                        not np.isnan(quantile_threshold.iloc[j]) and
                        second_derivative[j] < quantile_threshold.iloc[j]):
                        is_below_quantile = True
                        break
                
                if is_peak and is_below_quantile:
                    peak_detected.iloc[i] = True
                    if i + 1 < len(rebalance_point):
                        rebalance_point.iloc[i + 1] = True
        
        # Сохраняем пороговый квантиль для визуализации
        df_out['quantile_threshold'] = quantile_threshold
    
    df_out['peak_detected'] = peak_detected
    df_out['rebalance_point'] = rebalance_point
    
    # Логика веса: 1 по умолчанию, 0 на cooldown часов после rebalance_point
    DEFAULT_COOLDOWN = 24  # стандартный cooldown (часы)
    RECOVERY_PERIOD = 24  # период после возвращения веса, когда не проверяем пики (часы)
    weight = []
    cooldown = 0
    recovery_period = 0  # период после восстановления веса
    for i, rp in enumerate(df_out['rebalance_point']):
        if cooldown > 0:
            weight.append(0)
            cooldown -= 1
            # Когда cooldown заканчивается, начинаем recovery_period
            if cooldown == 0:
                recovery_period = RECOVERY_PERIOD
        elif recovery_period > 0:
            # В период восстановления вес = 1 и не проверяем пики
            weight.append(1)
            recovery_period -= 1
            # Игнорируем сигналы ребалансировки в период восстановления
            continue
        elif rp:
            # Устанавливаем стандартный период охлаждения
            cooldown = DEFAULT_COOLDOWN - 1
            weight.append(0)
        else:
            weight.append(1)
    df_out['weight'] = weight
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename.replace('.csv', '_joined_with_peak.csv'))
    df_out.to_csv(output_path)
    logger.debug(f"Saved: {output_path}")

def main():
    input_dir = r'c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\data_prep\adjusted_data\trend\complete'
    # Сохраняем файлы в директорию, где находится скрипт
    output_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(input_dir):
        if fname.endswith('.csv') and fname.startswith('merged_'):
            process_file(os.path.join(input_dir, fname), output_dir)

if __name__ == '__main__':
    main()
