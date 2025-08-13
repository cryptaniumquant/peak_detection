import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter

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
        smoothed_pnl = savgol_filter(series, window_length=window_length, polyorder=polyorder)
        derivative = np.gradient(smoothed_pnl)
        second_derivative = np.gradient(derivative)
        df_out['smoothed_pnl'] = smoothed_pnl
        df_out['derivative'] = derivative
        df_out['second_derivative'] = second_derivative
        # Build threshold series: absolute value if provided, else rolling quantile
        quantile_threshold = pd.Series(np.nan, index=series.index)
        if absolute_threshold is not None:
            # Fill from the point we have derivatives defined
            quantile_threshold.iloc[window_length:] = float(absolute_threshold)
        else:
            for i in range(window_length, len(series)):
                lookback_window = second_derivative[max(0, i - 24):i + 1]
                quantile_threshold.iloc[i] = np.quantile(lookback_window, QUANTILE_LEVEL)
        for i in range(1, len(derivative) - 1):
            is_peak = derivative[i - 1] > 0 and derivative[i + 1] < 0
            lookback_hours = 24
            start_lookback = max(0, i - lookback_hours)
            is_below_quantile = False
            for j in range(start_lookback, i + 1):
                if second_derivative[j] < quantile_threshold.iloc[j]:
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
        print(f"No 'backtest_Value' or 'real_Value' in {filepath}, skipping.")
        return
    df_out = df[columns_to_use].copy()
    
    # Создаем комбинированную серию из бэктеста и реальных торгов
    if 'backtest_Value' not in df_out.columns:
        print(f"No 'backtest_Value' in {filepath}, skipping.")
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
    
    # Параметры для расчета квантиля второй производной
    QUANTILE_LEVEL = 0.01  # 5% квантиль для выявления значимых изменений
    
    # Применяем фильтр Савицкого-Голея непосредственно к PnL (backtest_Value)
    # Проверяем, достаточно ли данных для применения фильтра
    if len(series) >= window_length:
        # Применяем фильтр Савицкого-Голея для сглаживания PnL
        smoothed_pnl = savgol_filter(series, window_length=window_length, polyorder=polyorder)
        
        # Рассчитываем первую производную (скорость изменения PnL)
        # Положительная производная означает рост PnL, отрицательная - падение
        derivative = np.gradient(smoothed_pnl)
        
        # Рассчитываем вторую производную (ускорение/замедление роста PnL)
        # Отрицательная вторая производная означает замедление роста или ускорение падения
        second_derivative = np.gradient(derivative)
        
        # Добавляем отладочную информацию
        df_out['smoothed_pnl'] = smoothed_pnl
        df_out['derivative'] = derivative
        df_out['second_derivative'] = second_derivative
        
        # Создаем серию для хранения порога квантиля
        quantile_threshold = pd.Series(np.nan, index=series.index)
        
        # Рассчитываем глобальный порог квантиля для второй производной
        # Используем только отрицательные значения второй производной
        second_derivative_series = pd.Series(second_derivative, index=series.index)
        
        # Получаем все отрицательные значения второй производной
        neg_values = second_derivative_series[second_derivative_series < 0]
        
        # Рассчитываем квантиль на всем периоде
        if len(neg_values) > 0:
            global_threshold = neg_values.quantile(QUANTILE_LEVEL)
            # Заполняем все значения одним и тем же порогом
            quantile_threshold.iloc[:] = global_threshold
        else:
            # Если нет отрицательных значений, используем очень низкое значение
            quantile_threshold.iloc[:] = -np.inf
        
        # Находим пики, где:
        # 1. Первая производная меняет знак с положительной на отрицательную (точка разворота тренда)
        # 2. Вторая производная была ниже глобального квантиля в последние 24 часа
        for i in range(1, len(derivative)-1):
            
            # Проверка смены знака первой производной (с + на -)
            # Это означает, что PnL перестал расти и начал падать
            is_peak = derivative[i-1] > 0 and derivative[i+1] < 0
            
            # Проверка, что вторая производная была ниже порогового квантиля в последние 24 часа
            # Это означает значительное замедление роста или ускорение падения в недавнем прошлом
            # Проверяем последние 24 часа (24 точки при часовом таймфрейме)
            lookback_hours = 24
            start_lookback = max(0, i - lookback_hours)
            # Проверяем, была ли вторая производная ниже порога в любой точке за последние 24 часа
            is_below_quantile = False
            for j in range(start_lookback, i + 1):
                if second_derivative[j] < quantile_threshold.iloc[j]:
                    is_below_quantile = True
                    break
            
            # Генерируем сигнал пика, если все условия выполнены
            if is_peak and is_below_quantile:
                peak_detected.iloc[i] = True
                rebalance_point.iloc[i+1] = True  # точка, где мы заметили пик
        
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
    print(f"Saved: {output_path}")

def main():
    input_dir = r'c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\data_prep\adjusted_data\trend\complete'
    # Сохраняем файлы в директорию, где находится скрипт
    output_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(input_dir):
        if fname.endswith('.csv') and fname.startswith('merged_'):
            process_file(os.path.join(input_dir, fname), output_dir)

if __name__ == '__main__':
    main()
