import pandas as pd
from typing import Optional

def limit_data_to_last_points(df: pd.DataFrame, max_points: int = 500) -> pd.DataFrame:
    """
    Ограничивает DataFrame последними max_points точками
    
    Args:
        df: DataFrame для ограничения
        max_points: Максимальное количество точек (по умолчанию 500)
    
    Returns:
        DataFrame с ограниченным количеством точек
    """
    if df is None or df.empty:
        return df
    
    if len(df) <= max_points:
        return df
    
    # Берем последние max_points точек
    limited_df = df.tail(max_points).copy()
    return limited_df

def get_default_time_range(df: pd.DataFrame, max_points: int = 500) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Возвращает диапазон дат для последних max_points точек
    
    Args:
        df: DataFrame для анализа
        max_points: Максимальное количество точек (по умолчанию 500)
    
    Returns:
        Кортеж (start_dt, end_dt) с датами начала и конца
    """
    if df is None or df.empty:
        return None, None
    
    limited_df = limit_data_to_last_points(df, max_points)
    start_dt = limited_df.index.min()
    end_dt = limited_df.index.max()
    
    return start_dt, end_dt

def initialize_limited_view(df: pd.DataFrame, max_points: int = 500, 
                          session_prefix: str = "") -> dict:
    """
    Инициализирует ограниченный вид данных в session_state
    
    Args:
        df: DataFrame для ограничения
        max_points: Максимальное количество точек
        session_prefix: Префикс для ключей session_state
    
    Returns:
        Словарь с ключами для session_state
    """
    limited_df = limit_data_to_last_points(df, max_points)
    
    return {
        f'filtered_df{session_prefix}': limited_df,
        f'original_df{session_prefix}': df,
        f'is_limited_view{session_prefix}': True,
        f'selected_sensors{session_prefix}': df.columns.tolist(),
        f'sensor_editor_temp{session_prefix}': df.columns.tolist()
    } 