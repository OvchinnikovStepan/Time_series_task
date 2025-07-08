import pandas as pd
import numpy as np
from scipy import stats
from typing import Union


def calculate_outlier_percentage(df: pd.DataFrame) -> float:
    """
    Вычисляет процент выбросов в DataFrame
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        float: Процент выбросов
    """
    total_values = df.size
    total_outliers = 0
    
    for col in df.columns:
        # Убираем NaN значения перед вычислением z-score
        clean_data = df[col].dropna()
        if len(clean_data) > 0:
            z_scores = np.abs(stats.zscore(clean_data))
            outliers = z_scores > 3
            total_outliers += np.sum(outliers)
    
    outlier_percentage = (total_outliers / total_values) * 100 if total_values > 0 else 0
    return float(round(outlier_percentage, 2))


def fill_missing_values(df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
    """
    Заполняет пропущенные значения в DataFrame
    
    Args:
        df: DataFrame для обработки
        method: Метод заполнения ('median', 'mean', 'forward', 'backward')
        
    Returns:
        pd.DataFrame: DataFrame с заполненными пропусками
    """
    if method == 'median':
        df.fillna(df.median(), inplace=True)
    elif method == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif method == 'forward':
        df.fillna(method='ffill', inplace=True)
    elif method == 'backward':
        df.fillna(method='bfill', inplace=True)
    else:
        raise ValueError(f"Неизвестный метод заполнения: {method}")
    
    return df


def sort_dataframe_by_index(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    Сортирует DataFrame по индексу
    
    Args:
        df: DataFrame для сортировки
        ascending: Сортировка по возрастанию (True) или убыванию (False)
        
    Returns:
        pd.DataFrame: Отсортированный DataFrame
    """
    if not df.index.is_monotonic_increasing if ascending else df.index.is_monotonic_decreasing:
        df.sort_index(ascending=ascending, inplace=True)
    
    return df 