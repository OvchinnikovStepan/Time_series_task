import pandas as pd
import streamlit as st
from typing import Tuple, Optional


def validate_feature(df: pd.DataFrame, feature: str) -> bool:
    """
    Проверяет, существует ли признак в DataFrame
    
    Args:
        df: DataFrame для проверки
        feature: Название признака
        
    Returns:
        bool: True если признак существует
    """
    if feature not in df.columns:
        st.error(f"Признак '{feature}' не найден в данных.")
        return False
    return True


def get_feature_series(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """
    Возвращает Series для указанного признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Optional[pd.Series]: Series признака или None
    """
    if not validate_feature(df, feature):
        return None
    return df[feature]


def calculate_mean(df: pd.DataFrame, feature: str) -> Optional[float]:
    """
    Вычисляет среднее значение признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Optional[float]: Среднее значение или None
    """
    series = get_feature_series(df, feature)
    return series.mean() if series is not None else None


def calculate_median(df: pd.DataFrame, feature: str) -> Optional[float]:
    """
    Вычисляет медиану признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Optional[float]: Медиана или None
    """
    series = get_feature_series(df, feature)
    return series.median() if series is not None else None


def calculate_std(df: pd.DataFrame, feature: str) -> Optional[float]:
    """
    Вычисляет стандартное отклонение признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Optional[float]: Стандартное отклонение или None
    """
    series = get_feature_series(df, feature)
    return series.std() if series is not None else None


def calculate_min(df: pd.DataFrame, feature: str) -> Optional[float]:
    """
    Вычисляет минимальное значение признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Optional[float]: Минимальное значение или None
    """
    series = get_feature_series(df, feature)
    return series.min() if series is not None else None


def calculate_max(df: pd.DataFrame, feature: str) -> Optional[float]:
    """
    Вычисляет максимальное значение признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Optional[float]: Максимальное значение или None
    """
    series = get_feature_series(df, feature)
    return series.max() if series is not None else None


def get_feature_statistics(df: pd.DataFrame, feature: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Возвращает статистики для указанного признака
    
    Args:
        df: DataFrame
        feature: Название признака
        
    Returns:
        Tuple: (среднее, медиана, стандартное_отклонение, минимум, максимум)
    """
    mean = calculate_mean(df, feature)
    median = calculate_median(df, feature)
    std = calculate_std(df, feature)
    minimal = calculate_min(df, feature)
    maximum = calculate_max(df, feature)
    
    return mean, median, std, minimal, maximum


# Для обратной совместимости
def info_about_feature(df: pd.DataFrame, feature: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Алиас для get_feature_statistics для обратной совместимости
    """
    return get_feature_statistics(df, feature)