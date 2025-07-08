import streamlit as st
import pandas as pd
from typing import Tuple, Optional


def get_features_count(df: pd.DataFrame) -> Optional[int]:
    """
    Возвращает количество признаков (столбцов) в DataFrame
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        Optional[int]: Количество признаков или None
    """
    return len(df.columns) if df is not None else None


def get_tuples_count(df: pd.DataFrame) -> Optional[int]:
    """
    Возвращает количество записей (строк) в DataFrame
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        Optional[int]: Количество записей или None
    """
    return len(df) if df is not None else None


def get_first_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Возвращает первую временную метку в DataFrame
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        Optional[pd.Timestamp]: Первая временная метка или None
    """
    return df.index.min() if df is not None and not df.empty else None


def get_last_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Возвращает последнюю временную метку в DataFrame
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        Optional[pd.Timestamp]: Последняя временная метка или None
    """
    return df.index.max() if df is not None and not df.empty else None


def get_dataframe_info(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Возвращает основную информацию о DataFrame
    
    Args:
        df: DataFrame для анализа
        
    Returns:
        Tuple: (количество_признаков, количество_записей, первая_метка, последняя_метка)
    """
    features_size = get_features_count(df)
    tuples_size = get_tuples_count(df)
    first_tuple = get_first_timestamp(df)
    last_tuple = get_last_timestamp(df)
    
    return features_size, tuples_size, first_tuple, last_tuple


# Для обратной совместимости
def info_about_dataframe(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Алиас для get_dataframe_info для обратной совместимости
    """
    return get_dataframe_info(df)