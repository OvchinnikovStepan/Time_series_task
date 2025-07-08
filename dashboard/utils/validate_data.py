import pandas as pd
from typing import Optional


def validate_data(df: pd.DataFrame) -> bool:
    """
    Простая валидация DataFrame (для обратной совместимости)
    
    Args:
        df: DataFrame для проверки
        
    Returns:
        bool: True если DataFrame валиден
    """
    if df is None or df.empty:
        return False
    return True


def validate_dataframe_structure(df: pd.DataFrame, raise_error: bool = True) -> bool:
    """
    Детальная валидация структуры DataFrame
    
    Args:
        df: DataFrame для проверки
        raise_error: Выбрасывать ли исключение при ошибке
        
    Returns:
        bool: True если структура валидна
        
    Raises:
        ValueError: Если raise_error=True и структура некорректна
    """
    if df is None:
        if raise_error:
            raise ValueError("DataFrame не может быть None")
        return False
    
    if df.empty:
        if raise_error:
            raise ValueError("DataFrame не может быть пустым")
        return False
    
    if len(df.columns) == 0:
        if raise_error:
            raise ValueError("DataFrame должен содержать хотя бы один столбец")
        return False
    
    return True


def validate_dataframe_with_error_handling(df: pd.DataFrame) -> bool:
    """
    Валидация DataFrame с обработкой ошибок (для UI)
    
    Args:
        df: DataFrame для проверки
        
    Returns:
        bool: True если DataFrame валиден
    """
    try:
        validate_dataframe_structure(df, raise_error=True)
        return True
    except ValueError as e:
        return False