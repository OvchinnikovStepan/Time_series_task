import pandas as pd
from typing import Tuple, Optional, List


def get_date_formats() -> List[str]:
    """Возвращает список поддерживаемых форматов дат"""
    return [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%d.%m.%Y %H:%M:%S',
        '%d.%m.%Y %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%Y-%m-%d',
        '%d.%m.%Y',
        '%Y/%m/%d',
        '%m/%d/%Y'
    ]


def is_date_column(column_data: pd.Series, sample_size: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Проверяет, является ли столбец датой
    
    Args:
        column_data: Данные столбца
        sample_size: Размер выборки для проверки
        
    Returns:
        Tuple[bool, Optional[str]]: (является_датой, найденный_формат)
    """
    date_formats = get_date_formats()
    sample = column_data.astype(str).head(sample_size).dropna()
    
    for date_format in date_formats:
        try:
            pd.to_datetime(sample, format=date_format, errors='raise')
            return True, date_format
        except:
            continue
    return False, None


def find_date_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Находит столбец с датами в DataFrame
    
    Args:
        df: DataFrame для поиска
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (название_столбца, формат_даты)
    """
    # Проверка, является ли индекс датой
    if isinstance(df.index, pd.DatetimeIndex):
        return str(df.index.name) if df.index.name else None, None
    
    # Поиск столбца с датами по форматам
    for column in df.columns:
        is_date, found_format = is_date_column(df[column])
        if is_date:
            return str(column), found_format
    
    # Автоматическое определение даты
    for column in df.columns:
        try:
            temp = pd.to_datetime(df[column], errors='raise')
            if temp.notna().sum() > len(temp) * 0.8:  # Минимум 80% валидных дат
                return str(column), None
        except:
            continue
    
    return None, None


def process_date_column(df: pd.DataFrame, date_column: str, date_format: Optional[str]) -> pd.DataFrame:
    """
    Обрабатывает столбец с датами и устанавливает его как индекс
    
    Args:
        df: DataFrame для обработки
        date_column: Название столбца с датами
        date_format: Формат даты (если известен)
        
    Returns:
        pd.DataFrame: DataFrame с обработанным индексом
    """
    try:
        df[date_column] = pd.to_datetime(
            df[date_column], 
            format=date_format if date_format else None,
            errors='raise'
        )
        df.set_index(date_column, inplace=True)
        
        # Удаляем дублирующий столбец, если он остался
        if date_column in df.columns:
            df.drop(columns=date_column, inplace=True)
            
        return df
    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании дат в столбце {date_column}: {str(e)}") 