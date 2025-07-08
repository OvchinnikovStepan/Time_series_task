import streamlit as st
import pandas as pd
from typing import Tuple, Optional

# Импортируем утилиты из отдельных файлов
from dashboard.utils.date_utils import find_date_column, process_date_column, get_date_formats
from dashboard.utils.validation_utils import validate_numeric_columns
from dashboard.utils.validate_data import validate_dataframe_structure
from dashboard.utils.statistics_utils import calculate_outlier_percentage, fill_missing_values, sort_dataframe_by_index


def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Обрабатывает DataFrame: валидирует данные, обрабатывает пропуски, вычисляет выбросы
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        Tuple[pd.DataFrame, float]: (обработанный_DataFrame, процент_выбросов)
    """
    # Валидируем базовую структуру
    validate_dataframe_structure(df)
    
    # Находим столбец с датами
    date_column, date_format = find_date_column(df)
    
    if date_column is None:
        st.error("Столбец с датой не найден. Поддерживаемые форматы: " + 
                ", ".join(get_date_formats()) + " или другие стандартные форматы дат")
        raise ValueError("Date column not found")
    
    # Обрабатываем столбец с датами
    df = process_date_column(df, date_column, date_format)
    
    # Сортируем по дате в порядке убывания
    df = sort_dataframe_by_index(df, ascending=False)
    
    # Валидируем числовые столбцы
    valid_columns = validate_numeric_columns(df)
    df = df[valid_columns]
    
    if df.empty:
        st.error("Нет валидных столбцов с данными датчиков.")
        raise ValueError("No valid columns found")
    
    # Заполняем пропущенные значения медианой
    df = fill_missing_values(df, method='median')
    
    # Вычисляем процент выбросов
    outlier_percentage = calculate_outlier_percentage(df)
    
    return df, outlier_percentage


def upload() -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Основная функция загрузки и обработки CSV файла
    
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[float]]: (DataFrame, процент_выбросов)
    """
    upload_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    
    if upload_file is not None:
        try:
            # Загружаем CSV
            df = pd.read_csv(upload_file)
            
            # Обрабатываем DataFrame
            df, outlier_percentage = process_dataframe(df)
            
            return df, outlier_percentage
            
        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")
            return None, None
    
    return None, None