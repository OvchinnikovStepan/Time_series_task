import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

def upload():
    upload_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file)

            # Список возможных форматов дат
            date_formats = [
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

            # Функция для проверки, является ли строка датой
            def is_date_column(column_data, sample_size=10):
                sample = column_data.astype(str).head(sample_size).dropna()
                for date_format in date_formats:
                    try:
                        pd.to_datetime(sample, format=date_format, errors='raise')
                        return True, date_format
                    except:
                        continue
                return False, None

            # Проверка, является ли индекс датой
            if isinstance(df.index, pd.DatetimeIndex):
                date_column = df.index.name
            else:
                date_column = None
                date_format = None
                
                # Поиск столбца с датами
                for column in df.columns:
                    is_date, found_format = is_date_column(df[column])
                    if is_date:
                        date_column = column
                        date_format = found_format
                        break
                
                if date_column is None:
                    # Если не нашли столбец с датами по форматам, пробуем распознать автоматически
                    for column in df.columns:
                        try:
                            temp = pd.to_datetime(df[column], errors='raise')
                            if temp.notna().sum() > len(temp) * 0.8:  # Минимум 80% валидных дат
                                date_column = column
                                date_format = None  # Автоматическое определение формата
                                break
                        except:
                            continue
                
                if date_column is None:
                    st.error("Столбец с датой не найден. Поддерживаемые форматы: " + 
                            ", ".join(date_formats) + " или другие стандартные форматы дат")
                    return None, None
                
                try:
                    # Преобразование столбца с датой
                    df[date_column] = pd.to_datetime(df[date_column], 
                                                  format=date_format if date_format else None,
                                                  errors='raise')
                    df.set_index(date_column, inplace=True)
                    if date_column in df.columns:
                        df.drop(columns=date_column, inplace=True)
                except Exception as e:
                    st.error(f"Ошибка при преобразовании дат в столбце {date_column}: {str(e)}")
                    return None, None

            # Сортировка по дате в порядке убывания
            if not df.index.is_monotonic_decreasing:
                df.sort_index(ascending=False, inplace=True)

            # Проверка наличия столбцов с данными датчиков
            if len(df.columns) == 0:
                st.error("Столбцы с данными датчиков не найдены.")
                return None, None

            # Проверка и преобразование столбцов с числовыми данными
            valid_columns = []
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise').astype(np.float64)
                    if df[col].isna().all():
                        st.warning(f"Столбец {col} содержит только NaN и будет исключён.")
                    else:
                        valid_columns.append(col)
                except:
                    st.error(f"Ошибка: столбец {col} содержит некорректные данные, не являющиеся показателями датчика.")
                    return None, None

            df = df[valid_columns]

            if df.empty:
                st.error("Нет валидных столбцов с данными датчиков.")
                return None, None

            # Заполнение пропущенных значений медианой
            df.fillna(df.median(), inplace=True)

            # Вычисление процента выбросов
            total_values = df.size
            total_outliers = 0
            for col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = z_scores > 3
                total_outliers += np.sum(outliers)

            outlier_percentage = (total_outliers / total_values) * 100 if total_values > 0 else 0
            outlier_percentage = round(outlier_percentage, 2)

            return df, outlier_percentage

        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")
            return None, None

    return None, None