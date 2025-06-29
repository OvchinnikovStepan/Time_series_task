import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats


def upload():
    upload_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file)

            if isinstance(df.index, pd.DatetimeIndex):
                date_column = df.index.name
            else:
                date_column = None
                date_pattern = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
                for column in df.columns:
                    sample = df[column].astype(str).head(10)
                    if sample.str.contains(date_pattern, na=False).all():
                        try:
                            pd.to_datetime(df[column], errors='raise')
                            date_column = column
                            break
                        except:
                            continue
                
                if date_column is None:
                    st.error("Столбец с датой (формат YYYY-MM-DD HH:MM:SS) не найден.")
                    return None, None
                
                df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S')
                df.set_index(date_column, inplace=True)
                if date_column in df.columns:
                    df.drop(columns=date_column, inplace=True)

            if not df.index.is_monotonic_decreasing:
                df.sort_index(ascending=False, inplace=True)

            if len(df.columns) == 0:
                st.error("Столбцы с данными датчиков не найдены.")
                return None, None

            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise').astype(np.float64)
                except:
                    st.error(f"Ошибка: датасет некорректен, присутствует переменная, которая не является показателем датчика.")
                    return None, None
                
            df.fillna(df.median(), inplace=True)

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