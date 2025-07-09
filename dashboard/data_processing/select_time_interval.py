import streamlit as st
import pandas as pd
from datetime import datetime
from dashboard.utils.validate_data import validate_data

def start_date(df, context="panel"):
    if validate_data(df):
        min_dt = df.index.min()
        max_dt = df.index.max()
        start_date = st.date_input(
            "Дата начала",
            value=min_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key=f"start_date_{context}"
        )
        start_time = st.time_input(
            "Время начала",
            value=min_dt.time(),
            key=f"start_time_{context}"
        )
        start_datetime = datetime.combine(start_date, start_time)
        return start_datetime
    else:
        start_date = st.date_input("Дата начала", value=None, key=f"start_date_empty_{context}")
        start_time = st.time_input("Время начала", value=None, key=f"start_time_empty_{context}")
        start_datetime = None if start_date is None or start_time is None else datetime.combine(start_date, start_time)
        return start_datetime

def end_date(df, context="panel"):
    if validate_data(df):
        min_dt = df.index.min()
        max_dt = df.index.max()
        end_date = st.date_input(
            "Дата конца",
            value=max_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key=f"end_date_{context}"
        )
        end_time = st.time_input(
            "Время конца",
            value=max_dt.time(),
            key=f"end_time_{context}"
        )
        end_datetime = datetime.combine(end_date, end_time)
        return end_datetime
    else:
        end_date = st.date_input("Дата конца", value=None, key=f"end_date_empty_{context}")
        end_time = st.time_input("Время конца", value=None, key=f"end_time_empty_{context}")
        end_datetime = None if end_date is None or end_time is None else datetime.combine(end_date, end_time)
        return end_datetime
    
def filter_dataframe(start_dt, end_dt, df):
    if not validate_data(df):
        st.warning("DataFrame пуст или индекс не является временным.")
        return None

    if start_dt is None or end_dt is None:
        st.warning("Укажите обе даты и время начала и конца.")
        return None

    if start_dt >= end_dt:
        st.error("Время начала должно быть меньше времени конца.")
        return None

    # Фильтрация данных
    filtered_df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
    if filtered_df.empty:
        st.warning(f"Нет данных в интервале с {start_dt} по {end_dt}.")
        return None

    st.session_state['filtered_df'] = filtered_df
    return filtered_df