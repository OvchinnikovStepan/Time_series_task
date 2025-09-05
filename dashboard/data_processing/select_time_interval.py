import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional
from dashboard.utils.validate_data import validate_data


def _align_to_index_tz(dt: datetime, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Приводит произвольный datetime к типу/таймзоне индекса:
    - если индекс с TZ (например, UTC) -> локализуем/конвертируем границы в ту же TZ
    - если индекс naive -> снимаем TZ у границ (делаем naive)
    """
    ts = pd.to_datetime(dt)
    tz = getattr(idx, "tz", None)

    if tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    else:
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

    return ts


def start_date(df: Optional[pd.DataFrame], context: str = "panel") -> Optional[pd.Timestamp]:
    if validate_data(df):
        min_dt = df.index.min()
        max_dt = df.index.max()

        start_date_val = st.date_input(
            "Дата начала",
            value=min_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key=f"start_date_{context}",
        )
        start_time_val = st.time_input(
            "Время начала",
            value=min_dt.time(),
            key=f"start_time_{context}",
        )

        naive_dt = datetime.combine(start_date_val, start_time_val)
        return _align_to_index_tz(naive_dt, df.index)
    else:
        start_date_val = st.date_input("Дата начала", value=None, key=f"start_date_empty_{context}")
        start_time_val = st.time_input("Время начала", value=None, key=f"start_time_empty_{context}")
        if start_date_val is None or start_time_val is None:
            return None
        return pd.Timestamp(datetime.combine(start_date_val, start_time_val))


def end_date(df: Optional[pd.DataFrame], context: str = "panel") -> Optional[pd.Timestamp]:
    if validate_data(df):
        min_dt = df.index.min()
        max_dt = df.index.max()

        end_date_val = st.date_input(
            "Дата конца",
            value=max_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key=f"end_date_{context}",
        )
        end_time_val = st.time_input(
            "Время конца",
            value=max_dt.time(),
            key=f"end_time_{context}",
        )

        naive_dt = datetime.combine(end_date_val, end_time_val)
        return _align_to_index_tz(naive_dt, df.index)
    else:
        end_date_val = st.date_input("Дата конца", value=None, key=f"end_date_empty_{context}")
        end_time_val = st.time_input("Время конца", value=None, key=f"end_time_empty_{context}")
        if end_date_val is None or end_time_val is None:
            return None
        return pd.Timestamp(datetime.combine(end_date_val, end_time_val))


def filter_dataframe(start_dt: Optional[datetime], end_dt: Optional[datetime], df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not validate_data(df):
        st.warning("DataFrame пуст или индекс не является временным.")
        return None

    if start_dt is None or end_dt is None:
        st.warning("Укажите обе даты и время начала и конца.")
        return None

    # Привести границы к типу/таймзоне индекса (устраняем «aware vs naive»)
    start = _align_to_index_tz(start_dt, df.index) if not isinstance(start_dt, pd.Timestamp) else _align_to_index_tz(start_dt.to_pydatetime(), df.index)
    end = _align_to_index_tz(end_dt, df.index) if not isinstance(end_dt, pd.Timestamp) else _align_to_index_tz(end_dt.to_pydatetime(), df.index)

    if start >= end:
        st.error("Время начала должно быть меньше времени конца.")
        return None

    # Фильтрация данных
    idx = df.index
    filtered_df = df.loc[(idx >= start) & (idx <= end)].copy()
    if filtered_df.empty:
        st.warning(f"Нет данных в интервале с {start} по {end}.")
        return None

    st.session_state['filtered_df'] = filtered_df
    return filtered_df
