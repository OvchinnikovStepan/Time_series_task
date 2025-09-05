import streamlit as st
import pandas as pd
import httpx
from typing import Tuple, Optional
from datetime import datetime, date, time

# Импортируем утилиты из отдельных файлов (оставлены из базовой реализации)
from dashboard.utils.date_utils import find_date_column, process_date_column, get_date_formats
from dashboard.utils.validation_utils import validate_numeric_columns
from dashboard.utils.validate_data import validate_dataframe_structure
from dashboard.utils.statistics_utils import calculate_outlier_percentage, fill_missing_values, sort_dataframe_by_index

from dashboard.utils.validate_datetime_index import validate_datetime_index


def _normalize_source(source: Optional[str]) -> str:
    return (source or "").strip().lower()


def _clean_measurement_tag(tag: Optional[str]) -> str:
    # Тег измерения чувствителен к регистру, поэтому только обрезаем пробелы
    return (tag or "").strip()


def _get_data_url_for_source(source: str) -> str:
    """
    Возвращает полный URL для эндпоинта /data в зависимости от выбранного источника.
    Источники: 'msk' -> API_DATA_MSK_URL, 'oms' -> API_DATA_OMS_URL.
    """
    s = _normalize_source(source)
    if s == "msk":
        base = st.secrets.get("API_DATA_MSK_URL")
        secret_key = "API_DATA_MSK_URL"
    elif s == "oms":
        base = st.secrets.get("API_DATA_OMS_URL")
        secret_key = "API_DATA_OMS_URL"
    else:
        raise ValueError("Поддерживаемые источники: 'msk' или 'oms'.")

    if not base:
        raise RuntimeError(f"Не найден секрет {secret_key}. Проверьте .streamlit/secrets.toml")

    return f"{base.rstrip('/')}/data"


@st.dialog("Параметры запроса данных", width="large")
def api_params_dialog():
    source = st.selectbox("Источник данных", options=["msk", "oms"], index=0, key="api_source_select")
    tag_name = st.text_input("Название тэга")

    use_start = st.checkbox("Указать начало периода", value=False)
    start_date = st.date_input("Начало периода — дата", value=date.today(), disabled=not use_start)
    start_time = st.time_input("Начало периода — время", value=time(0, 0, 0), disabled=not use_start, key="start_time")

    use_end = st.checkbox("Указать конец периода", value=False)
    end_date = st.date_input("Конец периода — дата", value=date.today(), disabled=not use_end)
    end_time = st.time_input("Конец периода — время", value=time(23, 59, 59), disabled=not use_end, key="end_time")

    def assemble_dt(use_flag: bool, d: date, t: time) -> Optional[datetime]:
        return datetime.combine(d, t) if use_flag else None

    def to_iso_or_none(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    start_dt = assemble_dt(use_start, start_date, start_time)
    end_dt = assemble_dt(use_end, end_date, end_time)

    if st.button("Отправить"):
        st.session_state.api_params = {
            "source": _normalize_source(source),
            "measurement_tag": _clean_measurement_tag(tag_name),
            "dateStart": to_iso_or_none(start_dt),
            "dateEnd": to_iso_or_none(end_dt),
        }
        st.session_state.api_dialog_closed = True
        st.rerun()


def _df_from_api_array(records) -> pd.DataFrame:
    """
    Ожидает массив объектов [{"d": "...", "v": ...}, ...].
    Делает 'd' datetime-индексом (UTC, если есть 'Z') и сортирует.
    Дополнительно валидирует базовую структуру и индекс.
    """
    if not isinstance(records, list):
        raise ValueError("Ожидался массив JSON-объектов от API.")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    if "d" not in df.columns:
        raise ValueError("В ответе API нет колонки 'd' с датой/временем.")

    # Преобразуем в datetime (UTC-aware) и чистим плохие строки
    df["d"] = pd.to_datetime(df["d"], utc=True, errors="coerce")
    if df["d"].isna().any():
        df = df.dropna(subset=["d"])

    if df.empty:
        return df

    # Делаем индексом для унификации с пайплайном
    df = df.set_index("d").sort_index()

    # Базовая валидация и валидация индекса времени
    validate_dataframe_structure(df, raise_error=True)
    validate_datetime_index(df)

    return df


def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Обрабатывает DataFrame: валидирует данные, удаляет дубликаты,
    обрабатывает пропуски, вычисляет выбросы.
    Поддерживает:
    1) DataFrame с колонкой даты
    2) DataFrame с DatetimeIndex
    """
    # Валидируем базовую структуру
    validate_dataframe_structure(df)

    # Если индекс уже временной — используем его.
    if isinstance(df.index, pd.DatetimeIndex):
        # Сортируем по дате в порядке убывания
        df = sort_dataframe_by_index(df, ascending=False)
    else:
        # Старый путь: находим и обрабатываем колонку с датами
        date_column, date_format = find_date_column(df)
        if date_column is None:
            st.error("Столбец с датой не найден. Поддерживаемые форматы: " +
                     ", ".join(get_date_formats()) + " или другие стандартные форматы дат")
            raise ValueError("Date column not found")

        # Обрабатываем столбец с датами (делает DatetimeIndex внутри)
        df = process_date_column(df, date_column, date_format)
        # Сортируем по дате в порядке убывания
        df = sort_dataframe_by_index(df, ascending=False)

    # Удаляем дубликаты (важно для обеих загрузок)
    df = df.drop_duplicates()

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
    Основная функция загрузки и обработки данных:
    - Локальная загрузка CSV/XLSX
    - Запрос к API через диалог и httpx
    Возвращает (DataFrame, процент_выбросов) либо (None, None) при ошибке/отсутствии данных.
    """
    # 1) Локальная загрузка
    uploaded_file = st.file_uploader("Загрузите файл CSV или Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)

            df, outlier_percentage = process_dataframe(raw_df)
            st.success("Данные из файла успешно загружены и обработаны!")
            return df, outlier_percentage

        except Exception as e:
            st.error(f"Ошибка при обработке данных из файла: {str(e)}")
            return None, None

    # 2) Запрос к API
    if st.button("Запросить данные с сервера"):
        st.session_state.api_dialog_closed = False
        api_params_dialog()

    if st.session_state.get("api_dialog_closed", False):
        params = st.session_state.get("api_params")
        if params:
            try:
                source = _normalize_source(params.get("source"))
                if source not in {"msk", "oms"}:
                    st.error("Выберите источник ('msk' или 'oms') перед запросом.")
                    return None, None

                measurement_tag = _clean_measurement_tag(params.get("measurement_tag"))
                if not measurement_tag:
                    st.error("Укажите тэг измерения.")
                    return None, None

                payload = {"tag": measurement_tag}
                if params.get("dateStart"):
                    payload["dateStart"] = params["dateStart"]
                if params.get("dateEnd"):
                    payload["dateEnd"] = params["dateEnd"]

                data_url = _get_data_url_for_source(source)

                with httpx.Client(timeout=30.0, verify=False) as client:
                    resp = client.post(
                        data_url,
                        json=payload,
                        headers={"Content-Type": "application/json", "accept": "application/json"},
                    )
                    resp.raise_for_status()
                    data_json = resp.json()

                # Преобразуем и валидируем ответ API
                api_df = _df_from_api_array(data_json)

                # Дальше единый пайплайн обработки (удаление дубликатов, выбросы и т.п.)
                df, outlier_percentage = process_dataframe(api_df)

                st.success("Данные получены с сервера и обработаны!")
                return df, outlier_percentage

            except httpx.HTTPStatusError as e:
                st.error(f"Ошибка API ({e.response.status_code}): {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"Сетевой сбой при обращении к API: {e}")
            except (ValueError, RuntimeError) as e:
                st.error(f"Ошибка конфигурации/данных: {e}")
            except Exception as e:
                st.error(f"Непредвиденная ошибка при обработке данных: {e}")

    # Если ничего не выбрано/не пришло
    return None, None
