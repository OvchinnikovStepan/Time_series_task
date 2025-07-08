import streamlit as st
import pandas as pd
import asyncio
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.visualization.plot_interactive_with_selection import plot_interactive_with_selection
from API.app.request_functions.create_model_payload_func import create_model_payload
from API.app.request_functions.model_request_func import get_prediction
from typing import Optional, List

def render_data_overview(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Отображает верхнюю панель с общей информацией о данных
    """
    top_cols = st.columns([2, 2, 2, 2, 2])
    features_size, tuples_size, first_tuple, last_tuple = info_about_dataframe(df)
    with top_cols[0]:
        st.markdown(f"Кол-во записей: {tuples_size if tuples_size is not None else 'Нет информации'}")
    with top_cols[1]:
        st.markdown(f"Количество признаков: {features_size if features_size is not None else 'Нет информации'}")
    with top_cols[2]:
        st.markdown(f"Первая запись: {first_tuple if first_tuple is not None else 'Нет информации'}")
    with top_cols[3]:
        st.markdown(f"Последняя запись: {last_tuple if last_tuple is not None else 'Нет информации'}")
    with top_cols[4]:
        st.markdown(f"Количество выбросов: {f'{outlier_percentage}% от всех значений' if outlier_percentage is not None else 'Нет информации'}")

def render_forecasting_main_panel(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Основная панель: фильтрация, интерактивный график, предпросмотр, параметры, панель управления прогнозом справа
    Возвращает training_df для передачи в панель управления прогнозом
    """
    main_cols = st.columns([9, 3])
    training_df = None
    with main_cols[0]:
        if df is not None and not df.empty:
            filtered_df = st.session_state.get('filtered_df', df)
            selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())
            if selected_sensors:
                training_df = plot_interactive_with_selection(filtered_df, selected_sensors=selected_sensors, flag=True)
            else:
                st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
        else:
            st.markdown("""<div class=\"block\" style=\"height: 420px;\"></div>""", unsafe_allow_html=True)
        lower_cols = st.columns([6, 6])
        with lower_cols[1]:
            st.markdown("#### Параметры:")
            if df is not None and not df.empty:
                sensor_df = pd.DataFrame({
                    'Датчики': df.columns,
                    'Отображать': [col in st.session_state.get('selected_sensors', df.columns.tolist()) for col in df.columns]
                })
                edited_sensor_df = st.data_editor(sensor_df, key="sensor_selector")
                st.session_state['sensor_editor_temp'] = edited_sensor_df[edited_sensor_df['Отображать']]['Датчики'].tolist()
            else:
                st.markdown("Нет информации", unsafe_allow_html=True)
        with lower_cols[0]:
            st.markdown("#### Предпросмотр:")
            filtered_df = st.session_state.get('filtered_df', df)
            if filtered_df is not None:
                st.dataframe(filtered_df)
            else:
                st.markdown("Нет информации", unsafe_allow_html=True)
    with main_cols[1]:
        render_forecasting_control_panel(df, training_df)
    return training_df

def show_training_data_dialog(training_df: Optional[pd.DataFrame]) -> None:
    @st.dialog("Тренировочные данные")
    def show_training_data():
        if training_df is not None and not training_df.empty:
            st.dataframe(training_df, height=300)
        else:
            st.markdown("Нет данных для отображения", unsafe_allow_html=True)
        if st.button("Выйти", key="exit"):
            st.rerun()
    if training_df is not None and not training_df.empty:
        if st.button("Отобразить тренировочные данные"):
            show_training_data()

def show_model_params_dialog(option: str) -> None:
    @st.dialog("Выберите параметры")
    def show_confirmation_dialog():
        params = {}
        dialog_cols = st.columns([1, 1])
        if option == "sarima":
            with dialog_cols[0]:
                params['S'] = st.number_input("Сезонность", key="S", value=0, step=1, format="%d")
                params['p'] = st.number_input("Порядок авторегрессии", key="p", value=0, step=1, format="%d")
                params['d'] = st.number_input("Порядок дифференцирования ряда", key="d", value=0, step=1, format="%d")
                params['q'] = st.number_input("Порядок скользящего среднего", key="q", value=0, step=1, format="%d")
            with dialog_cols[1]:
                params['P'] = st.number_input("Порядок сезонной авторегрессии", key="P", value=0, step=1, format="%d")
                params['D'] = st.number_input("Порядок сезонного дифференцирования", key="D", value=0, step=1, format="%d")
                params['Q'] = st.number_input("Порядок сезонного скользящего среднего", key="Q", value=0, step=1, format="%d")
        elif option == "ets":
            with dialog_cols[0]:
                params['error_type'] = st.selectbox("Тип ошибки", ["add", "mul"], key="error_type")
                params['trend_type'] = st.selectbox("Тип тренда", ["None", "add", "mul"], key="trend_type")
                params['season_type'] = st.selectbox("Тип сезона", ["None", "add", "mul"], key="season_type")
            with dialog_cols[1]:
                params['seasonal_periods'] = st.number_input("Сезонность", key="seasonal_periods", value=0, step=1, format="%d")
                params['damped_trend'] = st.selectbox("Дампируется ли тренд", ["True", "False"], key="damped_trend") == "True"
        elif option == "prophet":
            with dialog_cols[0]:
                params['growth'] = st.selectbox("Тип тренда", ["linear", "logistic"], key="growth")
                params['seasonality_mode'] = st.selectbox("Режим моделирования сезонных компонент", ["additive", "multiplicative"], key="seasonality_mode")
                params['yearly_seasonality'] = st.selectbox("Настройка годовой сезонности", ["True", "False"], key="yearly_seasonality") == "True"
                params['weekly_seasonality'] = st.selectbox("Настройка недельной сезонности", ["True", "False"], key="weekly_seasonality") == "True"
            with dialog_cols[1]:
                params['daily_seasonality'] = st.selectbox("Настройка дневной сезонности", ["True", "False"], key="daily_seasonality") == "True"
                params['seasonality_prior_scale'] = st.number_input("Выраженность сезонных компонент", key="seasonality_prior_scale", value=10.0)
                params['changepoint_prior_scale'] = st.number_input("Чувствительность автоматического механизма обнаружения точек излома в тренде временного ряда", key="changepoint_prior_scale", value=0.05)
        if st.button("Подтвердить параметры", key=f"{option}_start"):
            st.session_state['params'] = params
            st.rerun()
    if st.button("Выбрать параметры модели"):
        show_confirmation_dialog()

def select_seasonality_dialog(option: str) -> None:
    @st.dialog("Выберите сезонность")
    def select_seasonality():
        st.markdown("Задайте период сезонности")
        params = st.session_state.get('params', {})
        if option in ["sarima", "ets"]:
            if option == "sarima":
                params['S'] = st.number_input("Сезонность", min_value=1, value=params.get('S', 12), step=1, format="%d", key="seasonality_S")
            elif option == "ets":
                params['seasonal_periods'] = st.number_input("Сезонность", min_value=1, value=params.get('seasonal_periods', 12), step=1, format="%d", key="seasonality_ets")
        if st.button("Подтвердить", key="confirm_seasonality"):
            st.session_state['params'] = params
            st.rerun()
    select_seasonality()

def render_forecasting_control_panel(df: pd.DataFrame, training_df: Optional[pd.DataFrame]) -> None:
    """
    Панель управления прогнозированием: выбор интервала, фильтрация, целевой признак, модель, запуск
    """
    st.markdown("## Прогнозирование")
    st.markdown("#### Область прогнозирования")
    if 'reset_counter' not in st.session_state:
        st.session_state['reset_counter'] = 0
    context = f"display_panel_{st.session_state['reset_counter']}"
    start_cols = st.columns(2)
    with start_cols[0]:
        start_datetime = start_date(df, context=context)
    with start_cols[1]:
        end_datetime = end_date(df, context=context)
    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Применить фильтр"):
            if start_datetime is not None and end_datetime is not None:
                st.session_state['filtered_df'] = filter_dataframe(start_datetime, end_datetime, df)
            else:
                st.session_state['filtered_df'] = df
            if 'sensor_editor_temp' in st.session_state:
                if st.session_state['sensor_editor_temp']:
                    st.session_state['selected_sensors'] = st.session_state['sensor_editor_temp']
                else:
                    st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
                    st.session_state['selected_sensors'] = []
            else:
                st.session_state['selected_sensors'] = df.columns.tolist()
            st.rerun()
    with button_cols[1]:
        if st.button("Сбросить фильтр"):
            st.session_state['filtered_df'] = df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['reset_counter'] = st.session_state.get('reset_counter', 0) + 1
            old_context = f"display_panel_{st.session_state['reset_counter'] - 1}"
            keys_to_clear = [
                f'start_date_{old_context}',
                f'start_time_{old_context}',
                f'end_date_{old_context}',
                f'end_time_{old_context}'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    st.markdown("#### Целевые параметры:")
    target_sensor = None
    if df is not None and not df.empty:
        available_sensors = st.session_state.get('selected_sensors', df.columns.tolist())
        if training_df is not None and not training_df.empty:
            available_sensors = list(training_df.columns)
        if len(available_sensors) == 0:
            st.markdown("Нет доступных параметров для прогнозирования")
        else:
            target_sensor = st.selectbox(
                "Выберите целевой признак",
                options=available_sensors,
                index=0,
                key="target_sensor"
            )
    else:
        st.markdown("Нет информации", unsafe_allow_html=True)
    show_training_data_dialog(training_df)
    st.markdown("#### Выбрать модель")
    option = st.selectbox(
        "Выберите параметр",
        ["sarima", "ets", "prophet"],
        key="model_select"
    )

    if st.button("Начать прогнозирование"):
        @st.dialog("Настройка прогноза")
        def forecast_settings_dialog():
            auto_params = st.checkbox("Автоподбор параметров", key="auto_params_dialog")
            duration = st.number_input("Количество предсказаний", min_value=1, value=st.session_state.get('duration', 1), step=1, format="%d", key="duration_dialog")
            params = st.session_state.get('params', {})
            if auto_params and option in ["sarima", "ets"]:
                if option == "sarima":
                    params['S'] = st.number_input("Сезонность (S)", min_value=1, value=params.get('S', 12), step=1, format="%d", key="seasonality_S")
                elif option == "ets":
                    params['seasonal_periods'] = st.number_input("Сезонность (seasonal_periods)", min_value=1, value=params.get('seasonal_periods', 12), step=1, format="%d", key="seasonality_ets")
            if not auto_params:
                if option == "sarima":
                    params['S'] = st.number_input("Сезонность (S)", min_value=1, value=params.get('S', 12), step=1, format="%d", key="sarima_S")
                    params['p'] = st.number_input("Порядок авторегрессии (p)", min_value=0, value=params.get('p', 0), step=1, format="%d", key="sarima_p")
                    params['d'] = st.number_input("Порядок дифференцирования (d)", min_value=0, value=params.get('d', 0), step=1, format="%d", key="sarima_d")
                    params['q'] = st.number_input("Порядок скользящего среднего (q)", min_value=0, value=params.get('q', 0), step=1, format="%d", key="sarima_q")
                    params['P'] = st.number_input("Порядок сезонной авторегрессии (P)", min_value=0, value=params.get('P', 0), step=1, format="%d", key="sarima_P")
                    params['D'] = st.number_input("Порядок сезонного дифференцирования (D)", min_value=0, value=params.get('D', 0), step=1, format="%d", key="sarima_D")
                    params['Q'] = st.number_input("Порядок сезонного скользящего среднего (Q)", min_value=0, value=params.get('Q', 0), step=1, format="%d", key="sarima_Q")
                elif option == "ets":
                    params['error_type'] = st.selectbox("Тип ошибки", ["add", "mul"], index=["add", "mul"].index(params.get('error_type', 'add')), key="ets_error_type")
                    params['trend_type'] = st.selectbox("Тип тренда", ["None", "add", "mul"], index=["None", "add", "mul"].index(params.get('trend_type', 'None')), key="ets_trend_type")
                    params['season_type'] = st.selectbox("Тип сезона", ["None", "add", "mul"], index=["None", "add", "mul"].index(params.get('season_type', 'None')), key="ets_season_type")
                    params['seasonal_periods'] = st.number_input("Сезонность (seasonal_periods)", min_value=1, value=params.get('seasonal_periods', 12), step=1, format="%d", key="ets_seasonal_periods")
                    params['damped_trend'] = st.selectbox("Дампируется ли тренд", ["True", "False"], index=["True", "False"].index(str(params.get('damped_trend', 'True'))), key="ets_damped_trend") == "True"
                elif option == "prophet":
                    params['growth'] = st.selectbox("Тип тренда", ["linear", "logistic"], index=["linear", "logistic"].index(params.get('growth', 'linear')), key="prophet_growth")
                    params['seasonality_mode'] = st.selectbox("Режим моделирования сезонных компонент", ["additive", "multiplicative"], index=["additive", "multiplicative"].index(params.get('seasonality_mode', 'additive')), key="prophet_seasonality_mode")
                    params['yearly_seasonality'] = st.selectbox("Настройка годовой сезонности", ["True", "False"], index=["True", "False"].index(str(params.get('yearly_seasonality', 'True'))), key="prophet_yearly_seasonality") == "True"
                    params['weekly_seasonality'] = st.selectbox("Настройка недельной сезонности", ["True", "False"], index=["True", "False"].index(str(params.get('weekly_seasonality', 'True'))), key="prophet_weekly_seasonality") == "True"
                    params['daily_seasonality'] = st.selectbox("Настройка дневной сезонности", ["True", "False"], index=["True", "False"].index(str(params.get('daily_seasonality', 'True'))), key="prophet_daily_seasonality") == "True"
                    params['seasonality_prior_scale'] = st.number_input("Выраженность сезонных компонент", value=float(params.get('seasonality_prior_scale', 10.0)), key="prophet_seasonality_prior_scale")
                    params['changepoint_prior_scale'] = st.number_input("Чувствительность точек излома", value=float(params.get('changepoint_prior_scale', 0.05)), key="prophet_changepoint_prior_scale")
            if st.button("Подтвердить", key="confirm_forecast_dialog"):
                st.session_state['duration'] = duration
                st.session_state['params'] = params
                # Выполнить прогнозирование (логика не меняется)
                if training_df is not None and not training_df.empty and target_sensor:
                    if 'duration' not in st.session_state:
                        st.error("Ошибка: Сначала выберите количество предсказаний!")
                        return
                    elif auto_params and option in ["sarima", "ets"] and 'params' not in st.session_state:
                        st.error("Ошибка: Для выбранной модели с автоподбором необходимо указать сезонность!")
                        return
                    elif not auto_params and 'params' not in st.session_state:
                        st.error("Ошибка: Сначала выберите параметры модели!")
                        return
                    else:
                        df_train = training_df[[target_sensor]].copy()
                        df_train = df_train.rename(columns={target_sensor: 'sensor'})
                        if not isinstance(df_train.index, pd.DatetimeIndex):
                            st.error("Ошибка: Индекс df_train должен быть типа DatetimeIndex")
                            return
                        end_training_time = df_train.index.max() if df_train is not None and not df_train.empty else None
                        df_test = None
                        if df is not None and not df.empty and end_training_time is not None:
                            df_test = df[df.index > end_training_time][[target_sensor]]
                            df_test = df_test.rename(columns={target_sensor: 'sensor'})
                            if df_test.empty:
                                st.warning("Предупреждение: df_test пуст, так как training_df охватывает весь временной диапазон df.")
                        payload = create_model_payload(
                            model_type=option,
                            auto_params=auto_params,
                            duration=duration,
                            df_train=df_train,
                            df_test=df_test,
                            params=params
                        )
                        try:
                            response = asyncio.run(get_prediction(payload))
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state['forecast_result'] = result
                                st.success(f"Прогноз выполнен для {target_sensor}!")
                            else:
                                st.error(f"Ошибка API: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Ошибка при выполнении прогноза: {str(e)}")
                        st.rerun()
                else:
                    st.error("Ошибка: Загрузите DataFrame, выберите данные для обучения и целевую переменную.")
                    return
        forecast_settings_dialog()
    # Отображение результата прогнозирования
    result = st.session_state.get('forecast_result', None)
    if result:
        st.markdown(f"Результат прогнозирования: {result}")

def render_forecasting_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Прогнозирование"
    """
    st.set_page_config(page_title="Прогнозирование", layout="wide")
    render_data_overview(df, outlier_percentage)
    if df is not None:
        current_df_hash = hash(pd.util.hash_pandas_object(df, index=True).sum())
        if st.session_state.get('last_df_hash') != current_df_hash:
            st.session_state.clear()
            st.session_state['filtered_df'] = df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash
    render_forecasting_main_panel(df) 