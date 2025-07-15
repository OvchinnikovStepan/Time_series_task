import streamlit as st
import pandas as pd
import asyncio
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.visualization.plot_interactive_with_selection import plot_interactive_with_selection
from dashboard.request_functions.create_model_payload_func import create_model_payload
from dashboard.request_functions.model_request_func import get_prediction
from dashboard.request_functions.get_models_func import get_models
from dashboard.request_functions.metrics_request_func import get_metrics
from dashboard.request_functions.create_metrics_payload_func import create_metrics_payload
from dashboard.utils.data_limiting import limit_data_to_last_points, get_default_time_range
from typing import Optional, List
import json
import os
from io import StringIO
import numpy as np

from ..schemas import ModelRequest, MetricsRequest


def render_forecast_results():
    """
    Отображает результаты прогнозирования, если они есть в session_state.
    Показывает таблицы, графики и метрики в зависимости от наличия df_test и длины.
    """
    # Проверяем, есть ли данные для отображения
    filtered_df = st.session_state.get('filtered_df')
    selected_sensors = st.session_state.get('selected_sensors', [])
    
    if filtered_df is None or filtered_df.empty or not selected_sensors:
        return  # Не отображаем результаты, если данные пустые или выделение очищено
    
    forecast_result = st.session_state.get('forecast_result')
    metrics_result = st.session_state.get('metrics_result')
    df_test = st.session_state.get('df_test')
    duration = st.session_state.get('duration')
    target_sensor = st.session_state.get('target_sensor')
    if forecast_result is not None:
        st.markdown('---')
        st.markdown('### Результаты прогнозирования')
        try:
            df_predict_json = forecast_result.get("df_predict")
            if df_predict_json is None or df_predict_json == '' or df_predict_json == 'null':
                st.error('Ошибка: Предсказание невозможно. Значения датчиков сложно предсказуемы либо неправильно настроены параметры')
                return
            df_predict = pd.read_json(StringIO(df_predict_json), orient='table')
            # Проверяем, содержит ли DataFrame только NaN значения
            all_nan = df_predict.isna().all()
            if isinstance(all_nan, bool):
                only_nan = all_nan
            else:
                only_nan = all_nan.all()
            if only_nan:
                st.error('Ошибка: Предсказание невозможно. Значения датчиков сложно предсказуемы либо неправильно настроены параметры')
                return
            # Приводим имя столбца к 'sensor', если нужно
            if 'sensor' not in df_predict.columns and len(df_predict.columns) == 1:
                df_predict = df_predict.rename(columns={str(df_predict.columns[0]): 'sensor'})
        except Exception as e:
            st.error(f"Ошибка при чтении предсказанных значений: {str(e)}")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Предсказанные значения:**')
            st.dataframe(df_predict)
        # Аналогично для df_test
        if df_test is not None and isinstance(df_test, pd.DataFrame) and len(df_test) == len(df_predict):
            if 'sensor' not in df_test.columns and len(df_test.columns) == 1:
                df_test = df_test.rename(columns={str(df_test.columns[0]): 'sensor'})
            with col2:
                st.markdown('**Истинные значения:**')
                st.dataframe(df_test)
        # График на всю ширину
        st.markdown('**График прогнозирования:**')
        import plotly.graph_objs as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_predict.index,
            y=df_predict['sensor'],
            mode='lines+markers',
            name='Прогноз'
        ))
        if df_test is not None and isinstance(df_test, pd.DataFrame) and len(df_test) == len(df_predict):
            fig.add_trace(go.Scatter(
                x=df_test.index,
                y=df_test['sensor'],
                mode='lines+markers',
                name='Истинные значения'
            ))
        fig.update_layout(
            xaxis_title='Время',
            yaxis_title=str(target_sensor) if target_sensor else 'Значение',
            legend_title='Легенда',
            width=1200,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        # Метрики и гиперпараметры
        if metrics_result is not None and df_test is not None and len(df_test) == len(df_predict):
            col_metrics, col_hyperparams = st.columns(2)
            with col_metrics:
                st.markdown('### Метрики качества прогноза')
                st.json(metrics_result)
            with col_hyperparams:
                st.markdown('### Гиперпараметры модели')
                hyper_params = forecast_result.get("hyper_params", {})
                if hyper_params:
                    st.json(hyper_params)
                else:
                    st.info("Гиперпараметры не найдены")
        elif df_test is None or len(df_predict) > (len(df_test) if df_test is not None else 0):
            st.info('Недостаточно тестовых данных для расчёта метрик. Отображаются только предсказанные значения.')
            # Показываем только гиперпараметры
            st.markdown('### Гиперпараметры модели')
            hyper_params = forecast_result.get("hyper_params", {})
            if hyper_params:
                st.json(hyper_params)
            else:
                st.info("Гиперпараметры не найдены")

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
    # Добавляем отображение результатов прогнозирования на всю ширину
    render_forecast_results()
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

def get_api_url():
    """Получает URL API из секретов Streamlit"""
    return st.secrets.get("api_url")

def render_forecasting_control_panel(df: pd.DataFrame, training_df: Optional[pd.DataFrame]) -> None:
    """
    Панель управления прогнозированием: выбор интервала, фильтрация, целевой признак, модель, запуск
    """
    st.markdown("## Прогнозирование")
    
    # Информация о текущем режиме отображения
    original_df = st.session_state.get('original_df', df)
    if original_df is not None and len(original_df) > 500:
        if st.session_state.get('is_limited_view', False):
            st.info(f"Отображаются последние 500 из {len(original_df)} записей. Используйте фильтр для просмотра других периодов.")
            if st.button("Отобразить все записи", key="show_all_data"):
                st.session_state['filtered_df'] = original_df
                st.session_state['is_limited_view'] = False
                st.rerun()
        else:
            st.info(f"Отображаются все {len(original_df)} записей. Для лучшей производительности рекомендуется использовать ограниченный вид.")
            if st.button("Отобразить последние 500 записей", key="show_limited_data"):
                limited_df = limit_data_to_last_points(original_df, 500)
                st.session_state['filtered_df'] = limited_df
                st.session_state['is_limited_view'] = True
                st.rerun()
    
    st.markdown("#### Рассматриваемый временной промежуток")
    if 'reset_counter' not in st.session_state:
        st.session_state['reset_counter'] = 0
    context = f"display_panel_{st.session_state['reset_counter']}"
    
    # Используем отфильтрованный датасет для определения диапазона дат
    filtered_df = st.session_state.get('filtered_df', df)
    
    start_cols = st.columns(2)
    with start_cols[0]:
        start_datetime = start_date(filtered_df, context=context)
    with start_cols[1]:
        end_datetime = end_date(filtered_df, context=context)
    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Применить фильтр"):
            if start_datetime is not None and end_datetime is not None:
                # Применяем фильтр к оригинальному датасету, а не к отфильтрованному
                filtered_result = filter_dataframe(start_datetime, end_datetime, df)
                if filtered_result is not None:
                    st.session_state['filtered_df'] = filtered_result
                    # Если фильтр применен к полному датасету, сбрасываем флаг ограниченного вида
                    if len(filtered_result) == len(df):
                        st.session_state['is_limited_view'] = False
                    else:
                        st.session_state['is_limited_view'] = False  # Пользователь выбрал конкретный диапазон
                else:
                    st.error("Ошибка при применении фильтра")
                    return
            else:
                st.session_state['filtered_df'] = df
                st.session_state['is_limited_view'] = False
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
            # Возвращаемся к ограниченному виду (последние 500 точек)
            if st.session_state.get('is_limited_view', False) and st.session_state.get('original_df') is not None:
                limited_df = limit_data_to_last_points(st.session_state['original_df'], 500)
                st.session_state['filtered_df'] = limited_df
            else:
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
            # Сброс результатов прогнозирования, метрик и тестовых данных
            for key in ['forecast_result', 'metrics_result', 'df_test', 'duration']:
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
    # Получаем список доступных моделей из API
    api_url = get_api_url()
    if not api_url:
        st.error("❌ Не удалось получить URL API")
        return
        
    available_models = None
    try:
        models_response = asyncio.run(get_models(api_url))
        if models_response.status_code == 200:
            models_data = models_response.json().get('models', {})
            available_models = list(models_data.keys()) if models_data else None
    except Exception:
        pass
    
    if not available_models:
        st.error("❌ Не удалось получить список моделей от API")
        return
    
    option = st.selectbox(
        "Выберите модель",
        available_models,
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
                        if isinstance(df_train, pd.DataFrame):
                            df_train = df_train.rename(columns={str(target_sensor): 'sensor'})
                        if not isinstance(df_train.index, pd.DatetimeIndex):
                            st.error("Ошибка: Индекс df_train должен быть типа DatetimeIndex")
                            return
                        end_training_time = df_train.index.max() if df_train is not None and not df_train.empty else None
                        df_test = None
                        if df is not None and not df.empty and end_training_time is not None:
                            df_test = df[df.index > end_training_time][[target_sensor]]
                            if isinstance(df_test, pd.DataFrame):
                                df_test = df_test.rename(columns={str(target_sensor): 'sensor'})
                                # Берём только duration точек
                                if len(df_test) >= duration:
                                    df_test = df_test.iloc[:duration]
                                    st.session_state['df_test'] = df_test  # Сохраняем актуальный df_test
                                else:
                                    df_test = None  # Недостаточно данных для метрик
                                    st.session_state['df_test'] = None  # Очищаем старый df_test
                            else:
                                df_test = None
                                st.session_state['df_test'] = None
                        else:
                            df_test = None
                            st.session_state['df_test'] = None
                        if isinstance(df_train, pd.DataFrame):
                            payload = create_model_payload(
                                auto_params=auto_params,
                                horizon=duration,
                                df_train=df_train,
                                hyper_params=params
                            )
                        else:
                            st.error("Ошибка: df_train должен быть DataFrame")
                            return
                        try:
                            model_request = ModelRequest(**payload)
                            
                            # Показываем индикатор загрузки
                            with st.spinner("Выполняется прогнозирование..."):
                                response = asyncio.run(get_prediction(api_url, model_request.dict(), option))
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state['forecast_result'] = result
                                
                                # Метрики считаем только если df_test не None и длина совпадает с duration
                                if df_test is not None and len(df_test) == duration:
                                    try:
                                        df_predict = pd.read_json(StringIO(result["df_predict"]), orient='table')
                                        # Приводим имя столбца к 'sensor', если нужно
                                        if 'sensor' not in df_predict.columns and len(df_predict.columns) == 1:
                                            df_predict = df_predict.rename(columns={str(df_predict.columns[0]): 'sensor'})
                                        
                                        # Удаляем NaN и inf значения перед отправкой на метрики
                                        df_predict = df_predict.replace([np.inf, -np.inf, np.nan], 0)
                                        df_test = df_test.replace([np.inf, -np.inf, np.nan], 0)
                                        metrics_payload = create_metrics_payload(df_predict=df_predict, df_test=df_test)

                                        metrics_request = MetricsRequest(
                                            df_predict=str(metrics_payload["df_predict"]),
                                            df_test=str(metrics_payload["df_test"])
                                        )
                                        
                                        # Показываем индикатор загрузки для метрик
                                        with st.spinner("Рассчитываются метрики..."):
                                            metrics_response = asyncio.run(get_metrics(api_url, metrics_request.dict()))
                                        
                                        if metrics_response.status_code == 200:
                                            metrics_result = metrics_response.json()
                                            st.session_state['metrics_result'] = metrics_result
                                        else:
                                            st.warning(f"Не удалось получить метрики: {metrics_response.status_code}")
                                    except Exception as e:
                                        st.warning(f"Ошибка при расчете метрик: {str(e)}")
                                
                                st.success(f"Прогноз успешно выполнен для {target_sensor}!")
                                st.rerun()
                            else:
                                st.error(f"Ошибка API: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Ошибка при выполнении прогноза: {str(e)}")
                else:
                    st.error("Ошибка: Загрузите DataFrame, выберите данные для обучения и целевую переменную.")
        forecast_settings_dialog()  # Вызов диалога

def render_forecasting_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Прогнозирование"
    """
    st.set_page_config(page_title="Прогнозирование", layout="wide")
    # Не отображаем информацию о датасете, если df is None
    if df is None:
        st.session_state.clear()
        return
    render_data_overview(df, outlier_percentage)
    if df is not None:
        # Используем более простой способ хеширования DataFrame
        current_df_hash = hash(str(df.shape) + str(df.columns.tolist()) + str(df.index[-10:].tolist()) if len(df) > 10 else str(df.index.tolist()))
        if st.session_state.get('last_df_hash') != current_df_hash:
            st.session_state.clear()
            # При загрузке нового файла ограничиваем данные последними 500 точками
            limited_df = limit_data_to_last_points(df, 500)
            st.session_state['filtered_df'] = limited_df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash
            st.session_state['original_df'] = df  # Сохраняем оригинальный DataFrame
            st.session_state['is_limited_view'] = True  # Флаг, что отображается ограниченный вид
            # Сброс результатов прогнозирования, метрик и тестовых данных
            for key in ['forecast_result', 'metrics_result', 'df_test', 'duration']:
                if key in st.session_state:
                    del st.session_state[key]
    render_forecasting_main_panel(df)