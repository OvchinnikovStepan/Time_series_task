import streamlit as st
import pandas as pd
import json
import httpx
import asyncio
from datetime import timedelta
from dashboard.upload import upload
from dashboard.info_about_dataframe import info_about_dataframe
from dashboard.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.plot_interactive_with_selection import plot_interactive_with_selection
from dashboard.show_heatmap import show_heatmap
from dashboard.info_about_feature import info_about_feature
from dashboard.show_pairplot import show_pairplot
from dashboard.forecasting import forecasting
from dashboard.show_hist import show_hist
from dashboard.show_autocorrelation import show_autocorrelation
from API.app.request_functions.create_model_payload_func import create_model_payload
from API.app.request_functions.model_request_func import get_prediction
import os
os.environ["STREAMLIT_SERVER_WATCH_FILE_BLACKLIST"] = ".*/__pycache__/.*,.*\\.pyc,.*\\.pyo,.*\\.pyd"
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Полное отключение при проблемах

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        padding-top: 0rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
    }

    .block {
        border: 1px solid #ccc;
        background-color: #e0e0e0;
        border-radius: 4px;
    }

    .stDataFrame {
        height: 250px !important;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

header_cols = st.columns([2, 10])
with header_cols[0]:
    page = st.radio(
        "Выбор страницы",
        ["Прогнозирование", "Анализ данных"]
    )

with header_cols[1]:
    df, outlier_percentage = upload()

if page == "Прогнозирование":
    st.set_page_config(page_title="Прогнозирование", layout="wide")

    top_cols = st.columns([2, 2, 2, 2, 2])

    if df is not None:
        current_df_hash = hash(pd.util.hash_pandas_object(df, index=True).sum())
        if st.session_state.get('last_df_hash') != current_df_hash:
            st.session_state.clear()
            st.session_state['filtered_df'] = df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash

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

    main_cols = st.columns([9, 3])

    with main_cols[0]:
        training_df = None

        if df is not None and not df.empty:
            if 'filtered_df' in st.session_state:
                filtered_df = st.session_state['filtered_df']
            else:
                filtered_df = df

            selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())

            if selected_sensors:
                training_df = plot_interactive_with_selection(filtered_df, selected_sensors=selected_sensors, flag=True)
            else:
                st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
        else:
            st.markdown("""<div class="block" style="height: 420px;"></div>""", unsafe_allow_html=True)

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
            if 'filtered_df' in st.session_state:
                filtered_df = st.session_state['filtered_df']
            else:
                filtered_df = df
            if filtered_df is not None:
                st.dataframe(filtered_df)
            else:
                st.markdown("Нет информации", unsafe_allow_html=True)

    with main_cols[1]:
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

        st.markdown("#### Выбрать модель")
        option = st.selectbox(
            "Выберите параметр",
            ["sarima", "ets", "prophet"],
            key="model_select"
        )

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
                    params['error_type'] = st.selectbox(
                        "Тип ошибки",
                        ["add", "mul"],
                        key="error_type"
                    )
                    params['trend_type'] = st.selectbox(
                        "Тип тренда",
                        ["None", "add", "mul"],
                        key="trend_type"
                    )
                    params['season_type'] = st.selectbox(
                        "Тип сезона",
                        ["None", "add", "mul"],
                        key="season_type"
                    )
                with dialog_cols[1]:
                    params['seasonal_periods'] = st.number_input("Сезонность", key="seasonal_periods", value=0, step=1, format="%d")
                    params['damped_trend'] = st.selectbox(
                        "Дампируется ли тренд",
                        ["True", "False"],
                        key="damped_trend"
                    ) == "True"
            elif option == "prophet":
                with dialog_cols[0]:
                    params['growth'] = st.selectbox(
                        "Тип тренда",
                        ["linear", "logistic"],
                        key="growth"
                    )
                    params['seasonality_mode'] = st.selectbox(
                        "Режим моделирования сезонных компонент",
                        ["additive", "multiplicative"],
                        key="seasonality_mode"
                    )
                    params['yearly_seasonality'] = st.selectbox(
                        "Настройка годовой сезонности",
                        ["True", "False"],
                        key="yearly_seasonality"
                    ) == "True"
                    params['weekly_seasonality'] = st.selectbox(
                        "Настройка недельной сезонности",
                        ["True", "False"],
                        key="weekly_seasonality"
                    ) == "True"
                with dialog_cols[1]:
                    params['daily_seasonality'] = st.selectbox(
                        "Настройка дневной сезонности",
                        ["True", "False"],
                        key="daily_seasonality"
                    ) == "True"
                    params['seasonality_prior_scale'] = st.number_input("Выраженность сезонных компонент", key="seasonality_prior_scale", value=10.0)
                    params['changepoint_prior_scale'] = st.number_input("Чувствительность автоматического механизма обнаружения точек излома в тренде временного ряда", key="changepoint_prior_scale", value=0.05)

            if st.button("Подтвердить параметры", key=f"{option}_start"):
                # Сохраняем параметры в session_state
                st.session_state['params'] = params
                st.rerun()

        @st.dialog("Выберите количество предсказаний")
        def select_duration():
            st.markdown("Задайте количество предсказаний")
            duration = st.number_input("Количество предсказаний", min_value=1, value=1, step=1, format="%d")
            
            if st.button("Подтвердить", key="confirm_duration"):
                st.session_state['duration'] = duration
                st.rerun()
        
        @st.dialog("Выберите сезонность")
        def select_seasonality():
            st.markdown("Задайте период сезонности")
            params = {}  # Получаем существующие параметры или создаем пустой словарь
            if option in ["sarima", "ets"]:
                if option == "sarima":
                    params['S'] = st.number_input("Сезонность", min_value=1, value=12, step=1, format="%d")
                elif option == "ets":
                    params['seasonal_periods'] = st.number_input("Сезонность", min_value=1, value=12, step=1, format="%d")
            
            if st.button("Подтвердить", key="confirm_seasonality"):
                st.session_state['params'] = params
                st.rerun()

        CB_par = st.checkbox("Включить автоподбор параметров")

        st.markdown(" ")
        if st.button("Выбрать количество предсказаний"):
            select_duration()

        if CB_par and option in ["sarima", "ets"]:
            if st.button("Выбрать сезонность", key="select_seasonality_button"):
                select_seasonality()

        if not CB_par:
            if st.button("Выбрать параметры модели"):
                show_confirmation_dialog()

        if st.button("Начать прогнозирование"):
            if training_df is not None and not training_df.empty and target_sensor:
                if 'duration' not in st.session_state:
                    st.error("Ошибка: Сначала выберите количество предсказаний!")
                elif not CB_par and 'params' not in st.session_state:
                    st.error("Ошибка: Сначала выберите параметры модели!")
                elif CB_par and option in ["sarima", "ets"] and 'params' not in st.session_state:
                    st.error("Ошибка: Для выбранной модели с автоподбором необходимо указать сезонность!")
                else:
                    # Формируем df_train с только target_sensor, переименованным в 'sensor'
                    df_train = training_df[[target_sensor]].copy()
                    df_train = df_train.rename(columns={target_sensor: 'sensor'})
                    
                    # Проверяем, что индекс является DatetimeIndex
                    if not isinstance(df_train.index, pd.DatetimeIndex):
                        st.error("Ошибка: Индекс df_train должен быть типа DatetimeIndex")
                        df_train = None

                    # Определяем последнюю временную метку в df_train
                    end_training_time = df_train.index.max() if df_train is not None and not df_train.empty else None
                    
                    # Формируем df_test как остаток датасета с только target_sensor, переименованным в 'sensor'
                    df_test = None
                    if df is not None and not df.empty and end_training_time is not None:
                        df_test = df[df.index > end_training_time][[target_sensor]]
                        df_test = df_test.rename(columns={target_sensor: 'sensor'})
                        if df_test.empty:
                            st.warning("Предупреждение: df_test пуст, так как training_df охватывает весь временной диапазон df.")

                    if df_train is not None:
                        params = st.session_state.get('params', {})

                        # Формируем payload
                        payload = create_model_payload(
                            model_type=option,
                            auto_params=CB_par,
                            duration=st.session_state['duration'],
                            df_train=df_train,
                            df_test=df_test,
                            params=params
                        )
                        
                        # Выполняем запрос к API
                        try:
                            response = asyncio.run(get_prediction(payload))
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state['forecast_result'] = result
                                st.success(f"Прогноз выполнен для {target_sensor} с автоматическим подбором параметров!" if CB_par else f"Прогноз выполнен для {target_sensor}!")
                            else:
                                st.error(f"Ошибка API: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Ошибка при выполнении прогноза: {str(e)}")
                    else:
                        st.error("Ошибка: Не удалось подготовить df_train для прогнозирования.")
            else:
                st.error("Ошибка: Загрузите DataFrame, выберите данные для обучения и целевую переменную.")

    result = st.session_state.get('forecast_result', None)
    if result:
        st.markdown(f"Результат прогнозирования: {result}")


elif page == "Анализ данных":
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

    st.set_page_config(page_title="Анализ данных", layout="wide")

    if df is not None and not df.empty:
        if 'filtered_df' not in st.session_state:
            st.session_state['filtered_df'] = df
        filtered_df = st.session_state['filtered_df']

        button_cols = st.columns(2)
        with button_cols[0]:
            if st.button("Применить фильтр", key="apply_filter_analysis"):
                if 'sensor_editor_temp' in st.session_state:
                    if st.session_state['sensor_editor_temp']:
                        st.session_state['selected_sensors'] = st.session_state['sensor_editor_temp']
                        st.session_state['filtered_df'] = df[st.session_state['selected_sensors']]
                    else:
                        st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
                        st.session_state['selected_sensors'] = []
                else:
                    st.session_state['selected_sensors'] = df.columns.tolist()
                    st.session_state['filtered_df'] = df
                st.rerun()

        with button_cols[1]:
            if st.button("Сбросить фильтр", key="reset_filter_analysis"):
                st.session_state['filtered_df'] = df
                st.session_state['selected_sensors'] = df.columns.tolist()
                st.session_state['sensor_editor_temp'] = df.columns.tolist()
                st.rerun()

        selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())

        if selected_sensors:
            plot_interactive_with_selection(filtered_df, selected_sensors=selected_sensors, flag=False)
        else:
            st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
    else:
        st.markdown("""<div class="block" style="height: 420px;"></div>""", unsafe_allow_html=True)

    lower_cols = st.columns([6, 6])
    with lower_cols[1]:
        st.markdown("#### Параметры:")
        if df is not None and not df.empty:
            sensor_df = pd.DataFrame({
                'Датчики': df.columns,
                'Отображать': [col in st.session_state.get('selected_sensors', df.columns.tolist()) for col in df.columns]
            })
            edited_sensor_df = st.data_editor(sensor_df, key="sensor_selector_analysis")
            st.session_state['sensor_editor_temp'] = edited_sensor_df[edited_sensor_df['Отображать']]['Датчики'].tolist()
        else:
            st.markdown("Нет информации", unsafe_allow_html=True)

    with lower_cols[0]:
        st.markdown("#### Предпросмотр:")
        if df is not None:
            st.dataframe(filtered_df)
        else:
            st.markdown("Нет информации", unsafe_allow_html=True)

    panel_col1, panel_col2 = st.columns(2)
    with panel_col1:
        if st.button("Heatmap и pairplot"):
            st.session_state['active_panel'] = "heatmap_pairplot"
    with panel_col2:
        if st.button("Статистика датчиков"):
            st.session_state['active_panel'] = "sensor_statistics"

    if 'active_panel' not in st.session_state:
        st.session_state['active_panel'] = "heatmap_pairplot"

    if st.session_state['active_panel'] == "heatmap_pairplot":
        st.markdown("### Heatmap и pairplot")
        col_heat, col_pair = st.columns(2)
        with col_heat:
            show_heatmap(df)
        with col_pair:
            show_pairplot(df)
    elif st.session_state['active_panel'] == "sensor_statistics":
        st.markdown("### Статистика датчиков")
        features = filtered_df.columns.tolist() if df is not None and not df.empty else []
        selected_feature = st.selectbox("Выберите признак для подробной информации о нём", features, index=0)
        mean, median, std, minimal, maximum = info_about_feature(filtered_df, selected_feature)
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        with col1:
            st.subheader("Среднее")
            st.write(f"{mean:.3f}")
        with col2:
            st.subheader("СКО")
            st.write(f"{std:.3f}")
        with col3:
            st.subheader("Медиана")
            st.write(f"{median:.3f}")
        with col4:
            st.subheader("Мин. знач.")
            st.write(f"{minimal:.3f}")
        with col5:
            st.subheader("Макс. знач.")
            st.write(f"{maximum:.3f}")

        forecasting(df, column=selected_feature)

        col_hist, col_autocorr = st.columns(2)

        with col_hist:
            show_hist(df, selected_feature, bins=100)

        with col_autocorr:
            show_autocorrelation(df, selected_feature)