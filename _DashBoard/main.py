import streamlit as st
import pandas as pd
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

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу", ["Прогнозирование", "Анализ данных"])
st.markdown(" ")
st.markdown(" ")

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

df, outlier_percentage = upload()
if page == "Прогнозирование":
    st.set_page_config(page_title="Прогнозирование", layout="wide")

    top_cols = st.columns([2, 2, 2, 2, 2])

    if df is not None:
        current_df_hash = hash(pd.util.hash_pandas_object(df, index=True).sum())
        if st.session_state.get('last_df_hash') != current_df_hash:
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

        st.markdown("#### Выбрать модель")
        st.markdown("""
        <select style="width:100%; padding:8px; font-size:16px; font-family:Montserrat; font-weight:600; background-color: #2a2a2a; color: #ffffff; border: 1px solid #333; border-radius: 4px;">
        <option>Модель 1</option>
        <option>Модель 2</option>
        <option>Модель 3</option>
        </select>
        """, unsafe_allow_html=True)

        st.checkbox("Включить автоподбор параметров")

        st.markdown(" ")
        if st.button("Начать прогнозирование"):
            if training_df is not None and target_sensor is not None:
                st.write(f"Прогнозирование выполнено на основе {training_df.shape[0]} записей с целевой переменной: {target_sensor}")
            else:
                st.error("Загрузите DataFrame, выберите интервал, данные для обучения и целевую переменную.")

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

    st.subheader("График")
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