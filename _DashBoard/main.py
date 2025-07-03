import streamlit as st
import pandas as pd
from dashboard.upload import upload
from dashboard.info_about_dataframe import info_about_dataframe
from dashboard.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.plot_interactive_with_selection import plot_interactive_with_selection

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу", ["Прогнозирование", "Анализ данных"])
st.markdown(" ")
st.markdown(" ")

# Устанавливаем шрифт и убираем отступы
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

    /* Фиксированная высота для data_editor */
    .stDataFrame {
        height: 250px !important;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

if page == "Прогнозирование":
    # Настройка страницы
    st.set_page_config(page_title="Прогнозирование", layout="wide")

    # Верхняя панель — 12 колонок: 2+2+2+2+2+2
    top_cols = st.columns([2, 2, 2, 2, 2, 2])

    with top_cols[0]:
        df = upload()

    features_size, tuples_size, first_tuple, last_tuple = info_about_dataframe(df)

    with top_cols[1]:
        st.markdown(f"Кол-во записей: **{tuples_size if tuples_size is not None else 'Нет информации'}**")

    with top_cols[2]:
        st.markdown(f"Количество признаков: **{features_size if features_size is not None else 'Нет информации'}**")

    with top_cols[3]:
        st.markdown(f"Первая запись: **{first_tuple if first_tuple is not None else 'Нет информации'}**")

    with top_cols[4]:
        st.markdown(f"Последняя запись: **{last_tuple if last_tuple is not None else 'Нет информации'}**")

    # Основной блок — 9 + 3 (левая панель + правая панель)
    main_cols = st.columns([9, 3])

    with main_cols[0]:
        # График
        training_df = None
        if df is not None and not df.empty:
            # Отображение графика с полным df, если filtered_df еще не определен
            if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
                filtered_df = st.session_state['filtered_df']
            else:
                filtered_df = df
            if 'selected_sensors' in st.session_state and st.session_state['selected_sensors']:
                training_df = plot_interactive_with_selection(filtered_df, selected_sensors=st.session_state['selected_sensors'])
            else:
                training_df = plot_interactive_with_selection(filtered_df, selected_sensors=df.columns.tolist())  # По умолчанию все датчики
        else:
            st.markdown(
                """<div class="block" style="height: 420px;"></div>""",
                unsafe_allow_html=True
            )

        # Нижняя часть: График, Параметры, Предпросмотр в одной строке
        lower_cols = st.columns([6, 6])  # Соотношение 6:2:4 для графика, параметров и предпросмотра
        with lower_cols[1]:
            st.markdown("#### Параметры:")
            if df is not None and not df.empty:
                sensor_df = pd.DataFrame({'Датчики': df.columns, 'Отображать': [True] * len(df.columns)})
                edited_sensor_df = st.data_editor(sensor_df, key="sensor_selector", on_change=lambda: st.rerun())
                if 'selected_sensors' not in st.session_state or st.session_state['selected_sensors'] != edited_sensor_df[edited_sensor_df['Отображать']]['Датчики'].tolist():
                    st.session_state['selected_sensors'] = edited_sensor_df[edited_sensor_df['Отображать']]['Датчики'].tolist()
            else:
                st.markdown(
                    "Нет информации",
                    unsafe_allow_html=True
                )
        with lower_cols[0]:
            st.markdown("#### Предпросмотр:")
            if df is not None:
                st.dataframe(df)
            else:
                st.markdown(
                    "Нет информации",
                    unsafe_allow_html=True
                )

    with main_cols[1]:
        # Правая панель — 3 колонки
        st.markdown("## Прогнозирование")
        st.markdown("#### Область прогнозирования")

        # Создаем колонки для дат и времени
        start_cols = st.columns(2)
        with start_cols[0]:
            start_datetime = start_date(df, context="display_panel")
        with start_cols[1]:
            end_datetime = end_date(df, context="display_panel")

        if start_datetime is not None and end_datetime is not None:
            filtered_df = filter_dataframe(start_datetime, end_datetime, df)
            st.session_state['filtered_df'] = filtered_df
        else:
            if 'filtered_df' in st.session_state:
                del st.session_state['filtered_df']

        # Целевые параметры (выпадающий список с одним выбором)
        st.markdown("#### Целевые параметры:")
        target_sensor = None
        if df is not None and not df.empty:
            target_sensor = st.selectbox("Выберите целевой признак", options=df.columns.tolist(), index=0, key="target_sensor")
        else:
            st.markdown(
                    "Нет информации",
                    unsafe_allow_html=True
                )

        st.markdown("#### Выбрать модель")
        st.markdown("""
    <select style="width:100%; padding:8px; font-size:16px; font-family:Montserrat; font-weight:600; background-color: #2a2a2a; color: #ffffff; border: 1px solid #333; border-radius: 4px;">
    <option>Модель 1</option>
    <option>Модель 2</option>
    <option>Модель 3</option>
    </select>
    """, unsafe_allow_html=True)

        st.markdown(" ")

        st.checkbox("Включить автоподбор параметров")

        st.markdown(" ")
        if st.button("Начать прогнозирование"):
            if training_df is not None and target_sensor is not None:
                st.write(f"Прогнозирование выполнено на основе {training_df.shape[0]} записей с целевой переменной: {target_sensor}")
            else:
                st.error("Загрузите DataFrame, выберите интервал, данные для обучения и целевую переменную.")

elif page == "Анализ данных":

     # Верхняя панель — 12 колонок: 2+2+2+2+2+2
    top_cols = st.columns([2, 2, 2, 2, 2, 2])

    with top_cols[0]:
        df = upload()

    features_size, tuples_size, first_tuple, last_tuple = info_about_dataframe(df)

    with top_cols[1]:
        st.markdown(f"Кол-во записей: **{tuples_size if tuples_size is not None else 'Нет информации'}**")

    with top_cols[2]:
        st.markdown(f"Количество признаков: **{features_size if features_size is not None else 'Нет информации'}**")

    with top_cols[3]:
        st.markdown(f"Первая запись: **{first_tuple if first_tuple is not None else 'Нет информации'}**")

    with top_cols[4]:
        st.markdown(f"Последняя запись: **{last_tuple if last_tuple is not None else 'Нет информации'}**")


    # Настройка страницы
    st.set_page_config(page_title="Анализ данных", layout="wide")

    # Главная панель
    main_cols = st.columns([8, 4])

    # Левая колонка: График
    with main_cols[0]:
        st.subheader("График")
        st.markdown("<div style='height: 400px; background-color: lightgray;'></div>", unsafe_allow_html=True)

    # Правая колонка: Корреляция
    with main_cols[1]:
        st.subheader("Корреляция")
        st.markdown("<div style='height: 400px; background-color: lightgray;'></div>", unsafe_allow_html=True)
    
    data_cols = st.columns([8, 4])

    # Средние статистики
    with data_cols[0]:

        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        with col1:
            st.subheader("Среднее")
            st.write("1234")

        with col2:
            st.subheader("СКО")
            st.write("124124")

        with col3:
            st.subheader("Медиана")
            st.write("123124")

        with col4:
            st.subheader("Мин. знач.")
            st.write("123142")
        with col5:
            st.subheader("Макс. знач.")
            st.write("123123")
        
        # Предпросмотр и параметры
        col1, col2 = st.columns([4, 4])

        with col1:
            st.subheader("Предпросмотр:")
            st.markdown("<div style='height: 300px; background-color: lightgray;'></div>", unsafe_allow_html=True)

        with col2:
            st.subheader("Параметры:")
            st.markdown("<div style='height: 300px; background-color: lightgray;'></div>", unsafe_allow_html=True)

    # Парные графики
    with data_cols[1]:
        st.subheader("Парные графики")
        st.markdown("<div style='height: 400px; background-color: lightgray;'></div>", unsafe_allow_html=True)


    # Конкретизация параметра
    st.markdown("<h2 style='text-align: center;'>Конкретизация параметра</h2>", unsafe_allow_html=True)
     
    col11, col12 = st.columns([2, 4])

    with col11:
        st.subheader("Выбор параметра")
        st.markdown("<div style='height: 300px; background-color: lightgray;'></div>", unsafe_allow_html=True)

    with col12:
        st.subheader("Тренд/Сезонность")
        st.markdown("<div style='height: 300px; background-color: lightgray;'></div>", unsafe_allow_html=True)

    # Распределение признака и автокорреляция
     
    col13, col14 = st.columns(2)

    with col13:
        st.subheader("Распределение признака")
        st.markdown("<div style='height: 400px; background-color: lightgray;'></div>", unsafe_allow_html=True)

    with col14:
        st.subheader("Автокорреляция")
        st.markdown("<div style='height: 400px; background-color: lightgray;'></div>", unsafe_allow_html=True)