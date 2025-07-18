import streamlit as st

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
    </style>
""", unsafe_allow_html=True)

# Настройка страницы
st.set_page_config(page_title="Прогнозирование", layout="wide")

# Верхняя панель — 12 колонок: 2+2+2+2+2+2
top_cols = st.columns([2, 2, 2, 2, 2, 2])

with top_cols[0]:
    st.button("# Загрузить файл")

with top_cols[1]:
    st.markdown("Кол-во записей: **450**")

with top_cols[2]:
    st.markdown("Количество признаков: **9**")

with top_cols[3]:
    st.markdown("Первая запись: **30.01.25**")

with top_cols[4]:
    st.markdown("Последняя запись: **30.01.25**")

# Основной блок — 9 + 3 (левая панель + правая панель)
main_cols = st.columns([9, 3])

with main_cols[0]:
    # График
    st.markdown(
        """<div class="block" style="height: 420px;"></div>""",
        unsafe_allow_html=True
    )

    # Предпросмотр и параметры — нижняя часть 9 колонок: 6+6
    preview_cols = st.columns([6, 6])

    with preview_cols[0]:
        st.markdown("#### Предпросмотр:")
        st.markdown(
            """<div class="block" style="height: 300px;"></div>""",
            unsafe_allow_html=True
        )

    with preview_cols[1]:
        st.markdown("#### Параметры:")
        st.markdown(
            """<div class="block" style="height: 300px;"></div>""",
            unsafe_allow_html=True
        )

with main_cols[1]:
    # Правая панель — 3 колонки
    st.markdown("##  Прогнозирование")
    st.markdown("#### Область прогнозирования")

    start_cols = st.columns(2)
    with start_cols[0]:
        st.button("Начало")
    with start_cols[1]:
        st.button("Конец")

    st.markdown("#### Целевые параметры:")
    st.markdown(
        """<div class="block" style="height: 184px;"></div>""",
        unsafe_allow_html=True
    )
    st.markdown("#### Выбрать модель")
    st.markdown("""
<select style="width:100%; padding:8px; font-size:16px; font-family:Montserrat; font-weight:600;">
  <option>Модель 1</option>
  <option>Модель 2</option>
  <option>Модель 3</option>
</select>
""", unsafe_allow_html=True)
    st.markdown("")


    st.checkbox("Включить автоподбор параметров")

    st.markdown(" ")
    st.button("Начать прогнозирование")
