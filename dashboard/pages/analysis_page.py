import streamlit as st
import pandas as pd
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.visualization.plot_interactive_with_selection import plot_interactive_with_selection
from dashboard.visualization.show_heatmap import show_heatmap
from dashboard.data_processing.info_about_feature import info_about_feature
from dashboard.visualization.show_pairplot import show_pairplot
from dashboard.data_processing.forecasting import forecasting
from dashboard.visualization.show_hist import show_hist
from dashboard.visualization.show_autocorrelation import show_autocorrelation
from dashboard.utils.data_limiting import limit_data_to_last_points
from typing import Optional, List, Tuple
from pandas.util import hash_pandas_object

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

def handle_filter_buttons(df: pd.DataFrame) -> None:
    """
    Обрабатывает кнопки фильтрации и сброса фильтра
    """
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
            # Возвращаемся к ограниченному виду (последние 500 точек)
            if st.session_state.get('is_limited_view_analysis', False) and st.session_state.get('original_df_analysis') is not None:
                limited_df = limit_data_to_last_points(st.session_state['original_df_analysis'], 500)
                st.session_state['filtered_df'] = limited_df
            else:
                st.session_state['filtered_df'] = df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.rerun()

def render_interactive_plot(filtered_df: pd.DataFrame, selected_sensors: List[str]) -> None:
    """
    Отображает интерактивный график по выбранным сенсорам
    """
    if selected_sensors:
        plot_interactive_with_selection(filtered_df, selected_sensors=selected_sensors, flag=False)
    else:
        st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")

def render_parameter_and_preview_panel(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Отображает панель параметров и предпросмотра
    """
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

def render_heatmap_pairplot_panel(df: pd.DataFrame) -> None:
    """
    Отображает панель heatmap и pairplot
    """
    st.markdown("### Heatmap и pairplot")
    col_heat, col_pair = st.columns(2)
    with col_heat:
        show_heatmap(df)
    with col_pair:
        show_pairplot(df)

def render_sensor_statistics_panel(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Отображает панель статистики датчиков, прогнозирования, гистограммы и автокорреляции
    """
    st.markdown("### Статистика датчиков")
    features = filtered_df.columns.tolist() if df is not None and not df.empty else []
    if not features:
        st.info("Нет признаков для анализа.")
        return
    selected_feature = st.selectbox("Выберите признак для подробной информации о нём", features, index=0)
    if selected_feature:
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

def render_analysis_panels(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Отображает переключатель панелей анализа и выбранную панель
    """
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
        render_heatmap_pairplot_panel(df)
    elif st.session_state['active_panel'] == "sensor_statistics":
        render_sensor_statistics_panel(df, filtered_df)



def render_analysis_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Анализ данных"
    """
    st.set_page_config(page_title="Анализ данных", layout="wide")
    # Не отображаем информацию о датасете, если df is None
    if df is None:
        st.session_state.clear()
        return
    render_data_overview(df, outlier_percentage)
    
    # Инициализация данных при загрузке нового файла
    if df is not None and not df.empty:
        current_df_hash = hash(str(df.shape) + str(df.columns.tolist()) + str(df.index[-10:].tolist()) if len(df) > 10 else str(df.index.tolist()))
        if st.session_state.get('last_df_hash_analysis') != current_df_hash:
            # При загрузке нового файла ограничиваем данные последними 500 точками
            limited_df = limit_data_to_last_points(df, 500)
            st.session_state['filtered_df'] = limited_df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['last_df_hash_analysis'] = current_df_hash
            st.session_state['original_df_analysis'] = df  # Сохраняем оригинальный DataFrame
            st.session_state['is_limited_view_analysis'] = True  # Флаг, что отображается ограниченный вид
    
    filtered_df = st.session_state['filtered_df']
    selected_sensors = st.session_state.get('selected_sensors', filtered_df.columns.tolist())
    render_interactive_plot(filtered_df, selected_sensors)
    # Информация о текущем режиме отображения
    original_df = st.session_state.get('original_df_analysis', df)
    if original_df is not None and len(original_df) > 500:
        if st.session_state.get('is_limited_view_analysis', False):
            st.info(f"Отображаются последние 500 из {len(original_df)} записей. Используйте фильтр для просмотра других периодов.")
            if st.button("Отобразить все записи", key="show_all_data_analysis"):
                st.session_state['filtered_df'] = original_df
                st.session_state['is_limited_view_analysis'] = False
                st.rerun()
        else:
            st.info(f"Отображаются все {len(original_df)} записей. Для лучшей производительности рекомендуется использовать ограниченный вид.")
            if st.button("Отобразить последние 500 записей", key="show_limited_data_analysis"):
                limited_df = limit_data_to_last_points(original_df, 500)
                st.session_state['filtered_df'] = limited_df
                st.session_state['is_limited_view_analysis'] = True
                st.rerun()
    render_parameter_and_preview_panel(df, filtered_df)
    render_analysis_panels(df, filtered_df)