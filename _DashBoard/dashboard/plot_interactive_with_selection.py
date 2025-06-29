import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from datetime import datetime
import numpy as np
from scipy import stats

def plot_interactive_with_selection(filtered_df: pd.DataFrame, selected_sensors: list):
    """
    Создаёт интерактивный график зависимости значений датчиков от времени с возможностью выделения одного интервала.
    Выделенные точки, включая аномалии, отображаются сразу, предыдущее выделение сбрасывается при новом выборе.
    Аномалии подсвечиваются крестами того же цвета, что и датчик, и включаются в тренировочный датасет.
    
    Параметры:
    - filtered_df (pd.DataFrame): Отфильтрованный DataFrame с DatetimeIndex и столбцами-датчиками.
    - selected_sensors (list): Список выбранных датчиков для отображения.
    
    Возвращает:
    - pd.DataFrame: Датасет с выделенными точками (индекс — timestamp, колонки — tensor1, tensor2, ...).
    """
    if filtered_df is None or filtered_df.empty:
        st.warning("Нет данных для визуализации.")
        return None

    if not selected_sensors:
        st.info("Выберите хотя бы один датчик в таблице параметров.")
        return None

    # Инициализация состояния сессии
    if 'dragmode' not in st.session_state:
        st.session_state.dragmode = 'select'
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = {}

    # Создание контейнера для графика
    plot_placeholder = st.empty()

    # Функция для построения графика
    def build_plot(selected_points_data):
        fig = go.Figure()

        # Цвета для датчиков
        colors = {}
        for i, sensor in enumerate(selected_sensors):
            colors[sensor] = f"hsl({int(i * 360 / max(1, len(selected_sensors)))},70%,50%)"

        # Вычисление аномалий
        outliers_mask = {}
        for sensor in selected_sensors:
            if sensor in filtered_df.columns:
                z_scores = np.abs(stats.zscore(filtered_df[sensor].dropna()))
                outliers_mask[sensor] = z_scores > 3

        # Обновление выделенных точек
        selected_x = set(x for sensor in selected_sensors for x, _ in selected_points_data.get(sensor, [])) if selected_points_data else set()

        # Отрисовка линий, аномалий и выделенных точек
        for sensor in selected_sensors:
            if sensor in filtered_df.columns:
                y = filtered_df[sensor]
                mask_selected = filtered_df.index.isin(selected_x) if selected_x else pd.Series(False, index=filtered_df.index)
                mask_outliers = outliers_mask.get(sensor, pd.Series(False, index=filtered_df.index))

                # Подсветка аномалий крестами
                if mask_outliers.any():
                    fig.add_trace(go.Scatter(
                        x=filtered_df.index[mask_outliers],
                        y=y[mask_outliers],
                        mode='markers',
                        name=f"{sensor} (аномалии)",
                        marker=dict(color=colors[sensor], size=10, symbol='cross'),
                        opacity=1.0,
                        showlegend=True
                    ))

                # Подсветка выделенных точек (включая аномалии)
                if mask_selected.any():
                    fig.add_trace(go.Scatter(
                        x=filtered_df.index[mask_selected],
                        y=y[mask_selected],
                        mode='markers',
                        name=f"{sensor} (выделено)",
                        marker=dict(color=colors[sensor], size=10),
                        opacity=1.0
                    ))

                # Линии для остальных точек
                fig.add_trace(go.Scatter(
                    x=filtered_df.index[~mask_selected],
                    y=y[~mask_selected],
                    mode='lines+markers',
                    name=sensor,
                    line=dict(color=colors[sensor], width=2),
                    opacity=0.7,
                    showlegend=True
                ))

        # Настройка графика
        fig.update_layout(
            dragmode=st.session_state.dragmode,
            hovermode='x unified',
            height=420,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title='Время',
            yaxis_title='Значение датчиков',
            legend_title="Датчики"
        )

        return fig

    # Первоначальный рендеринг графика
    with plot_placeholder:
        fig = build_plot(st.session_state.selected_points)
        selected_points = plotly_events(fig, select_event=True, override_height=420, click_event=False)

    # Обработка выделенных точек
    training_df = None
    if selected_points:
        # Очищаем предыдущие выделенные точки
        st.session_state.selected_points = {}

        # Обработка нового выделения
        new_selected_points = {}
        for point in selected_points:
            x = pd.Timestamp(point['x']).tz_localize(None)
            for sensor in selected_sensors:
                if sensor in filtered_df.columns:
                    y = filtered_df[sensor].loc[filtered_df.index == x].iloc[0] if x in filtered_df.index else np.nan
                    if not np.isnan(y):
                        if sensor not in new_selected_points:
                            new_selected_points[sensor] = []
                        new_selected_points[sensor].append((x, y))

        # Сохраняем только текущий интервал
        st.session_state.selected_points = new_selected_points

        # Перерисовываем график с новым выделением
        with plot_placeholder:
            fig = build_plot(st.session_state.selected_points)
            plotly_events(fig, select_event=True, override_height=420, click_event=False)

    # Подготовка датасета для обучения
    if st.session_state.selected_points:
        if st.button("Использовать данные для обучения"):
            export_data = {}
            timestamps = sorted(set(x for sensor in st.session_state.selected_points for x, _ in st.session_state.selected_points[sensor]))
            for sensor in selected_sensors:
                values = [next((y for px, y in st.session_state.selected_points.get(sensor, []) if px == x), np.nan) for x in timestamps]
                export_data[f'tensor_{selected_sensors.index(sensor) + 1}'] = values
            training_df = pd.DataFrame(export_data, index=timestamps)
            st.success("Данные подготовлены для обучения.")
            st.write("Пример данных для обучения:")
            st.dataframe(training_df)

    # Кнопка для очистки выделения
    if st.button("Очистить выделение"):
        st.session_state.selected_points = {}
        with plot_placeholder:
            fig = build_plot(st.session_state.selected_points)
            plotly_events(fig, select_event=True, override_height=420, click_event=False)

    return training_df