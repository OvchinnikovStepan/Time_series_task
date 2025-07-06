import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from datetime import datetime
import numpy as np

def plot_interactive_with_selection(filtered_df: pd.DataFrame, selected_sensors: list):
    """
    Создаёт интерактивный график зависимости значений датчиков от времени с возможностью выделения точек.
    - Горизонтальная ось (x): время (индекс DatetimeIndex).
    - Вертикальная ось (y): значения датчиков.
    - Поддерживает выбор датчиков и выделение точек.
    - Возвращает датасет с выделенными точками в формате с тензорами (tensor1, tensor2, ...).
    
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

    # Создание интерактивного графика
    fig = go.Figure()

    # Цвета для датчиков
    colors = {}
    for i, sensor in enumerate(selected_sensors):
        colors[sensor] = f"hsl({int(i * 360 / len(selected_sensors))},70%,50%)"

    # Отрисовка линий без выделения
    for sensor in selected_sensors:
        if sensor in filtered_df.columns:  # Проверка наличия датчика
            y = filtered_df[sensor]
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=y,
                mode='lines+markers',
                name=sensor,
                line=dict(color=colors[sensor], width=2),
                opacity=0.7
            ))

    fig.update_layout(
        dragmode='select',
        hovermode='x unified',
        height=420,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title='Время',
        yaxis_title='Значение датчиков',
        legend_title="Датчики"
    )

    # Отображение графика с возможностью выделения
    selected_points = plotly_events(fig, select_event=True, override_height=420, click_event=False)

    # Обработка выделенных точек
    selected_data = {}
    if selected_points:
        for point in selected_points:
            x = pd.Timestamp(point['x'])  # Без tz_localize, если данные уже без временных зон
            for sensor in selected_sensors:
                if sensor in filtered_df.columns:
                    y = filtered_df[sensor].loc[filtered_df.index == x].iloc[0] if x in filtered_df.index else np.nan
                    if not np.isnan(y):
                        if sensor not in selected_data:
                            selected_data[sensor] = []
                        selected_data[sensor].append((x, y))

    # Подготовка датасета для обучения
    training_df = None
    if selected_data:
        if st.button("Использовать данные для обучения"):
            export_data = {}
            timestamps = sorted(set(x for sensor in selected_data for x, _ in selected_data[sensor]))
            for sensor in selected_sensors:
                values = [next((y for px, y in selected_data.get(sensor, []) if px == x), np.nan) for x in timestamps]
                export_data[f'tensor_{selected_sensors.index(sensor) + 1}'] = values
            training_df = pd.DataFrame(export_data, index=timestamps)
            st.success("Данные подготовлены для обучения.")

    return training_df