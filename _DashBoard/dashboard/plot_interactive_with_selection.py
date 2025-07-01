import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from datetime import datetime
import numpy as np
from scipy import stats

def plot_interactive_with_selection(filtered_df: pd.DataFrame, selected_sensors: list, flag: bool):
    if filtered_df is None or filtered_df.empty:
        st.warning("Нет данных для визуализации.")
        return None

    if not selected_sensors:
        st.error("Ошибка: Выберите хотя бы один датчик для отображения графика.")
        return None

    if flag:
        if 'dragmode' not in st.session_state:
            st.session_state.dragmode = 'select'
        if 'selected_points' not in st.session_state:
            st.session_state.selected_points = {}
        if 'training_triggered' not in st.session_state:
            st.session_state.training_triggered = False
        if 'plot_key' not in st.session_state:
            st.session_state.plot_key = 0
    else:
        st.session_state.dragmode = 'pan'

    plot_placeholder = st.empty()

    def build_plot(selected_points_data=None):
        fig = go.Figure()

        colors = {}
        for i, sensor in enumerate(selected_sensors):
            colors[sensor] = f"hsl({int(i * 360 / max(1, len(selected_sensors)))},70%,50%)"

        outliers_mask = {}
        for sensor in selected_sensors:
            if sensor in filtered_df.columns:
                z_scores = np.abs(stats.zscore(filtered_df[sensor].dropna()))
                outliers_mask[sensor] = z_scores > 3

        selected_x = set()
        if flag and selected_points_data:
            selected_x = set(x for sensor in selected_sensors for x, _ in selected_points_data.get(sensor, []))

        for sensor in selected_sensors:
            if sensor in filtered_df.columns:
                y = filtered_df[sensor]
                mask_selected = filtered_df.index.isin(selected_x) if selected_x else pd.Series(False, index=filtered_df.index)
                mask_outliers = outliers_mask.get(sensor, pd.Series(False, index=filtered_df.index))

                marker_symbol = ['cross' if outlier else 'circle' for outlier in mask_outliers]

                fig.add_trace(go.Scatter(
                    x=filtered_df.index,
                    y=y,
                    mode='lines+markers',
                    name=sensor,
                    line=dict(color=colors[sensor], width=2),
                    marker=dict(
                        color=colors[sensor],
                        size=10,
                        symbol=marker_symbol,
                        opacity=1.0 if any(mask_outliers) else 0.7
                    ),
                    opacity=0.7,
                    showlegend=True
                ))

                if flag and mask_selected.any():
                    fig.add_trace(go.Scatter(
                        x=filtered_df.index[mask_selected],
                        y=y[mask_selected],
                        mode='markers',
                        name=f"{sensor} (выделено)",
                        marker=dict(color=colors[sensor], size=10, symbol='circle'),
                        opacity=1.0,
                        showlegend=True
                    ))

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

    with plot_placeholder:
        if flag:
            fig = build_plot(st.session_state.selected_points)
            selected_points = plotly_events(fig, select_event=True, override_height=420, click_event=False, key=f"plotly_{st.session_state.plot_key}")
        else:
            fig = build_plot()
            plotly_events(fig, select_event=False, override_height=420, click_event=False, key=f"plotly_{st.session_state.plot_key}")

    training_df = None
    if flag and selected_points:
        st.session_state.selected_points = {}
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
        st.session_state.selected_points = new_selected_points
        st.session_state.training_triggered = False
        st.session_state.plot_key += 1
        with plot_placeholder:
            fig = build_plot(st.session_state.selected_points)
            plotly_events(fig, select_event=True, override_height=420, click_event=False, key=f"plotly_{st.session_state.plot_key}")

    main_col = st.container()

    with main_col:
        if flag and st.session_state.selected_points:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Очистить выделение", key="clear_selection"):
                    st.session_state.selected_points = {}
                    st.session_state.training_triggered = False
                    st.session_state.plot_key += 1
                    st.rerun()
            with col2:
                if st.button("Использовать данные для обучения", key="use_for_training"):
                    st.session_state.training_triggered = True
                    st.session_state.plot_key += 1

        if flag and st.session_state.selected_points and st.session_state.training_triggered:
            export_data = {}
            timestamps = sorted(set(x for sensor in st.session_state.selected_points for x, _ in st.session_state.selected_points[sensor]))
            for sensor in selected_sensors:
                values = [next((y for px, y in st.session_state.selected_points.get(sensor, []) if px == x), np.nan) for x in timestamps]
                export_data[sensor] = values
            training_df = pd.DataFrame(export_data, index=timestamps)
            st.success("Данные подготовлены для обучения.")
            st.dataframe(training_df)

    return training_df