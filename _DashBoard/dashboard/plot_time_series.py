import streamlit as st
import pandas as pd
import plotly.express as px

def plot_time_series(df):
    """
    Построение интерактивного графика временного ряда с выбором признаков.
    
    Parameters:
    - df: pandas DataFrame, где индекс — даты, а столбцы — числовые признаки
    """
    # Проверка, что индекс имеет тип datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        st.error("Индекс должен быть в формате datetime.")
        return

    # Выбор числовых признаков
    features = df.columns.tolist()
    
    # Выбор признаков для отображения
    selected_features = st.multiselect("Выберите признаки для отображения", features, default=features)

    # Проверка, выбраны ли признаки
    if selected_features:
        # Создаем график с помощью Plotly
        fig = px.line(df, x=df.index, y=selected_features)
        
        # Настройка интерактивной легенды и стиля графика
        fig.update_traces(mode='lines+markers')
        fig.update_layout(
            legend_title_text='Признаки',
            hovermode="x unified"
        )

        # Отображение графика
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Выберите хотя бы один признак.")