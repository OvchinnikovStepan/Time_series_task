import streamlit as st
import pandas as pd
import plotly.express as px

def show_correlation_matrix(df):
    # Вычисляем корреляционную матрицу
    corr_matrix = df.corr()

    # Преобразуем в формат long для plotly heatmap
    corr_melted = corr_matrix.reset_index().melt(id_vars='index')
    corr_melted.columns = ['Признак X', 'Признак Y', 'Корреляция']

    # Построение тепловой карты
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels=dict(x="Признак", y="Признак", color="Корреляция"),
        title="Матрица корреляции"
    )

    fig.update_layout(width=700, height=700)
    st.plotly_chart(fig, use_container_width=True)