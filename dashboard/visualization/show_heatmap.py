import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dashboard.utils.validate_data import validate_data

def select_heatmap_features(df: pd.DataFrame) -> List[str]:
    """
    UI-компонент для выбора признаков для тепловой карты
    """
    if df.shape[1] > 10:
        st.warning("Слишком много признаков. Рекомендуется выбрать не более 10 признаков для читаемой тепловой карты.")
    return st.multiselect(
        "Выберите признаки для отображения тепловой карты:",
        options=df.columns.tolist(),
        default=df.columns[:5].tolist()
    )

def calculate_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Вычисляет корреляционную матрицу по выбранным признакам
    """
    if len(columns) < 2:
        st.info("Выберите как минимум два признака для отображения тепловой карты.")
        return None
    return df[columns].corr()

def draw_heatmap(corr_matrix: pd.DataFrame) -> None:
    """
    Рисует тепловую карту по корреляционной матрице
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        ax=ax
    )
    ax.set_title("Корреляционная матрица", fontsize=14)
    st.pyplot(fig)

def show_heatmap(df: pd.DataFrame) -> None:
    """
    Основная функция для отображения тепловой карты корреляций
    """
    if not validate_data(df):
        return
    selected_columns = select_heatmap_features(df)
    corr_matrix = calculate_correlation_matrix(df, selected_columns)
    if corr_matrix is not None:
        draw_heatmap(corr_matrix)

