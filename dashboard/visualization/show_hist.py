import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def build_hist_figure(df: pd.DataFrame, feature: str, bins: int = 50):
    """
    Строит matplotlib Figure с гистограммой по признаку
    """
    if feature not in df.columns:
        return None
    data = df[feature]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, edgecolor='black')
    ax.set_title(f"Гистограмма: {feature}")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Частота")
    return fig

def show_hist(df: pd.DataFrame, selected_feature: str, bins: int = 50) -> None:
    """
    Отображает гистограмму выбранного признака через Streamlit
    """
    fig = build_hist_figure(df, selected_feature, bins)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.warning(f"Признак '{selected_feature}' не найден в DataFrame.")