import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import Optional

def build_acf_figure(df: pd.DataFrame, feature: str, nlags: int = 10):
    """
    Строит matplotlib Figure с автокорреляционной функцией по признаку
    """
    if feature not in df.columns:
        return None
    data = df[feature]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    sm.graphics.tsa.plot_acf(data, lags=nlags, ax=ax, alpha=0.05)
    ax.set_title(f"ACF для '{feature}' (nlags={nlags})")
    ax.set_xlabel("Лаги")
    ax.set_ylabel("Корреляция")
    ax.plot([], [], label=feature, color='blue')
    ax.legend()
    return fig

def show_autocorrelation(df: pd.DataFrame, selected_feature: str, nlags: int = 10) -> None:
    """
    Отображает автокорреляционную функцию выбранного признака через Streamlit
    """
    fig = build_acf_figure(df, selected_feature, nlags)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.warning(f"Признак '{selected_feature}' не найден в DataFrame.")