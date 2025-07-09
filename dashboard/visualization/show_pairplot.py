import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Optional
from dashboard.utils.validate_data import validate_data

def select_pairplot_features(df: pd.DataFrame) -> List[str]:
    """
    UI-компонент для выбора признаков для pairplot
    """
    if df.shape[1] > 10:
        st.warning("Слишком много признаков. График может быть трудночитаем. Рекомендуется выбрать не более 5 признаков.")
    return st.multiselect(
        "Выберите признаки для построения pairplot:",
        options=df.columns.tolist(),
        default=df.columns[:5].tolist()
    )

def build_pairplot(df: pd.DataFrame, columns: List[str]) -> Optional[sns.axisgrid.PairGrid]:
    """
    Строит pairplot по выбранным признакам
    """
    if len(columns) < 2:
        st.info("Выберите как минимум два признака для отображения pairplot.")
        return None
    with st.spinner():
        fig = sns.pairplot(
            df[list(columns)],
            height=2.5,
            aspect=1
        )
        fig.fig.set_size_inches(8, 6)
        return fig

def show_pairplot(df: pd.DataFrame) -> None:
    """
    Основная функция для отображения pairplot
    """
    if not validate_data(df):
        return
    selected_columns = select_pairplot_features(df)
    pairplot_fig = build_pairplot(df, selected_columns)
    if pairplot_fig is not None:
        st.pyplot(pairplot_fig.fig)

