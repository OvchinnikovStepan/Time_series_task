import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from dashboard.validate_data import validate_data

def show_heatmap(df: pd.DataFrame):
    if validate_data(df):
        if df.shape[1] > 10:
            st.warning("Слишком много признаков. Рекомендуется выбрать не более 10 признаков для читаемой тепловой карты.")

        selected_columns = st.multiselect(
            "Выберите признаки для отображения тепловой карты:",
            options=df.columns.tolist(),
            default=df.columns[:5].tolist()
        )

        if len(selected_columns) >= 2:
            corr_matrix = df[selected_columns].corr()

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
        else:
            st.info("Выберите как минимум два признака для отображения тепловой карты.")

