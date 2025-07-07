import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from dashboard.validate_data import validate_data

def show_pairplot(df: pd.DataFrame):
    if validate_data(df):
        if df.shape[1] > 10:
            st.warning("Слишком много признаков. График может быть трудночитаем. Рекомендуется выбрать не более 5 признаков.")
        
        selected_columns = st.multiselect(
            "Выберите признаки для построения pairplot:",
            options=df.columns.tolist(),
            default=df.columns[:5].tolist()
        )

        if len(selected_columns) >= 2:
            with st.spinner():
                fig = sns.pairplot(
                    df[selected_columns],
                    height=2.5,
                    aspect=1
                )
                fig.fig.set_size_inches(8, 6)
                st.pyplot(fig)
        else:
            st.info("Выберите как минимум два признака для отображения pairplot.")

