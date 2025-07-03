import streamlit as st
import matplotlib.pyplot as plt

def show_hist(df, selected_feature, bins=50):
    if selected_feature in df.columns:
        data = df[selected_feature]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=bins, edgecolor='black')
        ax.set_title(f"Гистограмма: {selected_feature}")
        ax.set_xlabel("Значение")
        ax.set_ylabel("Частота")
        st.pyplot(fig)
    else:
        st.warning(f"Признак '{selected_feature}' не найден в DataFrame.")