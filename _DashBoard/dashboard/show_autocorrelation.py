import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st


def show_autocorrelation(df, selected_feature, nlags=10):
    data = df[selected_feature]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    sm.graphics.tsa.plot_acf(data, lags=nlags, ax=ax, alpha=0.05)
    ax.set_title(f"ACF для '{selected_feature}' (nlags={nlags})")
    ax.set_xlabel("Лаги")
    ax.set_ylabel("Корреляция")

    # Добавим label вручную (в легенду)
    ax.plot([], [], label=selected_feature, color='blue')
    ax.legend()

    st.pyplot(fig)