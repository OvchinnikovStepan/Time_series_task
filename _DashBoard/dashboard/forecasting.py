import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose


def forecasting(df: pd.DataFrame, column: str, model: str = 'additive', period: int = 12):
    series = df[column].dropna()
    if len(series) < period * 2:
        st.warning(f"Недостаточно данных для декомпозиции с периодом {period}. Нужно минимум {period * 2} точек.")
        return
    
    try:
        result = seasonal_decompose(series, model=model, period=period)
    except Exception as e:
        st.error(f"Ошибка при декомпозиции: {e}")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    result.observed.plot(ax=axes[0], title='Наблюдаемые значения (Observed)')
    result.trend.plot(ax=axes[1], title='Тренд (Trend)')
    result.seasonal.plot(ax=axes[2], title='Сезонность (Seasonal)')
    result.resid.plot(ax=axes[3], title='Остатки (Residual)')

    plt.tight_layout()
    st.pyplot(fig)