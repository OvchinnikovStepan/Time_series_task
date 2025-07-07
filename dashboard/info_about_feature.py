import pandas as pd
import streamlit as st


def info_about_feature(df, feature):
    if feature not in df.columns:
        st.error(f"Признак '{feature}' не найден в данных.")
        return
    
    mean, median, std, minimal, maximum = [None for _ in range(5)]
    series = df[feature]
    if df is not None and feature is not None:
        mean = series.mean()
        median = series.median()
        std = series.std()
        minimal = series.min()
        maximum = series.max()

    return mean, median, std, minimal, maximum