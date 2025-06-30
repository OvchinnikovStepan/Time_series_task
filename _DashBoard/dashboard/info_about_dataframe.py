import streamlit as st
import pandas as pd


def info_about_dataframe(df: pd.DataFrame):
    features_size, tuples_size, first_tuple, last_tuple = [None for _ in range(4)]
    if df is not None:
        features_size = len(df.columns)
        tuples_size = len(df)
        first_tuple = df.index.min()
        last_tuple = df.index.max()

    return features_size, tuples_size, first_tuple, last_tuple