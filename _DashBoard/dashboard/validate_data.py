import pandas as pd

def validate_data(df):
    if df is None or df.empty:
        return False
    return True