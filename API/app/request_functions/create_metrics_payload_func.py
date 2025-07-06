import pandas as pd


def create_metrics_payload(df_train: pd.DataFrame,
                           df_test: pd.DataFrame):
    json_df_train = df_train.to_json(orient='table', date_format='iso')
    json_df_test = df_test.to_json(orient='table', date_format='iso')

    return {
        "df_test": json_df_train,
        "df_predict": json_df_test
    }
