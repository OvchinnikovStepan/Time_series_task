import pandas as pd


def create_metrics_payload(df_predict: pd.DataFrame,
                           df_test: pd.DataFrame):
    json_df_predict = df_predict.to_json(orient='table', date_format='iso')

    if isinstance(df_test, pd.DataFrame):
        json_df_test = df_test.to_json(orient='table', date_format='iso')
    else:
        empty_df = pd.DataFrame(columns=df_predict.columns).astype(df_predict.dtypes.to_dict())
        json_df_test = empty_df.to_json(orient='table', date_format='iso')

    return {
        "df_test": json_df_test,
        "df_predict": json_df_predict
    }
