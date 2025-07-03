import pandas as pd
import json


def create_model_payload(model_type: str,
                         auto_params: bool,
                         duration: int,
                         df_train: pd.DataFrame,
                         df_test: pd.DataFrame,
                         params: dict = {}):
    json_df_train = df_train.to_json(orient='table', date_format='iso')
    json_df_test = df_test.to_json(orient='table', date_format='iso')
    information = {
        "df_train": json_df_train,
        "df_test": json_df_test,
        "duration": duration,
        "params": json.dumps(params)
    }

    return {
        'model_type': model_type,
        'auto_params': auto_params,
        'information': json.dumps(information)
    }

