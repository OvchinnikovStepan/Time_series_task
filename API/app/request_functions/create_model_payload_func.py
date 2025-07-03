import pandas as pd
import json
from typing import Optional


def create_model_payload(model_type: str,
                         auto_params: bool,
                         duration: int,
                         df_train: pd.DataFrame,
                         df_test: Optional[pd.DataFrame] = None,
                         params: dict = {}):
    json_df_train = df_train.to_json(orient='table', date_format='iso')
    if isinstance(df_test, pd.DataFrame):
        json_df_test = df_test.to_json(orient='table', date_format='iso')
    else:
        empty_df = pd.DataFrame(columns=df_train.columns).astype(df_train.dtypes.to_dict())
        json_df_test = empty_df.to_json(orient='table', date_format='iso')

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

