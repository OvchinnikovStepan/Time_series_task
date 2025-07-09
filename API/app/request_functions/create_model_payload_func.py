import pandas as pd
import json


#Функция для создания payload для обращения к API моделей предсказания
def create_model_payload(auto_params: bool,
                         horizon: int,
                         df_train: pd.DataFrame,
                         hyper_params: dict = {}):
    json_df_train = df_train.to_json(orient='table', date_format='iso')

    return {
        'auto_params': auto_params,
        "horizon": horizon,
        "hyper_params": json.dumps(hyper_params),
        "df_train": json_df_train
    }

