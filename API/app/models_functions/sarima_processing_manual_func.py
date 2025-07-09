from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import json
from API.app.models_functions.make_prediction_dataframe_func import make_prediction_dataframe

def sarima_processing_manual(params):
    """
    params:
        - S - сезонность
        - p - порядок авторегрессии (число используемых предыдущих значений ряда)
        - d - порядок дифферненцирования ряда
        - q - порядок скользящего среднего (число используемых предыдущих ошибок)
        - P - порядок сезонной авторегрессии
        - D - порядок сезонного дифференциорования
        - Q - порядок сезонного скользящего среднего
    """
    df_train = pd.read_json(params["df_train"], orient='table')

    hyper_params = json.loads(params["hyper_params"])
    if  hyper_params.get("S", False):
        model = SARIMAX(
            df_train,
            order=(hyper_params["p"], hyper_params["d"], hyper_params["q"]),
            seasonal_order=(hyper_params["P"], hyper_params["D"], hyper_params["Q"], hyper_params["S"])
        ).fit(disp=-1)
    else:
        model = SARIMAX(
            df_train,
            order=(hyper_params["p"], hyper_params["d"], hyper_params["q"]),
        ).fit(disp=-1)

    forecast_steps = params["horizon"]
    predictions = pd.DataFrame(model.get_forecast(steps=forecast_steps).predicted_mean).rename(columns={'predicted_mean':"predictions"})
    print("ПРЕДСКАЗАНИЯ",predictions["predictions"])
    model_params = {
        'hyper_params':hyper_params,
        'params': model.params
    }
    
    return {
        "predictions":  predictions["predictions"],
        "model_params": model_params,
    }