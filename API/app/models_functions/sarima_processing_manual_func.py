from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

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

    hyper_params = params["params"]
    if  hyper_params["S"]:
        model = SARIMAX(
            params["df_train"],
            order=(hyper_params["p"], hyper_params["d"], hyper_params["q"]),
            seasonal_order=(hyper_params["P"], hyper_params["D"], hyper_params["Q"], hyper_params["S"])
        ).fit(disp=-1)
    else:
        model = SARIMAX(
            params["df_train"],
            order=(hyper_params["p"], hyper_params["d"], hyper_params["q"]),
        ).fit(disp=-1)

    forecast_steps = len(params["df_test"])
    predictions = model.get_forecast(steps=forecast_steps).predicted_mean
    
    predictions_df = pd.DataFrame(predictions, index=params["df_test"].index, columns=["predictions"])

    model_params = {
        'order': model.order,
        'seasonal_order': model.seasonal_order,
        'params': model.params()
    }
    return {
        "predictions": predictions_df,
        "model_params": model_params,
    }