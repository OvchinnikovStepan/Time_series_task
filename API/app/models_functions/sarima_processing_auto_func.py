from pmdarima import auto_arima
import pandas as pd
import json
import re
from .make_prediction_dataframe_func import make_prediction_dataframe

def sarima_processing_auto(params):
    """
    - params:
        S - сезонность
    """

    df_train = pd.read_json(params["df_train"], orient='table')
    df_test = pd.read_json(params["df_test"], orient='table')

    try:
        season = params["params"]["S"]
    except:
        season = 1

    model = auto_arima(
        df_train,
        m=season,
        trace=True,
        stepwise=True,
        suppress_warnings=True
    )
    
    forecast_steps = len(df_test)+params["duration"]
    predictions = model.predict(n_periods=forecast_steps)
    
    model_params = {
        'order': model.order,
        'seasonal_order': model.seasonal_order,
        'params': model.params()
    }


    return {
        "predictions": make_prediction_dataframe(df_train,predictions,forecast_steps),

        "model_params": model_params,
    }