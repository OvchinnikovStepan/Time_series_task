from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd
import json
from .make_prediction_dataframe_func import make_prediction_dataframe
def ets_processing_manual(params):
    """
    - params: словарь параметров модели:
        - error_type: 'add'/'mul'
        - trend_type: 'add'/'mul'/None
        - season_type: 'add'/'mul'/None
        - seasonal_periods: int/None
        - damped_trend: bool
    """
    df_train = pd.read_json(params["df_train"], orient='table')
    y = df_train["sensor"].values

    df_test = pd.read_json(params["df_test"], orient='table')

    hyper_params = json.loads(params["params"])
    model = ETSModel(
        y,
        error=hyper_params.get("error_type", "add"),
        trend=hyper_params.get("trend_type", None),
        seasonal=hyper_params.get("season_type", None),
        seasonal_periods=hyper_params.get("seasonal_periods", None),
        damped_trend=hyper_params.get("damped_trend", False)
    ).fit()
    
    forecast_steps = len(df_test)+params["duration"]
    predictions = model.forecast(steps=forecast_steps)
    
    model_params = {
        'model_type': {
            'error': model.error,
            'trend': model.trend,
            'seasonal': model.seasonal,
            'damped': model.damped_trend
        },
        'params': {
            "best_params": json.dumps(model.params.tolist())
        },
        'seasonal_periods': model.seasonal_periods,
        'aic': model.aic,
        'bic': model.bic
    }
    
    return {
        "predictions": make_prediction_dataframe(df_train,predictions,forecast_steps),
        "model_params": model_params
    }