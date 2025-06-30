from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd
from make_prediction_dataframe_func import make_prediction_dataframe
def ets_processing_manual(params):
    """
    - params: словарь параметров модели:
        - error_type: 'add'/'mul'
        - trend_type: 'add'/'mul'/None
        - season_type: 'add'/'mul'/None
        - seasonal_periods: int/None
        - damped_trend: bool
    """
    df_train = pd.read_json(params["df_train"], orient='records')
    df_test = pd.read_json(params["df_test"], orient='records')

    hyper_params = params["params"]
    
    model = ETSModel(
        df_train,
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
            'damped': model.damped
        },
        'params': {
            'smoothing_level': model.params['smoothing_level'],
            'smoothing_trend': model.params.get('smoothing_trend', None),
            'smoothing_seasonal': model.params.get('smoothing_seasonal', None),
            'initial_level': model.params['initial_level'],
            'initial_trend': model.params.get('initial_trend', None),
            'initial_seasons': model.params.get('initial_seasons', None)
        },
        'seasonal_periods': model.seasonal_periods
    }
    
    return {
        "predictions": make_prediction_dataframe(df_train,predictions,params["duration"]),
        "model_params": model_params
    }