from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd
from itertools import product
from .make_prediction_dataframe_func import make_prediction_dataframe

def ets_processing_auto(params):
    """
    - params: словарь с параметрами:
        - seasonal_periods: int/None (период сезонности)
    """
    df_train = pd.read_json(params["df_train"], orient='records')
    df_test = pd.read_json(params["df_test"], orient='records')

    error_types = ['add', 'mul']
    trend_types = [None, 'add', 'mul']
    season_types = [None, 'add', 'mul']
    damped_options = [False, True]
    seasonal_periods = params["params"].get("seasonal_periods", None)
    
    best_aic = float('inf')
    best_model = None
    
    for err, trend, season, damped in product(error_types, 
                                             trend_types, 
                                             season_types, 
                                             damped_options):
        if season is not None and seasonal_periods is None:
            continue
        if trend is None and damped:
            continue
            
        try:
            model = ETSModel(df_train,
                           error=err,
                           trend=trend,
                           seasonal=season,
                           seasonal_periods=seasonal_periods,
                           damped_trend=damped).fit()
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_params = {
                    'error_type': err,
                    'trend_type': trend,
                    'season_type': season,
                    'damped_trend': damped
                }
        except:
            continue
    
    forecast_steps = len(df_test)+params["duration"]
    predictions = best_model.forecast(steps=forecast_steps)
    
    model_params = {
        'model_type': {
            'error': best_model.error,
            'trend': best_model.trend,
            'seasonal': best_model.seasonal,
            'damped': best_model.damped
        },
        'params': {
            'smoothing_level': best_model.params['smoothing_level'],
            'smoothing_trend': best_model.params.get('smoothing_trend', None),
            'smoothing_seasonal': best_model.params.get('smoothing_seasonal', None),
            'initial_level': best_model.params['initial_level'],
            'initial_trend': best_model.params.get('initial_trend', None),
            'initial_seasons': best_model.params.get('initial_seasons', None)
        },
        'seasonal_periods': best_model.seasonal_periods,
        'aic': best_model.aic,
        'bic': best_model.bic
    }
    
    return {
        "predictions": make_prediction_dataframe(df_train,predictions,params["duration"]),
        "model_params": model_params,
    }