from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd
from itertools import product
from .make_prediction_dataframe_func import make_prediction_dataframe
import json
import logging

logging.getLogger('scipy').setLevel(logging.WARNING)


def ets_processing_auto(params):
    """
    - params: словарь с параметрами:
        - seasonal_periods: int/None (период сезонности)
    """
    df_train = pd.read_json(params["df_train"], orient='table')
    y = df_train["sensor"].values

    error_types = ['add', 'mul']
    trend_types = [None, 'add', 'mul']
    season_types = [None, 'add', 'mul']
    damped_options = [False, True]

    try:
        seasonal_periods = params["hyper_params"]["seasonal_periods"]
    except Exception as e:
        print(f"ОШИБКА ПРИ СЧИТЫВАНИИ ПЕРИОДА {e}")
        seasonal_periods = None

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
            model = ETSModel(y,
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
        except Exception as e:
            print(f"ОШИБКА В ТРАЙ {e}")


    forecast_steps = params["horizon"]
    predictions = best_model.forecast(forecast_steps)

    model_params = {
        'model_type': {
            'error': best_model.error,
            'trend': best_model.trend,
            'seasonal': best_model.seasonal,
            'damped': best_model.damped_trend
        },
        'params': {
            "best_params": json.dumps(best_model.params.tolist())
        },
        'seasonal_periods': best_model.seasonal_periods,
        'aic': best_model.aic,
        'bic': best_model.bic
    }

    return {
        "predictions": make_prediction_dataframe(df_train, predictions, forecast_steps),
        "model_params": model_params,
    }
