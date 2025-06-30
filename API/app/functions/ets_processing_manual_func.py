from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd

def ets_processing_manual(params):

    hyper_params = params["params"]
    
    model = ETSModel(
        params["df_train"],
        error=hyper_params.get("error_type", "add"),
        trend=hyper_params.get("trend_type", None),
        seasonal=hyper_params.get("season_type", None),
        seasonal_periods=hyper_params.get("seasonal_periods", None),
        damped_trend=hyper_params.get("damped_trend", False)
    ).fit()
    
    forecast_steps = len(params["df_test"])
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
        "predictions": pd.DataFrame(predictions, index=params["df_test"].index, columns=["predictions"]),
        "model_params": model_params
    }