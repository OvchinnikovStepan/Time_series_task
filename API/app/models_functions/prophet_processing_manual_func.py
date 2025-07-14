from prophet import Prophet
import pandas as pd
from API.app.models_functions.make_prediction_dataframe_func import make_prediction_dataframe
import json


def prophet_processing_manual(params):
    """
    - params: словарь параметров модели:
        - seasonality_mode: 'additive' или 'multiplicative'
        - yearly_seasonality: bool/int
        - weekly_seasonality: bool/int
        - daily_seasonality: bool/int
        - seasonality_prior_scale: float
        - changepoint_prior_scale: float
    """
    df_train = pd.read_json(params["df_train"], orient='table')
    
    train_df = pd.DataFrame({
        'ds': df_train.index,
        'y': df_train["sensor"].values
    })
    
    hyper_params = json.loads(params["hyper_params"])

    model = Prophet(
        growth="linear",
        seasonality_mode=hyper_params.get("seasonality_mode", "additive"),
        yearly_seasonality=hyper_params.get("yearly_seasonality", False),
        weekly_seasonality=hyper_params.get("weekly_seasonality", False),
        daily_seasonality=hyper_params.get("daily_seasonality", False),
        seasonality_prior_scale=hyper_params.get("seasonality_prior_scale", 10.0),
        changepoint_prior_scale=hyper_params.get("changepoint_prior_scale", 0.05),
    )
    
    model.fit(train_df)
    
    future = model.make_future_dataframe(
        periods=params["horizon"],
        freq=pd.infer_freq(df_train.index))
    
    forecast = model.predict(future)
    predictions = forecast.tail(params["horizon"])['yhat']
    
    model_params = {
    "growth": model.growth,
    "changepoints": model.changepoints.tolist(),  # datetime → список строк
    "n_changepoints": model.n_changepoints,
    "seasonality_mode": model.seasonality_mode,
    "seasonalities": model.seasonalities,  # сезонности (годовая, недельная)
    "params": {  # внутренние параметры (тренд, сезонности, шумы)
        "k": model.params["k"][0].tolist(),  # коэффициент тренда
        "m": model.params["m"][0].tolist(),  # смещение тренда
        "sigma_obs": model.params["sigma_obs"][0].tolist(),  # шум данных
        "beta": model.params["beta"][0].tolist(),  # коэффициенты сезонности
    },
    "holidays": model.holidays.to_dict(orient="records") if model.holidays is not None else None,
    }

    # Конвертируем в JSON (с обработкой datetime)
    model_params = json.dumps(model_params, indent=4, default=str)
    return {
        "predictions": make_prediction_dataframe(df_train,predictions.values,params["horizon"]),
        "model_params":  model_params
    }