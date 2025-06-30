from prophet import Prophet
import pandas as pd

def prophet_processing_manual(params):
    """
    - params: словарь параметров модели:
        - growth: 'linear' или 'logistic'
        - seasonality_mode: 'additive' или 'multiplicative'
        - yearly_seasonality: bool/int
        - weekly_seasonality: bool/int
        - daily_seasonality: bool/int
        - seasonality_prior_scale: float
        - changepoint_prior_scale: float
    """
    df_train = pd.read_json(params["df_train"], orient='records')
    df_test = pd.read_json(params["df_test"], orient='records')
    
    train_df = pd.DataFrame({
        'ds': df_train.index,
        'y': df_train.values
    })
    
    model = Prophet(
        growth=params["params"].get("growth", "linear"),
        seasonality_mode=params["params"].get("seasonality_mode", "additive"),
        yearly_seasonality=params["params"].get("yearly_seasonality", False),
        weekly_seasonality=params["params"].get("weekly_seasonality", False),
        daily_seasonality=params["params"].get("daily_seasonality", False),
        seasonality_prior_scale=params["params"].get("seasonality_prior_scale", 10.0),
        changepoint_prior_scale=params["params"].get("changepoint_prior_scale", 0.05),
    )
    
    model.fit(train_df)
    
    future = model.make_future_dataframe(
        periods=len(df_test),
        freq=pd.infer_freq(df_train.index))
    
    forecast = model.predict(future)
    
    predictions = forecast.tail(len(df_test))['yhat']
    
    model_params = {
        'growth': model.growth,
        'seasonality': {
            'mode': model.seasonality_mode,
            'yearly': model.yearly_seasonality,
            'weekly': model.weekly_seasonality,
            'daily': model.daily_seasonality,
        },
        'prior_scales': {
            'seasonality': model.seasonality_prior_scale,
            'changepoint': model.changepoint_prior_scale,
        },
        'changepoints': model.changepoints.tolist(),
        'trend_params': model.params['k'][0],
        'seasonality_params': model.params['delta'].mean(axis=0).tolist()
    }
    
    return {
        "predictions": pd.DataFrame(predictions.values, 
                                 index=df_test.index, 
                                 columns=["predictions"]),
        "model_params": model_params
    }