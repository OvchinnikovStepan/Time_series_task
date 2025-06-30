from pmdarima import auto_arima
import pandas as pd

def sarima_processing_auto(params):
    """
    - params:
        S - сезонность
    """
    df_train = pd.read_json(params["df_train"], orient='records')
    df_test = pd.read_json(params["df_test"], orient='records')

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
    
    forecast_steps = len(df_test)
    predictions = model.predict(n_periods=forecast_steps)
    
    predictions_df = pd.DataFrame(predictions, index=df_test.index, columns=["predictions"])
    
    model_params = {
        'order': model.order,
        'seasonal_order': model.seasonal_order,
        'params': model.params()
    }
    
    return {
        "predictions": predictions_df,
        "model_params": model_params,
    }