from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from itertools import product
from make_prediction_dataframe_func import make_prediction_dataframe

def prophet_processing_auto(params):

    df_train = pd.read_json(params["df_train"], orient='records')
    df_test = pd.read_json(params["df_test"], orient='records')

    train_df = pd.DataFrame({
        'ds': df_train.index,
        'y': df_train.values
    })
    
    param_grid = {
        'growth': ['linear', 'logistic'],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False]
    }
 
    validation_size = 0.2
    n_folds =  3
    
    best_score = np.inf
    best_params = {}
    best_model = None
    
    for fold in range(n_folds):

        split_idx = int(len(train_df) * (1 - validation_size))
        train = train_df.iloc[:split_idx]
        valid = train_df.iloc[split_idx:]
        
        train_df = train_df.iloc[split_idx//2:].reset_index(drop=True)
        
        for params_comb in product(*param_grid.values()):
            current_params = dict(zip(param_grid.keys(), params_comb))
            
            try:
                # Создание и обучение модели
                model = Prophet(**current_params)
                model.fit(train)
                
                # Прогноз на валидационном наборе
                future = model.make_future_dataframe(
                    periods=len(valid), 
                    freq=pd.infer_freq(train['ds']))
                forecast = model.predict(future)
                
                # Оценка качества
                val_predictions = forecast.tail(len(valid))['yhat'].values
                score = mean_squared_error(valid['y'].values, val_predictions)
                
                # Сохранение лучшей модели
                if score < best_score:
                    best_score = score
                    best_params = current_params
                    best_model = model
                    
            except Exception as e:
                continue
    
    # Обучение лучшей модели на всех данных
    final_model = Prophet(**best_params)
    final_model.fit(pd.DataFrame({
        'ds': df_train.index,
        'y': df_train.values
    }))
    
    # Прогноз на тестовом наборе
    future = final_model.make_future_dataframe(
        periods=len(df_test)+params["duration"], 
        freq=pd.infer_freq(df_train.index))
    forecast = final_model.predict(future)
    predictions = forecast.tail(len(df_test))['yhat']
    
    # Формирование возвращаемых данных
    model_params = {
        'best_params': best_params,
        'validation_score': best_score,
        'model_components': {
            'growth': best_params['growth'],
            'seasonality': {
                'mode': best_params['seasonality_mode'],
                'yearly': best_params.get('yearly_seasonality', True),
                'weekly': best_params.get('weekly_seasonality', True)
            },
            'prior_scales': {
                'changepoint': best_params['changepoint_prior_scale'],
                'seasonality': best_params['seasonality_prior_scale']
            }
        }
    }
    
    return {
        "predictions": make_prediction_dataframe(df_train,predictions.values,params["duration"]),
        "model_params": model_params
    }