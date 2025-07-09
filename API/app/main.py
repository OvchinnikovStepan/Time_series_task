from fastapi import FastAPI, Path
import pandas as pd
import os
import json

from models_functions.routing_func import routing_func
from metrics_functions.metrics_func import calculate_metrics
from schemas import ModelRequest, MetricsRequest

app = FastAPI()


#Функция обработки запроса получения предсказания
@app.post("/api/models/{model_type}")
async def process_data(
    model_type: str = Path(..., description="Тип модели для прогнозирования"),
    request: ModelRequest = ...
    ):

    predict = routing_func(model_type, request)
    predict_params = predict["model_params"]
    df_predict = predict["predictions"]

    response = {
        "hyper_params": predict_params,
        "df_predict": df_predict.to_json(orient='table', date_format='iso'),
    }

    return response

#Функция обработки запроса получения метрик
@app.post("/api/metrics")
async def process_data(request: MetricsRequest):

    try:
        df_predict = pd.read_json(request.df_predict, orient='table')
        df_test = pd.read_json(request.df_test, orient='table')

        if len(df_test) > 0:
            metrics = calculate_metrics(df_test, df_predict)
        else:
            metrics = None

    except Exception as e:
        return {
            'error': e
        }

    response = {
        "metrics": metrics
    }

    return response

#Функция обработки запроса получения списка предсказаний
@app.get("/api/models")
async def process_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '.', 'config.json')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config['models']

    except Exception as e:
        return {
            'error': e
        }
    response = {
        "models": models
    }

    return response
