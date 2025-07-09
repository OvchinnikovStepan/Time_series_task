import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import os
import json
from io import StringIO
from app.request_functions.get_models_func import get_models
from app.request_functions.model_request_func import get_prediction
from app.request_functions.create_model_payload_func import create_model_payload
from app.request_functions.metrics_request_func import get_metrics
from app.request_functions.create_metrics_payload_func import create_metrics_payload


def create_simple_frame(num_records = 100):
    # Начальная дата и количество записей
    start_date = datetime.now().replace(microsecond=0)

    # Генерируем даты (минута за минутой)
    dates = [start_date + timedelta(hours=i) for i in range(num_records)]

    # sensor_values = np.random.uniform(0, 100, size=num_records).round(2)

    # Генерируем случайные значения датчика
    t = np.linspace(0, 25 * np.pi, num_records)  # временная ось от 0 до 2π
    sensor_values = np.sin(t) * 50 + 50  # синусоида с амплитудой 50 и смещением 50
    sensor_values = np.round(sensor_values, 2)

    # Создаём DataFrame
    df_test = pd.DataFrame({
        'sensor': sensor_values
    }, index=dates)

    # Переименовываем индекс для наглядности
    df_test.index.name = 'timestamp'

    # Выводим первые строки
    print(df_test.head())

    return df_test


# ==Подготовка данных==


async def main():
    df_train = create_simple_frame()
    df_test = create_simple_frame(10)

    params = {
        "S":4,
        "p":1,
        "d":1,
        "q":0,
        "P":1,
        "D":1,
        "Q":1

        # "seasonality_mode":  'multiplicative',
        # "yearly_seasonality": False,
        # "weekly_seasonality": False,
        # "daily_seasonality": True,
        # "seasonality_prior_scale": 1,
        # "changepoint_prior_scale": 0.5


# Параметры для ETS manual
        # 'error_type': 'add',
        # 'trend_type': 'add',
        # 'season_type': 'mul',
        # 'damped_trend': False,
        # "seasonal_periods": 4
    }

    root_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir,'app', 'config.json')


    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        url = config['url']

        payload = create_model_payload(True, 5, df_train, params)

        response = await get_prediction(url, payload, "ets")

        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        print("______________________METRICS________________________")

        df_predict = pd.read_json(StringIO(response.json()["df_predict"]), orient='table')

        metrics_payload = create_metrics_payload(df_predict, df_test)

        response = await get_metrics(url, metrics_payload)

        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        print("______________________MODELS_LIST________________________")

        response = await get_models(url)

        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())




# Запуск
asyncio.run(main())