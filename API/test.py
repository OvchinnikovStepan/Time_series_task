import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from app.request_functions.model_request_func import get_prediction
from app.request_functions.create_model_payload_func import create_model_payload


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
    df = create_simple_frame()
    df_train=df[:90]
    df_test=df[90:]

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

    payload = create_model_payload('sarima', True, 5, df_train, df_test, params)

    response = await get_prediction(payload)

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Запуск
asyncio.run(main())