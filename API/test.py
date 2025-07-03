import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from app.request_functions.model_request_func import get_prediction
from app.request_functions.model_payload_func import create_model_payload


def create_simple_frame():
    # Начальная дата и количество записей
    start_date = datetime.now().replace(microsecond=0)
    num_records = 100  # например, 20 записей

    # Генерируем даты (минута за минутой)
    dates = [start_date + timedelta(minutes=i) for i in range(num_records)]

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
    df_test = create_simple_frame()

    params = {
        "seasonal_periods": 4
    }

    payload = create_model_payload('ets', True, 5, df_train, df_test, params)

    response = await get_prediction(payload)

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Запуск
asyncio.run(main())