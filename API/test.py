import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from send_request import send_model_request
from app.main import ModelRequestModel


def create_simple_frame():
    # Начальная дата и количество записей
    start_date = datetime.now().replace(microsecond=0)
    num_records = 20  # например, 20 записей

    # Генерируем даты (минута за минутой)
    dates = [start_date + timedelta(minutes=i) for i in range(num_records)]

    # Генерируем случайные значения датчика
    sensor_values = np.random.uniform(0, 100, size=num_records).round(2)

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
df_train = create_simple_frame()
df_test = create_simple_frame()

json_df_train = df_train.to_json(orient='records')
json_df_test = df_test.to_json(orient='records')

params = {
    "df_train": json_df_train,
    "df_test": json_df_test,
    "duration": 0
}

# Формируем payload
payload = {
    'model_type': 'sarima',
    'auto_params': True,
    'params': json.dumps(params)
}

response = send_model_request(payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
