from fastapi import FastAPI
from pydantic import BaseModel
from typing import  Optional
import pandas as pd
import json
import asyncio
import requests
import httpx
import isodate

app = FastAPI()

class ModelRequestModel(BaseModel):
    model_type: str
    auto_params: bool
    params: Optional[str] = None    # JSON строка
    df_train: str                   # JSON строка
    df_test: str                    # JSON строка
    duration: str

class MetricsRequestModel(BaseModel):
    df_predict: str             # JSON строка
    df_test: str                # JSON строка

@app.post("/api/v1/model_process")
async def process_data(request: ModelRequestModel):
    # Извлечение данных
    model_type = request.model_type
    auto_params = request.auto_params
    if auto_params:
        params = {'auto': 0.1}
    else:
        params = json.loads(request.params) if request.params else None
    duration = isodate.parse_duration(request.duration)

    # Преобразование JSON-строк в DataFrame
    try:
        df_train = pd.read_json(request.df_train, orient='records')
        df_test = pd.read_json(request.df_test, orient='records')

        # await asyncio.sleep(20)
        df_predict = combine_first_rows(df_train, df_test)

    except Exception as e:
        return {"error": f"Failed to parse DataFrame: {str(e)}"}

    try:
        metrics_response = await get_metrics(
            df_test.to_json(orient='records'),
            df_predict.to_json(orient='records')
        )

        if not metrics_response.is_success:
            return {"error": f"Metrics service returned error: {metrics_response.status_code}"}

    except Exception as e:
        return {"error": f"Failed to get metrics: {str(e)}"}

    # Пример обработки
    response = {
        "status": "success",
        "received_model_type": model_type,
        "auto_params": auto_params,
        "params": params,
        "df_predict": df_predict.to_dict(orient='records') if df_predict is not None else None,
        "metrics": metrics_response.json()
    }

    return response


def combine_first_rows(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Берёт первую строку из df_train и df_test и объединяет их как две отдельные строки.

    :param df_train: Первый DataFrame (например, обучающие данные)
    :param df_test: Второй DataFrame (например, тестовые данные)
    :return: Новый DataFrame с двумя строками — первые строки из df_train и df_test
    """
    if df_train.empty or df_test.empty:
        raise ValueError("Один из DataFrame пустой, не могу взять первую строку.")

    # Берём первую строку из каждого DataFrame
    row_train = df_train.iloc[[0]].copy()
    row_test = df_test.iloc[[0]].copy()

    # Сбрасываем индексы, чтобы не было проблем при объединении
    row_train.reset_index(drop=True, inplace=True)
    row_test.reset_index(drop=True, inplace=True)

    # Объединяем по строкам (axis=0)
    combined_df = pd.concat([row_train, row_test], axis=0, ignore_index=True)

    return combined_df

@app.post("/api/v1/metrics_process")
async def process_data(request: MetricsRequestModel):
    # Извлечение данных
    # Преобразование JSON-строк в DataFrame
    try:
        df_predict = pd.read_json(request.df_predict, orient='records')
        df_test = pd.read_json(request.df_test, orient='records')

        # await asyncio.sleep(20)

        means_test = calculate_sensor_means(df_test)
        means_predict = calculate_sensor_means(df_predict)

        means = {
            "means_test": means_test,
            "means_predict": means_predict
        }


    except Exception as e:
        return {"error": f"Failed to parse DataFrame: {str(e)}"}

    # Пример обработки
    response = {
        "metrics_status": "success",
        "means": means
    }

    return response

async def get_metrics(df_test_json: str, df_predict_json: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/metrics_process",
            json={
                "df_test": df_test_json,
                "df_predict": df_predict_json
            },
            headers={"Content-Type": "application/json"}
        )
        return response


def calculate_sensor_means(df: pd.DataFrame) -> dict:
    """
    Рассчитывает среднее значение для всех колонок с показаниями датчиков.

    :param df: Входной DataFrame с данными датчиков (например, df_train или df_test)
    :return: Словарь вида {'sensor_1': среднее, 'sensor_2': среднее, ...}
    """
    # Убираем колонку timestamp из расчёта, если она есть
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        raise ValueError("В DataFrame нет числовых колонок для расчёта среднего.")

    means = df[numeric_cols].mean().to_dict()
    return means