from fastapi import FastAPI
from pydantic import BaseModel
from typing import  Optional
import pandas as pd
import json
import httpx
from models_functions.routing_func import routing_func

app = FastAPI()

class ModelRequestModel(BaseModel):
    model_type: str
    auto_params: bool
    params: Optional[str] = None    # JSON строка



class MetricsRequestModel(BaseModel):
    df_predict: str             # JSON строка
    df_test: str                # JSON строка

@app.post("/api/v1/model_process")
async def process_data(request: ModelRequestModel):
    # Извлечение данных
    model_type = request.model_type
    auto_params = request.auto_params
    params = json.loads(request.params)


    # Преобразование JSON-строк в DataFrame
    try:
        df_train = pd.read_json(params["df_train"], orient='records')
        df_test = pd.read_json(params["df_test"], orient='records')

        # await asyncio.sleep(20)
        df_predict = routing_func(params)

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