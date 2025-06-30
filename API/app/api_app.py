from fastapi import FastAPI
from pydantic import BaseModel
from typing import  Optional
import pandas as pd
import json
import httpx
from models_functions.routing_func import routing_func
from metrics_functions.metrics_func import calculate_metrics

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
        "model_params": df_predict["model_params"],
        "df_predict": df_predict["predictions"].to_json(orient='records'),
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

        metrics = calculate_metrics(df_test, df_predict)


    except Exception as e:
        return {"error": f"Failed to parse DataFrame: {str(e)}"}

    # Пример обработки
    response = {
        "metrics_status": "success",
        "metrics": metrics
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


