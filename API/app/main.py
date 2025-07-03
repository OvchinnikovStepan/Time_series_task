from fastapi import FastAPI
import pandas as pd
import json
from .models_functions.routing_func import routing_func
from .metrics_functions.metrics_func import calculate_metrics
from API.app.schemas import ModelRequestModel, MetricsRequestModel
from API.app.request_functions.create_metrics_payload_func import create_metrics_payload
from API.app.request_functions.metrics_request_func import get_metrics

app = FastAPI()

@app.post("/api/v1/model_process")
async def process_data(request: ModelRequestModel):
    # Извлечение данных
    model_type = request.model_type
    auto_params = request.auto_params
    information = json.loads(request.information)

    predict = routing_func(request)
    predict_params = predict["model_params"]
    df_predict = predict["predictions"]
    df_test = pd.read_json(information["df_test"], orient='table')

    if len(df_test) > 0:
        try:
            payload = create_metrics_payload(df_test, df_predict)

            metrics_response = await get_metrics(payload)

            if not metrics_response.is_success:
                metrics_response = {
                    'error': f"Metrics service returned error: {metrics_response.status_code}"
                }

        except Exception as e:
            metrics_response = {
                'error': e
            }

        metrics_response_json = metrics_response.json()
    else:
        metrics_response_json = None




    # Пример обработки
    response = {
        "status": "success",
        "received_model_type": model_type,
        "auto_params": auto_params,
        "model_params": predict_params,
        "df_predict": df_predict.to_json(orient='table', date_format='iso'),
        "metrics": metrics_response_json
    }

    return response


@app.post("/api/v1/metrics_process")
async def process_data(request: MetricsRequestModel):

    try:
        df_predict = pd.read_json(request.df_predict, orient='table')
        df_test = pd.read_json(request.df_test, orient='table')

        metrics = calculate_metrics(df_test, df_predict)

    except Exception as e:
        return {"error": f"Failed to parse DataFrame: {str(e)}"}

    # Пример обработки
    response = {
        "metrics_status": "success",
        "metrics": metrics
    }

    return response

