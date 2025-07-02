from pydantic import BaseModel
import json
import requests

#Отправил

def send_model_request(payload: BaseModel):
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        url = config['url']

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

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

