import json
import httpx
from ..schemas import ModelRequest
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', '..', 'config.json')


#Функция получения предсказания от API
async def get_prediction(payload: ModelRequest, model_type: str) -> httpx.Response:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        url = config['url_models'] + f"/{model_type}"

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                url=url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response
