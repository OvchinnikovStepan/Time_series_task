import json
import httpx
from API.app.schemas import MetricsRequest
import os


root_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', '..', 'config.json')

async def get_metrics(payload: MetricsRequest) -> httpx.Response:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        url = config['url_metrics']

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                url=url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response



