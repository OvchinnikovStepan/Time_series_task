import httpx
from ..schemas import ModelRequest

#Функция получения предсказания от API
async def get_prediction(url, payload: ModelRequest, model_type: str) -> httpx.Response:
    url = url + f"models/{model_type}"

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            url=url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response
