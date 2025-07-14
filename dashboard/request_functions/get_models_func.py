import httpx


async def get_models(url: str) -> httpx.Response:
    url = url + "models"

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(
            url=url,
            headers={"Content-Type": "application/json"}
        )
        return response



