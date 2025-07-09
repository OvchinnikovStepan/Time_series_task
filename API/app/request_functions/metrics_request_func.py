import httpx
from ..schemas import MetricsRequest


async def get_metrics(url, payload: MetricsRequest) -> httpx.Response:
        url = url + "metrics"

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                url=url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response



