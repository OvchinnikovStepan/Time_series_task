from .sarima_processing_manual_func import sarima_processing_manual
from .sarima_processing_auto_func import sarima_processing_auto
from .ets_processing_manual_func import ets_processing_manual
from .ets_processing_auto_func import ets_processing_auto
from .prophet_processing_manual_func import prophet_processing_manual
from .prophet_processing_auto_func import prophet_processing_auto
from app.schemas import ModelRequestModel
import json

routing_map={
    "sarima":{
        True: sarima_processing_auto,
        False: sarima_processing_manual
        },
    "ets":{
        True: ets_processing_auto,
        False: ets_processing_manual
        },
    "prophet":{
        True: prophet_processing_auto,
        False: prophet_processing_manual
        }
}

def routing_func(request: ModelRequestModel) -> dict:
    information = json.loads(request.information)

    result = routing_map[request.model_type][request.auto_params](information)
    return result  # Возвращаем результат