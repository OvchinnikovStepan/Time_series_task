from API.app.models_functions.sarima_processing_manual_func import sarima_processing_manual
from API.app.models_functions.sarima_processing_auto_func import sarima_processing_auto
from API.app.models_functions.ets_processing_manual_func import ets_processing_manual
from API.app.models_functions.ets_processing_auto_func import ets_processing_auto
from API.app.models_functions.prophet_processing_manual_func import prophet_processing_manual
from API.app.models_functions.prophet_processing_auto_func import prophet_processing_auto
from API.app.schemas import ModelRequest

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

def routing_func(model_type, request: ModelRequest) -> dict:
    request_json = request.dict()
    result = routing_map[model_type][request.auto_params](request_json)
    return result  # Возвращаем результат