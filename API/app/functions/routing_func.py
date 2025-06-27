from sarima_processing_manual_func import sarima_processing_manual
from sarima_processing_auto_func import sarima_processing_auto

routing_map={
    "sarima":{
        True: sarima_processing_auto,
        False: sarima_processing_manual
        }
}

def routing_func(params: dict):
    routing_map[params.model_type][params.auto_params](params)