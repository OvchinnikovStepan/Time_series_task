from pydantic import BaseModel
from typing import Optional


class ModelRequestModel(BaseModel):
    model_type: str
    auto_params: bool
    information: str  # JSON строка


class MetricsRequestModel(BaseModel):
    df_predict: str  # JSON строка
    df_test: str  # JSON строка