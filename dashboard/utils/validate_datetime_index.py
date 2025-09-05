import pandas as pd


def validate_datetime_index(df: pd.DataFrame) -> None:
    """
    Проверяет, что индекс датафрейма — это даты.
    Если индекс — строки, пытается привести к DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, format='mixed')
        except Exception:
            raise ValueError("Индекс не удалось привести к DatetimeIndex. Проверьте формат дат в индексе.")