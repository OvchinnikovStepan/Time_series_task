import pandas as pd
import numpy as np
from typing import List


def validate_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Валидирует и преобразует столбцы в числовые
    
    Args:
        df: DataFrame для валидации
        
    Returns:
        List[str]: Список валидных столбцов
    """
    valid_columns = []
    
    for col in df.columns:
        try:
            # Преобразуем в числовой тип
            numeric_data = pd.to_numeric(df[col], errors='raise')
            df[col] = numeric_data.astype(np.float64)
            
            # Проверяем, что столбец не состоит только из NaN
            if df[col].isna().all():
                print(f"Столбец {col} содержит только NaN и будет исключён.")
            else:
                valid_columns.append(col)
        except Exception as e:
            raise ValueError(f"Ошибка: столбец {col} содержит некорректные данные, не являющиеся показателями датчика. {str(e)}")
    
    return valid_columns
