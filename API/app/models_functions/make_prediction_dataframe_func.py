import pandas as pd

def make_prediction_dataframe(original_df, new_data, num_new_periods):
    # Определяем частоту исходного индекса
     # типы колонок
    try:
        freq = pd.infer_freq(original_df.index)
        if freq is None:
            raise ValueError("Не удалось определить частоту временного индекса стандартным методов")
    except:
        freq = original_df.index[-1]-original_df.index[-2]

    # Получаем последнюю дату в исходном DataFrame
    last_date = original_df.index[-1]
    
    # Создаем новый временной индекс
    
    new_index = pd.date_range(
        start=last_date,
        periods=num_new_periods + 1,  # +1 потому что первая дата уже есть
        freq=freq
    )[1:]  # Исключаем первую дату (она уже есть)

    # Создаем новый DataFrame с расширенным индексом
    extended_df = pd.DataFrame(new_data, index=new_index, columns=["predictions"])
    
    return extended_df