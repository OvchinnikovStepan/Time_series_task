from typing import Optional, Sequence, Literal, Tuple
import pandas as pd

def edit_dataset_by_quantiles(
    df: pd.DataFrame,
    lower_q: float = 0.25,
    upper_q: float = 0.75,
    cols: Optional[Sequence[str]] = None,
    mode: Literal["clip", "filter"] = "clip",
    how: Literal["all", "any"] = "all",
    inclusive: Literal["both", "left", "right", "neither"] = "both",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает (base_df, modified_df), где base_df — неизменённая копия исходного df,
    modified_df — результат преобразования по квантилям.

    Параметры:
      lower_q, upper_q: квантили в [0, 1], lower_q < upper_q.
      cols: список колонок для обработки. Если None — берутся все числовые.
      mode:
        - "clip"   — винзоризация: обрезать значения колонок по [q_low, q_high].
        - "filter" — фильтрация строк: оставить строки, где значения в пределах
                     [q_low, q_high] (границы задаёт параметр inclusive).
      how (только для mode="filter"):
        - "all" — строка остаётся, если ВСЕ указанные колонки попадают в диапазон.
        - "any" — строка остаётся, если ХОТЯ БЫ ОДНА колонка попадает в диапазон.
      inclusive (для mode="filter"):
        - "both"   — [q_low, q_high]
        - "left"   — [q_low, q_high)
        - "right"  — (q_low, q_high]
        - "neither"— (q_low, q_high)

    Примечания:
      - Функция не модифицирует входной df.
      - Квантили считаются по каждой колонке отдельно.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("lower_q должен быть < upper_q и оба в диапазоне [0, 1].")

    base = df.copy(deep=True)

    # Выбор колонок
    if cols is None:
        cols = list(base.select_dtypes(include="number").columns)
    else:
        missing = [c for c in cols if c not in base.columns]
        if missing:
            raise KeyError(f"Колонки не найдены: {missing}")
        # Оставим только числовые среди заданных
        cols = list(base[cols].select_dtypes(include="number").columns)

    if not cols:
        raise ValueError("Нет числовых колонок для обработки.")

    q_low = base[cols].quantile(lower_q)
    q_high = base[cols].quantile(upper_q)

    if mode == "clip":
        mod = base.copy(deep=True)
        for c in cols:
            mod[c] = mod[c].clip(lower=q_low[c], upper=q_high[c])
        return base, mod

    elif mode == "filter":
        # Строим булевы маски по колонкам с учётом inclusive
        ops_left = {"both": ">=", "left": ">=", "right": ">", "neither": ">"}
        ops_right = {"both": "<=", "left": "<", "right": "<=", "neither": "<"}

        masks = []
        for c in cols:
            left_ok = (mod := base[c]) >= q_low[c] if ops_left[inclusive] == ">=" else (base[c] > q_low[c])
            right_ok = (base[c] <= q_high[c]) if ops_right[inclusive] == "<=" else (base[c] < q_high[c])
            masks.append(left_ok & right_ok)

        if how == "all":
            mask = masks[0]
            for m in masks[1:]:
                mask &= m
        elif how == "any":
            mask = masks[0]
            for m in masks[1:]:
                mask |= m
        else:
            raise ValueError("how должен быть 'all' или 'any'.")

        mod = base.loc[mask].copy()
        return base, mod

    else:
        raise ValueError("mode должен быть 'clip' или 'filter'.")
