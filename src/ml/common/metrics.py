"""
@File - metrics.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    pearson_r: float
    r2: float


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson_r = _pearson_r(y_true, y_pred)

    return Metrics(
        rmse=rmse,
        mae=mae,
        pearson_r=pearson_r,
        r2=r2,
    )


def mean_std_se(values: pd.Series) -> Tuple[float, float, float, int]:
    v = values.dropna()
    n = int(v.shape[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    se = float(std / math.sqrt(n)) if n > 0 else float("nan")
    return mean, std, se, n
