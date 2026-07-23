"""
@File - metrics.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    r2: float
    bias_mean: float
    acc_within_0_5: float
    acc_within_1_0: float
    spearman_rho: float


def _acc_within(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    return float(np.mean(np.abs(y_pred - y_true) <= tol))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return Metrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        bias_mean=bias,
        acc_within_0_5=_acc_within(y_true, y_pred, 0.5),
        acc_within_1_0=_acc_within(y_true, y_pred, 1.0),
        spearman_rho=_spearman(y_true, y_pred),
    )
