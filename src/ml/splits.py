"""
@File - splits.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def group_kfold_splits(
    df: pd.DataFrame, *, group_col: str, n_splits: int
) -> Iterator[Split]:
    if group_col not in df.columns:
        raise ValueError(f"group_col not present: {group_col}")

    groups = df[group_col].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    X_dummy = np.zeros((len(df), 1), dtype=np.int8)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        yield Split(fold=fold, train_idx=train_idx, test_idx=test_idx)
