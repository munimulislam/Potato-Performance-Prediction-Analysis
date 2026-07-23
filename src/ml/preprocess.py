"""
@File - preprocess.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ThresholdResult:
    train_kept_mask: np.ndarray
    n_train: int
    n_train_kept: int


def make_onehot_dense() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def row_missing_fraction(df: pd.DataFrame, numeric_cols: list[str]) -> pd.Series:
    missing = df[numeric_cols].isna().sum(axis=1)
    return missing / float(len(numeric_cols))


def apply_train_threshold(
    train_df: pd.DataFrame, *, numeric_cols: list[str], threshold: float
) -> ThresholdResult:
    mf = row_missing_fraction(train_df, numeric_cols)
    mask = (mf <= threshold).to_numpy()
    return ThresholdResult(
        train_kept_mask=mask, n_train=len(train_df), n_train_kept=int(mask.sum())
    )


def usable_numeric_columns(
    train_df_kept: pd.DataFrame, numeric_cols: list[str]
) -> list[str]:
    return [c for c in numeric_cols if not train_df_kept[c].isna().all()]


def build_numeric_pipeline(
    scale_numeric: bool, use_missing_indicator: bool
) -> Pipeline:
    value_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        value_steps.append(("scaler", StandardScaler()))

    values_pipe = Pipeline(steps=value_steps)

    if not use_missing_indicator:
        return values_pipe

    flags = MissingIndicator(features="all", sparse=False)
    union = FeatureUnion(
        [
            ("values", values_pipe),
            ("flags", flags),
        ]
    )
    return Pipeline([("union", union)])


def build_preprocessor(
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
    use_missing_indicator: bool,
) -> ColumnTransformer:

    numeric_pipe = build_numeric_pipeline(scale_numeric, use_missing_indicator)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_dense()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
