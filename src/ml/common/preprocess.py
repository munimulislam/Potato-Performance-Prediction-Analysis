"""
@File - preprocess.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 25/07/2026
"""

import numpy as np

from typing import List
from scipy import sparse
from dataclasses import dataclass
from typing import Literal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

ProfileName = Literal["linear", "tree", "booster"]

RANDOM_STATE = 42


@dataclass(frozen=True)
class ProfileSpec:
    numeric_impute: Literal["none", "median"]
    numeric_scale: Literal["none", "standard"]


def get_profile_spec(profile: str) -> ProfileSpec:
    p = profile.lower()
    if p == "booster":
        return ProfileSpec(
            numeric_impute="none",
            numeric_scale="standard",
        )
    if p == "tree":
        return ProfileSpec(
            numeric_impute="median",
            numeric_scale="standard",
        )
    if p == "linear":
        return ProfileSpec(
            numeric_impute="median",
            numeric_scale="standard",
        )
    raise ValueError(f"Unknown profile: {profile}")


def _make_onehot(handle_unknown: str, sparse_out: bool) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_out)


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


def _identity(x):
    return x


def _to_float32(X):
    return X.astype(np.float32)


def make_preprocessor(
    *,
    profile: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    spec = get_profile_spec(profile)

    transformers = []

    num_steps = []
    if spec.numeric_impute == "none":
        num_steps.append(
            (
                "identity",
                FunctionTransformer(_identity, feature_names_out="one-to-one"),
            )
        )
    elif spec.numeric_impute == "median":
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    elif spec.numeric_impute == "mean":
        num_steps.append(("impute", SimpleImputer(strategy="mean")))
    elif spec.numeric_impute == "knn":
        num_steps.append(("impute", KNNImputer(n_neighbors=15)))
    elif spec.numeric_impute == "iterative":
        num_steps.append(
            (
                "impute",
                IterativeImputer(max_iter=10, random_state=RANDOM_STATE),
            )
        )
    else:
        raise ValueError(spec.numeric_impute)

    if spec.numeric_scale == "standard":
        num_steps.append(("scale", StandardScaler()))
    elif spec.numeric_scale == "none":
        pass
    else:
        raise ValueError(spec.numeric_scale)

    transformers.append(("num_values", Pipeline(steps=num_steps), numeric_cols))

    if spec.numeric_impute != "none":
        transformers.append(
            (
                "num_miss",
                Pipeline(
                    steps=[
                        ("miss", MissingIndicator(features="all")),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                (
                    "impute",
                    SimpleImputer(strategy="constant", fill_value="__CONSTANT__"),
                ),
                (
                    "onehot",
                    _make_onehot(handle_unknown="ignore", sparse_out=True),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    coltf = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
        sparse_threshold=0.3,
    )

    steps = [("columns", coltf)]

    return Pipeline(steps=steps)
