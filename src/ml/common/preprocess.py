"""
@File - preprocess.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 25/07/2026
"""

from typing import List

import numpy as np
from scipy import sparse
from dataclasses import dataclass
from typing import Literal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

ProfileName = Literal["linear", "tree", "booster"]


@dataclass(frozen=True)
class ProfileSpec:
    numeric_impute: Literal["none", "median"]
    numeric_scale: Literal["none", "standard"]
    cat_onehot_sparse: bool
    output_dense: bool
    missing_indicator: bool


def get_profile_spec(profile: str) -> ProfileSpec:
    p = profile.lower()
    if p == "booster":
        return ProfileSpec(
            numeric_impute="none",
            numeric_scale="none",
            cat_onehot_sparse=True,
            output_dense=False,
            missing_indicator=False,
        )
    if p == "tree":
        return ProfileSpec(
            numeric_impute="median",
            numeric_scale="none",
            cat_onehot_sparse=False,
            output_dense=True,
            missing_indicator=True,
        )
    if p == "linear":
        return ProfileSpec(
            numeric_impute="median",
            numeric_scale="standard",
            cat_onehot_sparse=False,
            output_dense=True,
            missing_indicator=True,
        )
    raise ValueError(f"Unknown profile: {profile}")


def _make_onehot(handle_unknown: str, sparse_out: bool) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_out)


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


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
                FunctionTransformer(lambda X: X, feature_names_out="one-to-one"),
            )
        )
    elif spec.numeric_impute == "median":
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    else:
        raise ValueError(spec.numeric_impute)

    if spec.numeric_scale == "standard":
        num_steps.append(("scale", StandardScaler()))
    elif spec.numeric_scale == "none":
        pass
    else:
        raise ValueError(spec.numeric_scale)

    transformers.append(("num_values", Pipeline(steps=num_steps), numeric_cols))

    if spec.missing_indicator:
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
                    _make_onehot(
                        handle_unknown="ignore", sparse_out=spec.cat_onehot_sparse
                    ),
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
    if spec.output_dense:
        steps.append(("densify", FunctionTransformer(_to_dense)))

    return Pipeline(steps=steps)
