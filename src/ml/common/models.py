"""
@File - models.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    params: dict
    profile: str


def make_model(model_name: str, params: dict[str, Any]) -> ModelSpec:
    m = model_name.lower().strip()

    if m == "ridge":
        est = Ridge(**params)
        return ModelSpec(name="ridge", estimator=est, params=params, profile="linear")

    if m == "rf":
        p = dict(params)
        p.setdefault("n_jobs", -1)
        est = RandomForestRegressor(**p)
        return ModelSpec(
            name="rf",
            estimator=est,
            params=p,
            profile="tree",
        )

    if m == "knn":
        est = KNeighborsRegressor(**params)
        return ModelSpec(
            name="knn",
            estimator=est,
            params=params,
            profile="linear",
        )

    if m == "svr":
        est = SVR(**params)
        return ModelSpec(
            name="svr",
            estimator=est,
            params=params,
            profile="linear",
        )

    if m == "xgb":
        from xgboost import XGBRegressor

        est = XGBRegressor(**params)
        return ModelSpec(
            name="xgb",
            estimator=est,
            params=params,
            profile="booster",
        )

    if m == "lgbm":
        from lightgbm import LGBMRegressor

        est = LGBMRegressor(**params)
        return ModelSpec(
            name="lgbm",
            estimator=est,
            params=params,
            profile="booster",
        )

    raise ValueError(f"Unknown model_name: {model_name}.")
