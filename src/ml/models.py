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
    scale_numeric: bool


def make_model(model_name: str, *, random_state: int) -> ModelSpec:
    m = model_name.lower().strip()

    if m == "ridge":
        est = Ridge(alpha=1.0)
        return ModelSpec(
            name="ridge", estimator=est, params={"alpha": 1.0}, scale_numeric=True
        )

    if m == "rf":
        est = RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        return ModelSpec(
            name="rf",
            estimator=est,
            params={"n_estimators": 500, "min_samples_leaf": 2},
            scale_numeric=False,
        )

    if m == "knn":
        est = KNeighborsRegressor(n_neighbors=15, weights="distance")
        return ModelSpec(
            name="knn",
            estimator=est,
            params={"n_neighbors": 15, "weights": "distance"},
            scale_numeric=False,
        )

    if m == "svr":
        est = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
        return ModelSpec(
            name="svr",
            estimator=est,
            params={"C": 10.0, "epsilon": 0.1, "kernel": "rbf"},
            scale_numeric=True,
        )

    if m == "xgb":
        from xgboost import XGBRegressor

        est = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        return ModelSpec(
            name="xgb",
            estimator=est,
            params={
                "n_estimators": 800,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            scale_numeric=False,
        )

    if m == "lgbm":
        from lightgbm import LGBMRegressor

        est = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
        return ModelSpec(
            name="lgbm",
            estimator=est,
            params={
                "n_estimators": 2000,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            scale_numeric=False,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. Expected one of ridge, rf, knn, svr, xgb, lgbm"
    )
