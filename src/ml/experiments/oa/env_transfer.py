"""
@File - env_transfer.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 15/08/2026
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact
from ml.common.preprocess import make_preprocessor

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "oa/env_transfer"
PARENT_RUN_NAME = "env_transfer_test"

ENV_COL = "env_type"
GROUP_COL = "name1"

THRESHOLD = 1
N_SPLITS = 5

XGB_PARAMS = {
    "n_estimators": 1246,
    "learning_rate": 0.05579378978522556,
    "max_depth": 4,
    "min_child_weight": 16.17686906786567,
    "subsample": 0.7266044212478868,
    "colsample_bytree": 0.715268716314553,
    "reg_lambda": 14.155955356878192,
    "reg_alpha": 6.324081256438533,
    "gamma": 0.16930231325154688,
}

PROGRESS = True


def _progress(msg: str) -> None:
    if PROGRESS:
        print(msg, flush=True)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("dataset.yaml must parse to a dict")
    return obj


def _upper_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def _numeric_missing_frac(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    if not numeric_cols:
        return np.zeros(len(df), dtype=float)
    return df[numeric_cols].isna().mean(axis=1).to_numpy(dtype=float)


def _metrics_rmse_mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    if int(mask.sum()) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    return _metrics_rmse_mae_r2(yt, yp)


def _aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not fold_metrics:
        return {
            k: float("nan")
            for k in (
                "rmse_all",
                "mae_all",
                "r2_all",
                "rmse_ne",
                "mae_ne",
                "r2_ne",
                "rmse_med",
                "mae_med",
                "r2_med",
            )
        }
    dfm = pd.DataFrame(fold_metrics)
    return {k: float(dfm[k].mean(skipna=True)) for k in dfm.columns}


def _make_pipe(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    pre = make_preprocessor(
        profile="booster", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )
    est = XGBRegressor(**XGB_PARAMS)
    return Pipeline([("pre", pre), ("est", est)])


def eval_global_cv1(
    df: pd.DataFrame,
    *,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, float]:

    feature_cols = numeric_cols + categorical_cols

    y_all = df[target_col].to_numpy(dtype=float)
    env = _upper_str(df[ENV_COL]).to_numpy()
    miss = _numeric_missing_frac(df, numeric_cols)

    gkf = GroupKFold(n_splits=N_SPLITS)
    pipe = _make_pipe(numeric_cols, categorical_cols)

    fold_metrics: List[Dict[str, float]] = []

    for fold, (tri, tei) in enumerate(gkf.split(df, groups=df[GROUP_COL].astype(str))):
        keep = miss[tri] <= THRESHOLD
        kept_tri = tri[keep]
        kept_tri = kept_tri[~np.isnan(y_all[kept_tri])]

        y_te = y_all[tei]
        obs = ~np.isnan(y_te)
        if len(kept_tri) == 0 or int(obs.sum()) == 0:
            continue

        X_tr = df.iloc[kept_tri][feature_cols]
        y_tr = y_all[kept_tri]
        X_te = df.iloc[tei][feature_cols]
        pred = pipe.fit(X_tr, y_tr).predict(X_te)
        pred = np.asarray(pred, dtype=float)

        env_te = env[tei]

        m_all = obs
        m_ne = obs & (env_te == "NE")
        m_med = obs & (env_te == "MED")

        met_all = _subset_metrics(y_te, pred, m_all)
        met_ne = _subset_metrics(y_te, pred, m_ne)
        met_med = _subset_metrics(y_te, pred, m_med)

        fold_metrics.append(
            {
                "rmse_all": met_all["rmse"],
                "mae_all": met_all["mae"],
                "r2_all": met_all["r2"],
                "rmse_ne": met_ne["rmse"],
                "mae_ne": met_ne["mae"],
                "r2_ne": met_ne["r2"],
                "rmse_med": met_med["rmse"],
                "mae_med": met_med["mae"],
                "r2_med": met_med["r2"],
            }
        )

    return _aggregate_fold_metrics(fold_metrics)


def eval_env_specific_cv1(
    df: pd.DataFrame,
    *,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    env_value: str,
) -> Dict[str, float]:

    feature_cols = numeric_cols + categorical_cols
    env_upper = _upper_str(df[ENV_COL])
    df_sub = df.loc[env_upper == env_value].copy()

    if df_sub.empty:
        return {
            k: float("nan")
            for k in (
                "rmse_all",
                "mae_all",
                "r2_all",
                "rmse_ne",
                "mae_ne",
                "r2_ne",
                "rmse_med",
                "mae_med",
                "r2_med",
            )
        }

    y_all = df_sub[target_col].to_numpy(dtype=float)
    env = _upper_str(df_sub[ENV_COL]).to_numpy()
    miss = _numeric_missing_frac(df_sub, numeric_cols)

    gkf = GroupKFold(n_splits=N_SPLITS)
    pipe = _make_pipe(numeric_cols, categorical_cols)

    fold_metrics: List[Dict[str, float]] = []

    for fold, (tri, tei) in enumerate(
        gkf.split(df_sub, groups=df_sub[GROUP_COL].astype(str))
    ):
        keep = miss[tri] <= THRESHOLD
        kept_tri = tri[keep]
        kept_tri = kept_tri[~np.isnan(y_all[kept_tri])]

        y_te = y_all[tei]
        obs = ~np.isnan(y_te)
        if len(kept_tri) == 0 or int(obs.sum()) == 0:
            continue

        X_tr = df_sub.iloc[kept_tri][feature_cols]
        y_tr = y_all[kept_tri]
        X_te = df_sub.iloc[tei][feature_cols]
        pred = pipe.fit(X_tr, y_tr).predict(X_te)
        pred = np.asarray(pred, dtype=float)

        env_te = env[tei]
        m_all = obs
        m_ne = obs & (env_te == "NE")
        m_med = obs & (env_te == "MED")

        met_all = _subset_metrics(y_te, pred, m_all)
        met_ne = _subset_metrics(y_te, pred, m_ne)
        met_med = _subset_metrics(y_te, pred, m_med)

        fold_metrics.append(
            {
                "rmse_all": met_all["rmse"],
                "mae_all": met_all["mae"],
                "r2_all": met_all["r2"],
                "rmse_ne": met_ne["rmse"],
                "mae_ne": met_ne["mae"],
                "r2_ne": met_ne["r2"],
                "rmse_med": met_med["rmse"],
                "mae_med": met_med["mae"],
                "r2_med": met_med["r2"],
            }
        )

    return _aggregate_fold_metrics(fold_metrics)


def eval_transfer_unseen_variety(
    df: pd.DataFrame,
    *,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    train_env: str,
    test_env: str,
) -> Dict[str, float]:

    feature_cols = numeric_cols + categorical_cols

    env_upper = _upper_str(df[ENV_COL])
    df_sub = df.loc[env_upper.isin([train_env, test_env])].reset_index(drop=True)
    if df_sub.empty:
        return {
            k: float("nan")
            for k in (
                "rmse_all",
                "mae_all",
                "r2_all",
                "rmse_ne",
                "mae_ne",
                "r2_ne",
                "rmse_med",
                "mae_med",
                "r2_med",
            )
        }

    env = _upper_str(df_sub[ENV_COL]).to_numpy()
    y_all = df_sub[target_col].to_numpy(dtype=float)
    miss = _numeric_missing_frac(df_sub, numeric_cols)

    gkf = GroupKFold(n_splits=N_SPLITS)
    pipe = _make_pipe(numeric_cols, categorical_cols)

    fold_metrics: List[Dict[str, float]] = []

    for fold, (tri, tei) in enumerate(
        gkf.split(df_sub, groups=df_sub[GROUP_COL].astype(str))
    ):
        tr_mask = np.zeros(len(df_sub), dtype=bool)
        tr_mask[tri] = True
        te_mask = np.zeros(len(df_sub), dtype=bool)
        te_mask[tei] = True

        train_sel = (
            tr_mask & (env == train_env) & (miss <= THRESHOLD) & (~np.isnan(y_all))
        )
        test_sel = te_mask & (env == test_env)

        df_train = df_sub.loc[train_sel]
        df_test = df_sub.loc[test_sel]
        if df_train.empty or df_test[target_col].notna().sum() == 0:
            continue

        if set(df_train[GROUP_COL]) & set(df_test[GROUP_COL]):
            raise AssertionError(
                "Transfer leakage: variety overlap between train and test"
            )

        X_tr = df_train[feature_cols]
        y_tr = df_train[target_col].to_numpy(dtype=float)
        X_te = df_test[feature_cols]

        pred = pipe.fit(X_tr, y_tr).predict(X_te)
        pred = np.asarray(pred, dtype=float)

        y_te = df_test[target_col].to_numpy(dtype=float)
        obs = ~np.isnan(y_te)
        if int(obs.sum()) == 0:
            continue

        env_te = _upper_str(df_test[ENV_COL]).to_numpy()

        m_all = obs
        m_ne = obs & (env_te == "NE")
        m_med = obs & (env_te == "MED")

        met_all = _subset_metrics(y_te, pred, m_all)
        met_ne = _subset_metrics(y_te, pred, m_ne)
        met_med = _subset_metrics(y_te, pred, m_med)

        fold_metrics.append(
            {
                "rmse_all": met_all["rmse"],
                "mae_all": met_all["mae"],
                "r2_all": met_all["r2"],
                "rmse_ne": met_ne["rmse"],
                "mae_ne": met_ne["mae"],
                "r2_ne": met_ne["r2"],
                "rmse_med": met_med["rmse"],
                "mae_med": met_med["mae"],
                "r2_med": met_med["r2"],
            }
        )

    return _aggregate_fold_metrics(fold_metrics)


def main() -> int:
    cfg = _read_yaml(DATASET_YAML_PATH)
    if "dataset" not in cfg:
        raise ValueError("dataset.yaml must contain top-level key 'dataset:'")

    ds = cfg["dataset"]
    feats = ds["features"]
    dataset_path = str(ds["path"])
    target_col = str(feats["target"])
    key_cols: List[str] = list(feats.get("keys", []))
    numeric_cols: List[str] = list(feats.get("numeric", []))
    categorical_cols: List[str] = list(feats.get("categorical", []))
    feature_cols = numeric_cols + categorical_cols

    init_mlflow(TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    _progress(f"Loading dataset: {dataset_path}")
    df = load_dataframe(dataset_path)

    required = sorted(
        set([target_col, ENV_COL, GROUP_COL] + numeric_cols + categorical_cols)
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required].copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols + [ENV_COL, GROUP_COL]:
        df[c] = df[c].astype("string")

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags({"phase": "env_transfer"})
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "threshold": THRESHOLD,
                "n_splits": N_SPLITS,
                "moel_params": json.dumps(XGB_PARAMS),
            }
        )

        summary_rows = []

        _progress("global_cv1 ...")
        with mlflow.start_run(run_name="A_global_cv1", nested=True):
            m = eval_global_cv1(
                df,
                target_col=target_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            )
            mlflow.log_metrics(m)
            summary_rows.append({"run": "A_global_cv1", **m})

        _progress("ne_only_cv1 ...")
        with mlflow.start_run(run_name="B_ne_only_cv1", nested=True):
            m = eval_env_specific_cv1(
                df,
                target_col=target_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                env_value="NE",
            )
            mlflow.log_metrics(m)
            summary_rows.append({"run": "B_ne_only_cv1", **m})

        _progress("med_only_cv1 ...")
        with mlflow.start_run(run_name="B_med_only_cv1", nested=True):
            m = eval_env_specific_cv1(
                df,
                target_col=target_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                env_value="MED",
            )
            mlflow.log_metrics(m)
            summary_rows.append({"run": "B_med_only_cv1", **m})

        _progress("ne_to_med_unseen ...")
        with mlflow.start_run(run_name="C_transfer_ne_to_med_unseen", nested=True):
            m = eval_transfer_unseen_variety(
                df,
                target_col=target_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                train_env="NE",
                test_env="MED",
            )
            mlflow.log_metrics(m)
            summary_rows.append({"run": "C_transfer_ne_to_med_unseen", **m})

        _progress("med_to_ne_unseen ...")
        with mlflow.start_run(run_name="C_transfer_med_to_ne_unseen", nested=True):
            m = eval_transfer_unseen_variety(
                df,
                target_col=target_col,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                train_env="MED",
                test_env="NE",
            )
            mlflow.log_metrics(m)
            summary_rows.append({"run": "C_transfer_med_to_ne_unseen", **m})

        summary_df = pd.DataFrame(summary_rows)
        log_df_artifact(summary_df, "env_transfer_summary.csv")

        _progress("Complete. See MLflow artifact: env_transfer_summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
