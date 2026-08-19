"""
@File - prod_model.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 15/08/2026
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

from xgboost import XGBRegressor

from ml.common.data import load_dataframe
from ml.common.mlflow import (
    init_mlflow,
    log_dict_artifact,
    log_joblib_artifact,
    log_df_artifact,
)
from ml.common.preprocess import make_preprocessor
from ml.common.splits import group_kfold_splits

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "oa/prod_model"
RUN_NAME_DEFAULT = f"xgb_prod"

REGISTERED_MODEL_NAME_DEFAULT = "oa_xgb_prod"

THRESHOLD = 1
SIGNATURE_SAMPLE_N = 200

RANDOM_STATE = 42

GROUP_COL = "name1"
ENV_COL = "env_type"

XGB_FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

ENQUEUE_BASELINE = True
BASELINE_TRIAL = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"dataset.yaml must parse to a dict, got {type(obj)}")
    return obj


def _upper_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def _numeric_missing_frac(df: pd.DataFrame, numeric_cols: List[str]) -> pd.Series:
    if not numeric_cols:
        return pd.Series(0.0, index=df.index)
    return df[numeric_cols].isna().mean(axis=1)


def _validate_feature_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def _coerce_types(
    df: pd.DataFrame,
    *,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in categorical_cols:
        out[c] = out[c].astype("string")
    return out


def _metrics_rmse_mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    if int(mask.sum()) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return _metrics_rmse_mae_r2(y_true[mask].astype(float), y_pred[mask].astype(float))


def _cv_metrics_best_params(
    *,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    best_params: Dict[str, Any],
) -> Dict[str, float]:

    y_all = df[target_col].to_numpy(dtype=float)
    env_upper = _upper_str(df["env_type"])

    miss_frac = _numeric_missing_frac(df, numeric_cols=numeric_cols).to_numpy(
        dtype=float
    )
    splits = list(
        group_kfold_splits(df, target_col=target_col, group_col="name1", n_splits=5)
    )

    pre = make_preprocessor(
        profile="booster", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )

    fold_vals = {
        "rmse_all": [],
        "mae_all": [],
        "r2_all": [],
        "rmse_ne": [],
        "mae_ne": [],
        "r2_ne": [],
        "rmse_med": [],
        "mae_med": [],
        "r2_med": [],
    }

    for sp in splits:
        tri, tei = sp.train_idx, sp.test_idx

        keep = miss_frac[tri] <= THRESHOLD
        kept_tri = tri[keep]
        kept_tri = kept_tri[~np.isnan(y_all[kept_tri])]

        y_te = y_all[tei]
        obs = ~np.isnan(y_te)
        if len(kept_tri) == 0 or int(obs.sum()) == 0:
            continue

        X_tr = df.iloc[kept_tri][feature_cols]
        y_tr = y_all[kept_tri]
        X_te = df.iloc[tei][feature_cols]

        model = XGBRegressor(**{**XGB_FIXED_PARAMS, **best_params})
        pipe = Pipeline([("pre", pre), ("est", model)])
        pipe.fit(X_tr, y_tr)

        pred = np.asarray(pipe.predict(X_te), dtype=float)

        env_te = env_upper.iloc[tei].to_numpy()
        m_all = obs
        m_ne = obs & (env_te == "NE")
        m_med = obs & (env_te == "MED")

        met_all = _subset_metrics(y_te, pred, m_all)
        met_ne = _subset_metrics(y_te, pred, m_ne)
        met_med = _subset_metrics(y_te, pred, m_med)

        for k in ("rmse", "mae", "r2"):
            fold_vals[f"{k}_all"].append(met_all[k])
            fold_vals[f"{k}_ne"].append(met_ne[k])
            fold_vals[f"{k}_med"].append(met_med[k])

    out = {}
    for k, v in fold_vals.items():
        out[k] = float(np.nanmean(v)) if len(v) else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=RUN_NAME_DEFAULT, help="MLflow run name")
    ap.add_argument(
        "--register", action="store_true", help="Register model to MLflow registry"
    )
    ap.add_argument("--registered-model-name", default=REGISTERED_MODEL_NAME_DEFAULT)
    ap.add_argument("--n-trials", type=int, default=40)
    args = ap.parse_args()

    cfg = _read_yaml(DATASET_YAML_PATH)
    if "dataset" not in cfg:
        raise ValueError("dataset.yaml must contain top-level key 'dataset:'")

    ds = cfg["dataset"]
    feats = ds["features"]

    dataset_path = str(ds["path"])
    target_col = str(feats["target"])
    numeric_cols: List[str] = list(feats.get("numeric", []))
    categorical_cols: List[str] = list(feats.get("categorical", []))
    feature_cols = numeric_cols + categorical_cols

    init_mlflow(TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    t0 = time.perf_counter()
    df = load_dataframe(dataset_path)

    required_cols = sorted(set([target_col, GROUP_COL, ENV_COL] + feature_cols))
    _validate_feature_columns_exist(df, required_cols)
    df = df[required_cols].copy()

    df = _coerce_types(
        df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    df[GROUP_COL] = df[GROUP_COL].astype("string")
    df[ENV_COL] = df[ENV_COL].astype("string")

    if df[target_col].isna().all():
        raise ValueError(
            f"Target '{target_col}' is all-null after coercion. Cannot train."
        )

    miss_frac = _numeric_missing_frac(df, numeric_cols=numeric_cols)
    train_mask = (miss_frac <= THRESHOLD) & (df[target_col].notna())
    df_train = df.loc[train_mask].copy()

    if df_train.empty:
        raise ValueError(
            "No rows left after training filter. Check threshold policy and data."
        )

    y_all = df[target_col].to_numpy(dtype=float)
    miss_frac_np = miss_frac.to_numpy(dtype=float)
    splits = list(
        group_kfold_splits(df, target_col=target_col, group_col=GROUP_COL, n_splits=5)
    )

    pre = make_preprocessor(
        profile="booster", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        }

        fold_rmses = []

        for sp in splits:
            tri, tei = sp.train_idx, sp.test_idx

            keep = miss_frac_np[tri] <= THRESHOLD
            kept_tri = tri[keep]
            kept_tri = kept_tri[~np.isnan(y_all[kept_tri])]

            y_te = y_all[tei]
            obs = ~np.isnan(y_te)

            if len(kept_tri) == 0 or int(obs.sum()) == 0:
                continue

            X_tr = df.iloc[kept_tri][feature_cols]
            y_tr = y_all[kept_tri]
            X_te = df.iloc[tei][feature_cols]

            model = XGBRegressor(**{**XGB_FIXED_PARAMS, **params})
            pipe = Pipeline([("pre", pre), ("est", model)])
            pipe.fit(X_tr, y_tr)

            pred = np.asarray(pipe.predict(X_te), dtype=float)
            rmse = float(
                np.sqrt(
                    mean_squared_error(y_te[obs].astype(float), pred[obs].astype(float))
                )
            )
            fold_rmses.append(rmse)

        if not fold_rmses:
            return float("inf")

        return float(np.mean(fold_rmses))

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if ENQUEUE_BASELINE:
        study.enqueue_trial(BASELINE_TRIAL)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "stage": "prod_tune_and_train",
                "model_family": "xgb",
                "preprocess_profile": "booster",
                "target": target_col,
            }
        )

        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "threshold_train_missing_frac_numeric_leq": THRESHOLD,
                "n_rows_total": int(len(df)),
                "n_rows_train_used": int(len(df_train)),
                "train_kept_frac": (
                    float(len(df_train) / len(df)) if len(df) else float("nan")
                ),
                "n_trials": int(args.n_trials),
                **{f"xgb_fixed__{k}": v for k, v in XGB_FIXED_PARAMS.items()},
            }
        )

        study.optimize(objective, n_trials=args.n_trials)

        best_params = dict(study.best_params)

        log_dict_artifact(best_params, "best_params.json")
        log_df_artifact(study.trials_dataframe(), "optuna_trials.csv")

        cv_metrics = _cv_metrics_best_params(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            best_params=best_params,
        )

        mlflow.log_metrics(cv_metrics)

        X_train = df_train[feature_cols]
        y_train = df_train[target_col].to_numpy(dtype=float)

        final_model = XGBRegressor(**{**XGB_FIXED_PARAMS, **best_params})
        final_pipe = Pipeline([("pre", pre), ("est", final_model)])
        final_pipe.fit(X_train, y_train)

        sample_n = min(SIGNATURE_SAMPLE_N, len(df_train))
        X_sig = X_train.head(sample_n)
        y_sig_pred = final_pipe.predict(X_sig)
        signature = infer_signature(X_sig, y_sig_pred)

        schema = {
            "target": target_col,
            "features": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "feature_cols_order": feature_cols,
            },
            "training_policy": {
                "train_row_filter": f"missing_frac_numeric <= {THRESHOLD} AND target_not_null",
                "inference_filter": "none (pipeline handles missingness)",
            },
            "tuning": {
                "method": "optuna_full_data_cv1",
                "objective": "rmse_all (cv mean)",
                "n_trials": int(args.n_trials),
            },
        }

        log_df_artifact(X_sig, "sample.csv")
        log_dict_artifact(schema, "model_schema.json")

        log_joblib_artifact(final_pipe, "pipeline.joblib")

        register_name = args.registered_model_name if args.register else None
        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_sig,
            registered_model_name=register_name,
            skops_trusted_types=True,
            serialization_format="cloudpickle",
        )

        elapsed = time.perf_counter() - t0
        mlflow.log_metric("wall_time_sec", float(elapsed))

        print(f"Trained+tuned+logged XGB production model in {elapsed:.1f}s")
        print(f"MLflow run_id: {run_id}")
        print(f"Load by run: mlflow.sklearn.load_model('runs:/{run_id}/model')")
        if args.register:
            print(f"Registered as: models:/{args.registered_model_name}/latest")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
