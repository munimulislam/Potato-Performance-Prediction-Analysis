"""
@File - xgb_tuning.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 14/08/2026
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
import mlflow
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact
from ml.common.splits import group_kfold_splits
from ml.common.preprocess import make_preprocessor

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "oa/model_tuning"
PARENT_RUN_NAME = "xgb"

GROUP_COL = "name1"
ENV_COL = "env_type"

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3

THRESHOLD = 1
RANDOM_STATE = 42
N_TRIALS_PER_OUTER_FOLD = 20

XGB_FIXED = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
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


def _to_float32(X):
    return X.astype(np.float32)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _metrics_rmse_mse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    if int(mask.sum()) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return _metrics_rmse_mse_r2(y_true[mask].astype(float), y_pred[mask].astype(float))


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
    out[GROUP_COL] = out[GROUP_COL].astype("string")
    out[ENV_COL] = out[ENV_COL].astype("string")
    return out


def main() -> int:
    cfg = _read_yaml(DATASET_YAML_PATH)
    ds = cfg["dataset"]
    feats = ds["features"]

    dataset_path = str(ds["path"])
    target_col = str(feats["target"])
    numeric_cols: List[str] = list(feats.get("numeric", []))
    categorical_cols: List[str] = list(feats.get("categorical", []))
    feature_cols = numeric_cols + categorical_cols

    init_mlflow(TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    _progress(f"Loading dataset: {dataset_path}")
    df = load_dataframe(dataset_path)

    required_cols = sorted(set([target_col, GROUP_COL, ENV_COL] + feature_cols))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required_cols].copy()
    df = _coerce_types(
        df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    if df[target_col].isna().all():
        raise ValueError(f"Target '{target_col}' is all-null after coercion.")

    pre = make_preprocessor(
        profile="booster", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )
    preprocessor = Pipeline(
        [("pre", pre), ("to_float32", FunctionTransformer(_to_float32))]
    )

    outer_splits = list(
        group_kfold_splits(
            df, target_col=target_col, group_col=GROUP_COL, n_splits=OUTER_N_SPLITS
        )
    )

    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.set_tags(
            {"design": "nested_cv", "model": "xgb", "cv": "cv1_groupkfold_outer_inner"}
        )
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "group_col": GROUP_COL,
                "env_col": ENV_COL,
                "threshold": THRESHOLD,
                "outer_n_splits": OUTER_N_SPLITS,
                "inner_n_splits": INNER_N_SPLITS,
                "n_trials_per_outer_fold": N_TRIALS_PER_OUTER_FOLD,
            }
        )

        best_params_so_far: List[Dict[str, Any]] = []

        for outer_fold_i, outer in enumerate(outer_splits, start=1):
            t_outer = time.perf_counter()
            _progress(f"\nOUTER fold {outer_fold_i}/{OUTER_N_SPLITS}")

            df_outer_train = df.iloc[outer.train_idx].reset_index(drop=True)
            df_outer_test = df.iloc[outer.test_idx].reset_index(drop=True)

            miss_outer_train = _numeric_missing_frac(df_outer_train, numeric_cols)
            y_outer_train = df_outer_train[target_col].to_numpy(dtype=float)

            y_outer_test = df_outer_test[target_col].to_numpy(dtype=float)
            env_outer_test = _upper_str(df_outer_test[ENV_COL]).to_numpy()

            inner_splits = list(
                group_kfold_splits(
                    df_outer_train,
                    target_col=target_col,
                    group_col=GROUP_COL,
                    n_splits=INNER_N_SPLITS,
                )
            )

            with mlflow.start_run(
                run_name=f"outer_fold_{outer_fold_i:02d}", nested=True
            ):
                mlflow.log_param("outer_fold", outer_fold_i)

                sampler = TPESampler(seed=RANDOM_STATE + outer_fold_i)
                study = optuna.create_study(direction="minimize", sampler=sampler)

                def inner_objective(trial: optuna.Trial) -> float:
                    params = {
                        **XGB_FIXED,
                        "n_estimators": trial.suggest_int("n_estimators", 300, 1800),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.01, 0.3, log=True
                        ),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "min_child_weight": trial.suggest_float(
                            "min_child_weight", 1.0, 20.0, log=True
                        ),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree", 0.5, 1.0
                        ),
                        "reg_lambda": trial.suggest_float(
                            "reg_lambda", 1e-3, 50.0, log=True
                        ),
                        "reg_alpha": trial.suggest_float(
                            "reg_alpha", 1e-8, 10.0, log=True
                        ),
                        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                    }

                    rmses = []
                    for inner in inner_splits:
                        tri = inner.train_idx
                        vai = inner.test_idx

                        keep = miss_outer_train[tri] <= THRESHOLD
                        kept_tri = tri[keep]
                        kept_tri = kept_tri[~np.isnan(y_outer_train[kept_tri])]

                        y_va = y_outer_train[vai]
                        obs = ~np.isnan(y_va)

                        if len(kept_tri) == 0 or int(obs.sum()) == 0:
                            return float("inf")

                        X_tr = df_outer_train.iloc[kept_tri][feature_cols]
                        y_tr = y_outer_train[kept_tri]
                        X_va = df_outer_train.iloc[vai][feature_cols]

                        pipe = Pipeline(
                            [("pre", preprocessor), ("est", XGBRegressor(**params))]
                        )
                        pipe.fit(X_tr, y_tr)
                        pred = np.asarray(pipe.predict(X_va), dtype=float)

                        rmses.append(
                            _rmse(y_va[obs].astype(float), pred[obs].astype(float))
                        )

                    return float(np.mean(rmses))

                study.optimize(inner_objective, n_trials=N_TRIALS_PER_OUTER_FOLD)

                best_params = dict(study.best_params)
                best_value = float(study.best_value)
                best_params_so_far.append({"outer_fold": outer_fold_i, **best_params})
                mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

                keep_outer = miss_outer_train <= THRESHOLD
                kept = np.where(keep_outer)[0]
                kept = kept[~np.isnan(y_outer_train[kept])]
                if len(kept) == 0:
                    raise RuntimeError(
                        f"Outer fold {outer_fold_i}: no training rows after threshold filter."
                    )

                X_tr = df_outer_train.iloc[kept][feature_cols]
                y_tr = y_outer_train[kept]
                X_te = df_outer_test[feature_cols]

                final_params = {**XGB_FIXED, **best_params}
                final_pipe = Pipeline(
                    [("pre", preprocessor), ("est", XGBRegressor(**final_params))]
                )
                final_pipe.fit(X_tr, y_tr)

                pred_te = np.asarray(final_pipe.predict(X_te), dtype=float)

                obs_all = ~np.isnan(y_outer_test)
                is_ne = env_outer_test == "NE"
                is_med = env_outer_test == "MED"

                m_all = obs_all
                m_ne = obs_all & is_ne
                m_med = obs_all & is_med

                met_all = _subset_metrics(y_outer_test, pred_te, m_all)
                met_ne = _subset_metrics(y_outer_test, pred_te, m_ne)
                met_med = _subset_metrics(y_outer_test, pred_te, m_med)

                mlflow.log_metrics(
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

            _progress(
                f"OUTER fold {outer_fold_i} done in {time.perf_counter() - t_outer:.1f}s"
            )

        if best_params_so_far:
            best_df = pd.DataFrame(best_params_so_far).sort_values("outer_fold")
            log_df_artifact(best_df, "best_params.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
