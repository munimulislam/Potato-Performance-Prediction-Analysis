"""
@File - model_tuning.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 27/07/2026
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import mlflow
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.pipeline import Pipeline

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact, log_dict_artifact
from ml.common.splits import group_kfold_splits
from ml.common.preprocess import make_preprocessor
from ml.common.metrics import compute_metrics, mean_std_se

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"oa/exp3_model_tuning_{datetime.now()}"
PARENT_RUN_NAME = "exp3_model_tuning"

GROUP_COL = "name1"
N_SPLITS = 5

ENV_COL = "env_type"

THRESHOLD = 0.5

MEDIUM_MAX_MISSING_FRAC = 0.5

RANDOM_STATE = 42
N_TRIALS = 30
TIMEOUT_SECONDS = None
N_STARTUP_TRIALS = 5
PRUNER_WARMUP_STEPS = 1

PROGRESS = True
PROGRESS_EVERY_FOLD = True

LOG_FOLD_PREDICTIONS_PER_TRIAL = False
LOG_FOLD_METRICS_PER_TRIAL = True


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"dataset.yaml must parse to a dict, got {type(obj)}")
    return obj


def _progress(msg: str) -> None:
    if PROGRESS:
        print(msg, flush=True)


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


def _upper_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def _build_row_id(df: pd.DataFrame, key_cols: List[str], sep: str = "|") -> np.ndarray:
    if not key_cols:
        return df.index.astype(str).to_numpy()
    tmp = df[key_cols].astype("string").fillna("<NA>")
    out = tmp.iloc[:, 0]
    for c in tmp.columns[1:]:
        out = out + sep + tmp[c]
    return out.astype("string").to_numpy()


def _numeric_missing_frac(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    if not numeric_cols:
        return np.zeros(len(df), dtype=float)
    return df[numeric_cols].isna().mean(axis=1).to_numpy(dtype=float)


def _mask_missing_band(
    miss_frac: np.ndarray, band: str, medium_max: float
) -> np.ndarray:
    if band == "complete":
        return miss_frac == 0.0
    if band == "medium":
        return (miss_frac > 0.0) & (miss_frac <= medium_max)
    if band == "incomplete":
        return miss_frac > medium_max
    raise ValueError(f"Unknown band: {band}")


def _metrics_for_mask(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    n = int(mask.sum())
    if n <= 0:
        return {
            "n": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson_r": float("nan"),
            "r2": float("nan"),
        }

    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    met = compute_metrics(yt, yp)

    return {
        "n": float(n),
        "rmse": float(met.rmse),
        "mae": float(met.mae),
        "pearson_r": float(met.pearson_r),
        "r2": float(met.r2),
    }


def _spearman_variety_level(
    name1: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, int]:
    if mask.sum() == 0:
        return float("nan"), 0

    d = pd.DataFrame(
        {
            "name1": name1[mask],
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
        }
    ).dropna(subset=["name1", "y_true", "y_pred"])

    if d.empty:
        return float("nan"), 0

    agg = d.groupby("name1", as_index=False).agg(
        y_true=("y_true", "mean"),
        y_pred=("y_pred", "mean"),
    )
    n_var = int(len(agg))
    if n_var < 2:
        return float("nan"), n_var

    rho = agg["y_true"].corr(agg["y_pred"], method="spearman")
    return float(rho), n_var


def main() -> int:
    dataset_cfg = _read_yaml(DATASET_YAML_PATH)
    if "dataset" not in dataset_cfg:
        raise ValueError("dataset.yaml must contain top-level key 'dataset:'")

    ds = dataset_cfg["dataset"]
    feats = ds["features"]

    dataset_path = str(ds["path"])
    target_col = str(feats["target"])
    key_cols: List[str] = list(feats.get("keys", []))
    numeric_cols: List[str] = list(feats.get("numeric", []))
    categorical_cols: List[str] = list(feats.get("categorical", []))

    init_mlflow(TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    _progress(f"[Exp3] Loading dataset: {dataset_path}")
    df = load_dataframe(dataset_path)

    required_cols = sorted(
        set(
            [target_col, GROUP_COL, ENV_COL]
            + key_cols
            + numeric_cols
            + categorical_cols
        )
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required_cols].copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols:
        df[c] = df[c].astype("string")
    df[GROUP_COL] = df[GROUP_COL].astype("string")
    df[ENV_COL] = df[ENV_COL].astype("string")
    for c in key_cols:
        df[c] = df[c].astype("string")

    if df[target_col].isna().all():
        raise ValueError(f"Target '{target_col}' is all-null after coercion.")

    feature_cols = numeric_cols + categorical_cols

    y_all = df[target_col].to_numpy(dtype=float)
    env_upper_all = _upper_str(df[ENV_COL]).to_numpy()
    name1_all = df[GROUP_COL].to_numpy()
    miss_frac_all = _numeric_missing_frac(df, numeric_cols)
    row_id_all = _build_row_id(df, key_cols)

    splits = list(
        group_kfold_splits(
            df, target_col=target_col, group_col=GROUP_COL, n_splits=N_SPLITS
        )
    )
    _progress(f"[Exp3] Prepared {len(splits)} folds (GroupKFold on {GROUP_COL}).")

    pre = make_preprocessor(
        profile="linear", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        shrinking = trial.suggest_categorical("shrinking", [True, False])
        cache_size = trial.suggest_categorical("cache_size", [200, 500, 1000])

        params = {
            "kernel": "rbf",
            "C": C,
            "epsilon": epsilon,
            "gamma": gamma,
            "shrinking": shrinking,
            "cache_size": cache_size,
        }

        model_spec = None
        try:
            from ml.common.models import (
                make_model,
            )

            model_spec = make_model("svr", params)
        except Exception as e:
            raise RuntimeError(f"make_model('svr', ...) failed: {e}")

        pipe = Pipeline(steps=[("pre", pre), ("est", model_spec.estimator)])

        with mlflow.start_run(run_name=f"trial_{trial.number:04d}", nested=True):
            mlflow.set_tags(
                {
                    "phase": "exp3",
                    "model": "svr",
                    "profile": "linear",
                    "optuna_trial": str(trial.number),
                }
            )
            mlflow.log_params(
                {
                    "threshold": THRESHOLD,
                    "group_col": GROUP_COL,
                    "n_splits": N_SPLITS,
                    **{f"svr__{k}": v for k, v in params.items()},
                }
            )

            fold_rows: List[Dict[str, Any]] = []
            t0 = time.perf_counter()

            for fold_i, sp in enumerate(splits, start=1):
                fold_t0 = time.perf_counter()

                train_idx = sp.train_idx
                test_idx = sp.test_idx

                train_keep = miss_frac_all[train_idx] <= THRESHOLD
                kept_train_idx = train_idx[train_keep]
                kept_train_idx = kept_train_idx[~np.isnan(y_all[kept_train_idx])]

                y_test = y_all[test_idx]
                y_obs_mask = ~np.isnan(y_test)

                if PROGRESS and PROGRESS_EVERY_FOLD:
                    _progress(
                        f"    [trial {trial.number:04d}] fold {fold_i}/{len(splits)} | "
                        f"kept_train={len(kept_train_idx)} scored_test={int(y_obs_mask.sum())}"
                    )

                fold_row: Dict[str, Any] = {
                    "trial": int(trial.number),
                    "fold": int(sp.fold),
                    "train_total_n": int(len(train_idx)),
                    "train_kept_n": int(len(kept_train_idx)),
                    "train_kept_frac": (
                        float(len(kept_train_idx) / len(train_idx))
                        if len(train_idx)
                        else float("nan")
                    ),
                    "test_total_n": int(len(test_idx)),
                    "test_scored_n": int(y_obs_mask.sum()),
                }

                if len(kept_train_idx) == 0 or y_obs_mask.sum() == 0:
                    fold_row.update({"rmse_all": float("nan")})
                    fold_rows.append(fold_row)
                    continue

                X_train = df.iloc[kept_train_idx][feature_cols]
                y_train = y_all[kept_train_idx]
                X_test = df.iloc[test_idx][feature_cols]

                pipe.fit(X_train, y_train)
                y_pred = np.asarray(pipe.predict(X_test), dtype=float)

                env_test = env_upper_all[test_idx]
                miss_test = miss_frac_all[test_idx]

                masks = {
                    "all": y_obs_mask,
                    "ne": y_obs_mask & (env_test == "NE"),
                    "med": y_obs_mask & (env_test == "MED"),
                    "complete": y_obs_mask
                    & _mask_missing_band(
                        miss_test, "complete", MEDIUM_MAX_MISSING_FRAC
                    ),
                    "medium": y_obs_mask
                    & _mask_missing_band(miss_test, "medium", MEDIUM_MAX_MISSING_FRAC),
                    "incomplete": y_obs_mask
                    & _mask_missing_band(
                        miss_test, "incomplete", MEDIUM_MAX_MISSING_FRAC
                    ),
                }

                for subset, mask in masks.items():
                    mm = _metrics_for_mask(y_true=y_test, y_pred=y_pred, mask=mask)
                    fold_row.update(
                        {
                            f"rmse_{subset}": mm["rmse"],
                            f"mae_{subset}": mm["mae"],
                            f"r2_{subset}": mm["r2"],
                            f"pearson_r_{subset}": mm["pearson_r"],
                            f"n_{subset}": mm["n"],
                        }
                    )

                for subset in ("all", "ne", "med"):
                    rho, nvar = _spearman_variety_level(
                        name1=name1_all[test_idx],
                        y_true=y_test,
                        y_pred=y_pred,
                        mask=masks[subset],
                    )
                    fold_row.update(
                        {
                            f"spearman_variety_{subset}": float(rho),
                            f"n_varieties_{subset}": float(nvar),
                        }
                    )

                fold_rows.append(fold_row)

                done = pd.DataFrame(fold_rows)
                if "rmse_all" in done.columns:
                    running_mean = float(done["rmse_all"].dropna().mean())
                else:
                    running_mean = float(done["rmse_all"].dropna().mean())

                trial.report(running_mean, step=fold_i)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    mlflow.log_metric("rmse_all_running_mean_at_prune", running_mean)
                    raise optuna.TrialPruned()

                if PROGRESS and PROGRESS_EVERY_FOLD:
                    _progress(
                        f"      fold time: {_fmt_seconds(time.perf_counter() - fold_t0)}"
                    )

            fold_metrics_df = pd.DataFrame(fold_rows)

            rmse_mean, rmse_std, rmse_se, n_valid = mean_std_se(
                fold_metrics_df["rmse_all"]
            )

            def _mean(col: str) -> float:
                return float(fold_metrics_df[col].mean(skipna=True))

            summary = {
                "rmse_all_mean": float(rmse_mean),
                "rmse_all_std": float(rmse_std),
                "rmse_all_se": float(rmse_se),
                "rmse_all_n_folds_valid": float(n_valid),
                "rmse_ne_mean": _mean("rmse_ne"),
                "rmse_med_mean": _mean("rmse_med"),
                "r2_all_mean": _mean("r2_all"),
                "r2_ne_mean": _mean("r2_ne"),
                "r2_med_mean": _mean("r2_med"),
                "mae_all_mean": _mean("mae_all"),
                "mae_ne_mean": _mean("mae_ne"),
                "mae_med_mean": _mean("mae_med"),
                "spearman_variety_all_mean": _mean("spearman_variety_all"),
                "spearman_variety_ne_mean": _mean("spearman_variety_ne"),
                "spearman_variety_med_mean": _mean("spearman_variety_med"),
                "train_kept_frac_mean": float(
                    fold_metrics_df["train_kept_frac"].mean(skipna=True)
                ),
                "test_scored_n_mean": float(
                    fold_metrics_df["test_scored_n"].mean(skipna=True)
                ),
            }

            mlflow.log_metrics(summary)
            mlflow.log_metric("trial_wall_time_sec", float(time.perf_counter() - t0))

            if LOG_FOLD_METRICS_PER_TRIAL:
                log_df_artifact(
                    fold_metrics_df, f"exp3_fold_metrics__trial_{trial.number:04d}.csv"
                )

            return float(summary["rmse_all_mean"])

    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=PRUNER_WARMUP_STEPS
    )

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags(
            {
                "phase": "exp3",
                "cv": "cv1_groupkfold",
                "model": "svr",
            }
        )
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "env_col": ENV_COL,
                "group_col": GROUP_COL,
                "n_splits": N_SPLITS,
                "threshold": THRESHOLD,
                "n_trials": N_TRIALS,
                "timeout_seconds": (
                    TIMEOUT_SECONDS if TIMEOUT_SECONDS is not None else "None"
                ),
                "medium_max_missing_frac": MEDIUM_MAX_MISSING_FRAC,
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "n_startup_trials": N_STARTUP_TRIALS,
                "pruner_warmup_steps": PRUNER_WARMUP_STEPS,
            }
        )

        _progress(
            f"[Exp3] Starting Optuna: n_trials={N_TRIALS}, threshold={THRESHOLD}, folds={N_SPLITS}"
        )
        t_study0 = time.perf_counter()

        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

        _progress(
            f"[Exp3] Optuna finished in {_fmt_seconds(time.perf_counter() - t_study0)}"
        )
        _progress(f"[Exp3] Best value (rmse_all_mean): {study.best_value}")
        _progress(f"[Exp3] Best params: {study.best_params}")

        log_dict_artifact(
            {
                "best_value_rmse_all_mean": float(study.best_value),
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            },
            "exp3_best_params.json",
        )

        trials_df = study.trials_dataframe()
        log_df_artifact(trials_df, "exp3_optuna_trials.csv")

        for k, v in study.best_params.items():
            mlflow.log_param(f"best__{k}", v)
        mlflow.log_metric("best_rmse_all_mean", float(study.best_value))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
