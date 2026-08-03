"""
@File - model_tuning.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 27/07/2026
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import mlflow
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact, log_dict_artifact
from ml.common.splits import group_kfold_splits
from ml.common.preprocess import make_preprocessor
from ml.common.metrics import compute_metrics, mean_std_se

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "oa/exp3_svr_nestedcv_tuning"
PARENT_RUN_NAME = "exp3_svr_nestedcv_tuning"

GROUP_COL = "name1"
ENV_COL = "env_type"

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3

THRESHOLD = 0.5

MEDIUM_MAX_MISSING_FRAC = 0.5

RANDOM_STATE = 42
N_TRIALS_PER_OUTER_FOLD = 20

SVR_KERNEL = "rbf"
SVR_CACHE_SIZE = 1000

PROGRESS = True
PROGRESS_EVERY_OUTER_FOLD = True
PROGRESS_EVERY_TRIAL = True


def _progress(msg: str) -> None:
    if PROGRESS:
        print(msg, flush=True)


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


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


def _numeric_missing_frac(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    if not numeric_cols:
        return np.zeros(len(df), dtype=float)
    return df[numeric_cols].isna().mean(axis=1).to_numpy(dtype=float)


def _variety_level_rmse(
    *,
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

    rmse = float(
        root_mean_squared_error(agg["y_true"].to_numpy(), agg["y_pred"].to_numpy())
    )
    return rmse, n_var


def _variety_level_spearman(
    *,
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

    _progress(f"[Exp3] Loading dataset: {dataset_path}")
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
        raise ValueError(
            f"Target '{target_col}' is all-null after coercion. Cannot proceed."
        )

    outer_splits = list(
        group_kfold_splits(
            df, target_col=target_col, group_col=GROUP_COL, n_splits=OUTER_N_SPLITS
        )
    )
    _progress(
        f"[Exp3] Prepared {len(outer_splits)} OUTER folds: GroupKFold({GROUP_COL}), n_splits={OUTER_N_SPLITS}"
    )

    outer_results: List[Dict[str, Any]] = []
    outer_best_params: List[Dict[str, Any]] = []

    parent_start = time.perf_counter()

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags(
            {
                "phase": "exp3",
                "design": "nested_cv",
                "cv": "cv1_groupkfold_outer_inner",
                "model": "svr",
            }
        )
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "group_col": GROUP_COL,
                "env_col": ENV_COL,
                "threshold_train_missing_frac_numeric_leq": THRESHOLD,
                "outer_n_splits": OUTER_N_SPLITS,
                "inner_n_splits": INNER_N_SPLITS,
                "n_trials_per_outer_fold": N_TRIALS_PER_OUTER_FOLD,
                "svr_kernel": SVR_KERNEL,
                "objective": "variety_level_rmse_all (minimise)",
                "random_state": RANDOM_STATE,
                "n_rows": int(len(df)),
                "n_numeric_features": int(len(numeric_cols)),
                "n_categorical_features": int(len(categorical_cols)),
            }
        )

        preprocessor = make_preprocessor(
            profile="linear",
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        for outer_fold_i, outer in enumerate(outer_splits, start=1):
            fold_start = time.perf_counter()

            outer_train_idx = outer.train_idx
            outer_test_idx = outer.test_idx

            df_outer_train = df.iloc[outer_train_idx].reset_index(drop=True)
            df_outer_test = df.iloc[outer_test_idx].reset_index(drop=True)

            miss_outer_train = _numeric_missing_frac(df_outer_train, numeric_cols)
            miss_outer_test = _numeric_missing_frac(df_outer_test, numeric_cols)

            y_outer_train = df_outer_train[target_col].to_numpy(dtype=float)
            y_outer_test = df_outer_test[target_col].to_numpy(dtype=float)

            env_outer_test = _upper_str(df_outer_test[ENV_COL]).to_numpy()
            name1_outer_train = df_outer_train[GROUP_COL].to_numpy()
            name1_outer_test = df_outer_test[GROUP_COL].to_numpy()

            inner_splits = list(
                group_kfold_splits(
                    df_outer_train,
                    target_col=target_col,
                    group_col=GROUP_COL,
                    n_splits=INNER_N_SPLITS,
                )
            )

            if PROGRESS and PROGRESS_EVERY_OUTER_FOLD:
                _progress(
                    f"\n[Exp3] OUTER fold {outer_fold_i}/{OUTER_N_SPLITS} | "
                    f"outer_train_rows={len(df_outer_train)} outer_test_rows={len(df_outer_test)} | "
                    f"outer_train_varieties={df_outer_train[GROUP_COL].nunique()} outer_test_varieties={df_outer_test[GROUP_COL].nunique()}"
                )

            with mlflow.start_run(
                run_name=f"outer_fold_{outer_fold_i:02d}", nested=True
            ):
                mlflow.set_tags({"outer_fold": str(outer_fold_i)})
                mlflow.log_params(
                    {
                        "outer_fold": outer_fold_i,
                        "outer_train_rows": int(len(df_outer_train)),
                        "outer_test_rows": int(len(df_outer_test)),
                        "outer_train_varieties": int(
                            df_outer_train[GROUP_COL].nunique()
                        ),
                        "outer_test_varieties": int(df_outer_test[GROUP_COL].nunique()),
                    }
                )

                sampler = TPESampler(seed=RANDOM_STATE + outer_fold_i)

                study = optuna.create_study(direction="minimize", sampler=sampler)

                trial_records: List[Dict[str, Any]] = []

                def inner_objective(trial: optuna.Trial) -> float:
                    C = trial.suggest_float("C", 1e-2, 1e3, log=True)
                    epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)

                    gamma_mode = trial.suggest_categorical(
                        "gamma_mode", ["scale", "auto", "float"]
                    )
                    if gamma_mode == "float":
                        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                    else:
                        gamma = gamma_mode

                    params = {
                        "kernel": SVR_KERNEL,
                        "C": C,
                        "epsilon": epsilon,
                        "gamma": gamma,
                        "cache_size": SVR_CACHE_SIZE,
                    }

                    fold_scores: List[float] = []
                    fold_plot_rmse: List[float] = []
                    fold_n_varieties: List[int] = []

                    for inner in inner_splits:
                        inner_train_idx = inner.train_idx
                        inner_valid_idx = inner.test_idx

                        inner_train_keep = (
                            miss_outer_train[inner_train_idx] <= THRESHOLD
                        )
                        kept_inner_train_idx = inner_train_idx[inner_train_keep]

                        kept_inner_train_idx = kept_inner_train_idx[
                            ~np.isnan(y_outer_train[kept_inner_train_idx])
                        ]

                        y_valid = y_outer_train[inner_valid_idx]
                        y_obs_mask = ~np.isnan(y_valid)

                        if len(kept_inner_train_idx) == 0 or y_obs_mask.sum() == 0:
                            return float("inf")

                        X_tr = df_outer_train.iloc[kept_inner_train_idx][feature_cols]
                        y_tr = y_outer_train[kept_inner_train_idx]
                        X_va = df_outer_train.iloc[inner_valid_idx][feature_cols]

                        pipe = Pipeline(
                            steps=[("pre", preprocessor), ("est", SVR(**params))]
                        )
                        pipe.fit(X_tr, y_tr)

                        y_pred_va = np.asarray(pipe.predict(X_va), dtype=float)

                        rmse_var, n_var = _variety_level_rmse(
                            name1=name1_outer_train[inner_valid_idx],
                            y_true=y_valid,
                            y_pred=y_pred_va,
                            mask=y_obs_mask,
                        )

                        if not np.isfinite(rmse_var) or n_var < 2:
                            return float("inf")

                        rmse_plot = float(
                            root_mean_squared_error(
                                y_valid[y_obs_mask].astype(float),
                                y_pred_va[y_obs_mask].astype(float),
                            )
                        )

                        fold_scores.append(rmse_var)
                        fold_plot_rmse.append(rmse_plot)
                        fold_n_varieties.append(n_var)

                    mean_var_rmse = float(np.mean(fold_scores))
                    mean_plot_rmse = float(np.mean(fold_plot_rmse))
                    mean_nvar = float(np.mean(fold_n_varieties))

                    trial_records.append(
                        {
                            "outer_fold": outer_fold_i,
                            "trial": int(trial.number),
                            "C": C,
                            "epsilon": epsilon,
                            "gamma_mode": gamma_mode,
                            "gamma": gamma if gamma_mode == "float" else np.nan,
                            "inner_variety_rmse_mean": mean_var_rmse,
                            "inner_plot_rmse_mean": mean_plot_rmse,
                            "inner_n_varieties_mean": mean_nvar,
                        }
                    )

                    if PROGRESS and PROGRESS_EVERY_TRIAL:
                        _progress(
                            f"  [outer {outer_fold_i:02d}] trial {trial.number:03d} | "
                            f"inner_var_rmse={mean_var_rmse:.4f} (plot_rmse={mean_plot_rmse:.4f}) | "
                            f"C={C} eps={epsilon} gamma={gamma_mode if gamma_mode!='float' else gamma}"
                        )

                    return mean_var_rmse

                _progress(
                    f"[Exp3] OUTER {outer_fold_i:02d}: running Optuna trials={N_TRIALS_PER_OUTER_FOLD} ..."
                )
                study.optimize(inner_objective, n_trials=N_TRIALS_PER_OUTER_FOLD)

                best_params = dict(study.best_params)
                best_value = float(study.best_value)

                outer_best_params.append(
                    {
                        "outer_fold": outer_fold_i,
                        "best_inner_variety_rmse": best_value,
                        **best_params,
                    }
                )

                if trial_records:
                    trials_df = pd.DataFrame(trial_records).sort_values(
                        "inner_variety_rmse_mean"
                    )
                    log_df_artifact(
                        trials_df, f"exp3_inner_trials__outer{outer_fold_i:02d}.csv"
                    )

                mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
                mlflow.log_metric("best_inner_variety_rmse", best_value)

                outer_train_keep = miss_outer_train <= THRESHOLD
                kept_outer_train_idx = np.where(outer_train_keep)[0]

                kept_outer_train_idx = kept_outer_train_idx[
                    ~np.isnan(y_outer_train[kept_outer_train_idx])
                ]

                if len(kept_outer_train_idx) == 0:
                    raise RuntimeError(
                        f"Outer fold {outer_fold_i}: no training rows left after threshold filter."
                    )

                X_outer_tr = df_outer_train.iloc[kept_outer_train_idx][feature_cols]
                y_outer_tr = y_outer_train[kept_outer_train_idx]
                X_outer_te = df_outer_test[feature_cols]

                svr_params = {
                    "kernel": SVR_KERNEL,
                    "C": float(best_params["C"]),
                    "epsilon": float(best_params["epsilon"]),
                    "cache_size": SVR_CACHE_SIZE,
                }
                gamma_mode = best_params["gamma_mode"]
                if gamma_mode in ("scale", "auto"):
                    svr_params["gamma"] = gamma_mode
                else:
                    svr_params["gamma"] = float(best_params["gamma"])

                final_pipe = Pipeline(
                    steps=[("pre", preprocessor), ("est", SVR(**svr_params))]
                )
                final_pipe.fit(X_outer_tr, y_outer_tr)

                y_pred_outer = np.asarray(final_pipe.predict(X_outer_te), dtype=float)
                y_obs_mask_outer = ~np.isnan(y_outer_test)

                plot_metrics = compute_metrics(
                    y_outer_test[y_obs_mask_outer].astype(float),
                    y_pred_outer[y_obs_mask_outer].astype(float),
                )

                outer_var_rmse_all, outer_nvar_all = _variety_level_rmse(
                    name1=name1_outer_test,
                    y_true=y_outer_test,
                    y_pred=y_pred_outer,
                    mask=y_obs_mask_outer,
                )

                outer_var_spear_all, _ = _variety_level_spearman(
                    name1=name1_outer_test,
                    y_true=y_outer_test,
                    y_pred=y_pred_outer,
                    mask=y_obs_mask_outer,
                )

                is_ne = env_outer_test == "NE"
                is_med = env_outer_test == "MED"

                outer_var_rmse_ne, outer_nvar_ne = _variety_level_rmse(
                    name1=name1_outer_test,
                    y_true=y_outer_test,
                    y_pred=y_pred_outer,
                    mask=y_obs_mask_outer & is_ne,
                )
                outer_var_rmse_med, outer_nvar_med = _variety_level_rmse(
                    name1=name1_outer_test,
                    y_true=y_outer_test,
                    y_pred=y_pred_outer,
                    mask=y_obs_mask_outer & is_med,
                )

                is_complete = miss_outer_test == 0.0
                is_medium = (miss_outer_test > 0.0) & (
                    miss_outer_test <= MEDIUM_MAX_MISSING_FRAC
                )
                is_incomplete = miss_outer_test > MEDIUM_MAX_MISSING_FRAC

                rmse_complete = float("nan")
                rmse_medium = float("nan")
                rmse_incomplete = float("nan")

                if np.sum(y_obs_mask_outer & is_complete) > 0:
                    rmse_complete = float(
                        root_mean_squared_error(
                            y_outer_test[y_obs_mask_outer & is_complete].astype(float),
                            y_pred_outer[y_obs_mask_outer & is_complete].astype(float),
                        )
                    )
                if np.sum(y_obs_mask_outer & is_medium) > 0:
                    rmse_medium = float(
                        root_mean_squared_error(
                            y_outer_test[y_obs_mask_outer & is_medium].astype(float),
                            y_pred_outer[y_obs_mask_outer & is_medium].astype(float),
                        )
                    )
                if np.sum(y_obs_mask_outer & is_incomplete) > 0:
                    rmse_incomplete = float(
                        root_mean_squared_error(
                            y_outer_test[y_obs_mask_outer & is_incomplete].astype(
                                float
                            ),
                            y_pred_outer[y_obs_mask_outer & is_incomplete].astype(
                                float
                            ),
                        )
                    )

                mlflow.log_metrics(
                    {
                        "outer_plot_rmse_all": float(plot_metrics.rmse),
                        "outer_plot_mae_all": float(plot_metrics.mae),
                        "outer_plot_r2_all": float(plot_metrics.r2),
                        "outer_plot_pearson_r_all": float(plot_metrics.pearson_r),
                        "outer_variety_rmse_all": float(outer_var_rmse_all),
                        "outer_variety_spearman_all": float(outer_var_spear_all),
                        "outer_n_varieties_all": float(outer_nvar_all),
                        "outer_variety_rmse_ne": float(outer_var_rmse_ne),
                        "outer_variety_rmse_med": float(outer_var_rmse_med),
                        "outer_n_varieties_ne": float(outer_nvar_ne),
                        "outer_n_varieties_med": float(outer_nvar_med),
                        "outer_rmse_complete": float(rmse_complete),
                        "outer_rmse_medium": float(rmse_medium),
                        "outer_rmse_incomplete": float(rmse_incomplete),
                    }
                )

                outer_results.append(
                    {
                        "outer_fold": outer_fold_i,
                        "outer_test_rows": int(len(df_outer_test)),
                        "outer_test_scored_rows": int(np.sum(y_obs_mask_outer)),
                        "outer_test_varieties": int(df_outer_test[GROUP_COL].nunique()),
                        "best_inner_variety_rmse": best_value,
                        "outer_plot_rmse_all": float(plot_metrics.rmse),
                        "outer_plot_mae_all": float(plot_metrics.mae),
                        "outer_plot_r2_all": float(plot_metrics.r2),
                        "outer_plot_pearson_r_all": float(plot_metrics.pearson_r),
                        "outer_variety_rmse_all": float(outer_var_rmse_all),
                        "outer_variety_spearman_all": float(outer_var_spear_all),
                        "outer_n_varieties_all": int(outer_nvar_all),
                        "outer_variety_rmse_ne": float(outer_var_rmse_ne),
                        "outer_variety_rmse_med": float(outer_var_rmse_med),
                        "outer_n_varieties_ne": int(outer_nvar_ne),
                        "outer_n_varieties_med": int(outer_nvar_med),
                        "outer_rmse_complete": float(rmse_complete),
                        "outer_rmse_medium": float(rmse_medium),
                        "outer_rmse_incomplete": float(rmse_incomplete),
                        "refit_train_rows_used": int(len(kept_outer_train_idx)),
                        "refit_train_kept_frac": (
                            float(len(kept_outer_train_idx) / len(df_outer_train))
                            if len(df_outer_train)
                            else float("nan")
                        ),
                        "wall_time_sec_outer_fold": float(
                            time.perf_counter() - fold_start
                        ),
                    }
                )

            _progress(
                f"[Exp3] OUTER fold {outer_fold_i}/{OUTER_N_SPLITS} completed in {_fmt_seconds(time.perf_counter() - fold_start)}"
            )

        outer_results_df = pd.DataFrame(outer_results).sort_values("outer_fold")
        outer_best_params_df = pd.DataFrame(outer_best_params).sort_values("outer_fold")

        log_df_artifact(outer_results_df, "exp3_outer_fold_results.csv")
        log_df_artifact(outer_best_params_df, "exp3_outer_fold_best_params.csv")

        rmse_mean, rmse_std, rmse_se, n_valid = mean_std_se(
            outer_results_df["outer_variety_rmse_all"]
        )
        plot_rmse_mean, plot_rmse_std, plot_rmse_se, _ = mean_std_se(
            outer_results_df["outer_plot_rmse_all"]
        )

        summary = {
            "primary_metric": "outer_variety_rmse_all",
            "outer_variety_rmse_all_mean": float(rmse_mean),
            "outer_variety_rmse_all_std": float(rmse_std),
            "outer_variety_rmse_all_se": float(rmse_se),
            "outer_variety_rmse_all_n_folds_valid": int(n_valid),
            "outer_plot_rmse_all_mean": float(plot_rmse_mean),
            "outer_plot_rmse_all_std": float(plot_rmse_std),
            "outer_plot_rmse_all_se": float(plot_rmse_se),
            "total_wall_time_sec": float(time.perf_counter() - parent_start),
        }

        log_dict_artifact(summary, "exp3_nestedcv_summary.json")
        mlflow.log_metrics(
            {
                "outer_variety_rmse_all_mean": float(rmse_mean),
                "outer_variety_rmse_all_std": float(rmse_std),
                "outer_variety_rmse_all_se": float(rmse_se),
                "outer_plot_rmse_all_mean": float(plot_rmse_mean),
            }
        )

        _progress("\n[Exp3] NESTED CV COMPLETE")
        _progress(
            f"  Reportable (unbiased) metric: outer_variety_rmse_all_mean = {rmse_mean:.4f} (std={rmse_std:.4f})"
        )
        _progress(
            f"  Plot RMSE (context): outer_plot_rmse_all_mean = {plot_rmse_mean:.4f}"
        )
        _progress(f"  Total time: {_fmt_seconds(time.perf_counter() - parent_start)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
