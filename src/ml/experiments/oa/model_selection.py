"""
@File - model_selection.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 27/07/2026
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn.pipeline import Pipeline

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact, log_dict_artifact
from ml.common.splits import group_kfold_splits
from ml.common.preprocess import make_preprocessor
from ml.common.models import make_model
from ml.common.metrics import compute_metrics, mean_std_se

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"oa/exp2_model_selection_{datetime.now()}"
PARENT_RUN_NAME = "exp2_model_selection"

GROUP_COL = "name1"
N_SPLITS = 5
ENV_COL = "env_type"
THRESHOLD = 0.5
MEDIUM_MAX_MISSING_FRAC = 0.5
RANDOM_STATE = 42
LOG_FOLD_PREDICTIONS = True

MODEL_SPECS: List[Dict[str, Any]] = [
    {"name": "baseline_mean", "params": {}},
    {"name": "baseline_env_mean", "params": {}},
    {"name": "ridge", "params": {"alpha": 1.0}},
    {
        "name": "rf",
        "params": {"n_estimators": 500, "n_jobs": -1, "random_state": RANDOM_STATE},
    },
    {"name": "knn", "params": {"n_neighbors": 5, "weights": "uniform"}},
    {"name": "svr", "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}},
    {
        "name": "xgb",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        },
    },
    {
        "name": "lgbm",
        "params": {
            "n_estimators": 5000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
    },
]


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

    df = pd.DataFrame(
        {
            "name1": name1[mask],
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
        }
    ).dropna(subset=["y_true", "y_pred", "name1"])

    if df.empty:
        return float("nan"), 0

    agg = df.groupby("name1", as_index=False).agg(
        y_true=("y_true", "mean"),
        y_pred=("y_pred", "mean"),
    )

    n_var = int(len(agg))
    if n_var < 2:
        return float("nan"), n_var

    # pandas Spearman
    rho = agg["y_true"].corr(agg["y_pred"], method="spearman")
    return float(rho), n_var


def run_exp2_best_model_cv1(
    *,
    df: pd.DataFrame,
    dataset_path: str,
    target_col: str,
    key_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    # Required columns
    for c in (target_col, GROUP_COL, ENV_COL):
        if c not in df.columns:
            raise ValueError(f"Required column missing from dataset: '{c}'")

    feature_cols = list(numeric_cols) + list(categorical_cols)

    required_cols = sorted(
        set([target_col, GROUP_COL, ENV_COL] + key_cols + feature_cols)
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

    y_all = df[target_col].to_numpy(dtype=float)
    env_upper_all = _upper_str(df[ENV_COL]).to_numpy()
    name1_all = df[GROUP_COL].to_numpy()
    miss_frac_all = _numeric_missing_frac(df, numeric_cols=numeric_cols)
    row_id_all = _build_row_id(df, key_cols=key_cols)

    splits = list(
        group_kfold_splits(
            df, target_col=target_col, group_col=GROUP_COL, n_splits=N_SPLITS
        )
    )

    # Parent run
    model_summary_rows: List[Dict[str, Any]] = []

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags(
            {"phase": "exp2", "cv": "cv1_groupkfold", "group_col": GROUP_COL}
        )
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "env_col": ENV_COL,
                "group_col": GROUP_COL,
                "n_splits": N_SPLITS,
                "threshold": THRESHOLD,
                "medium_max_missing_frac": MEDIUM_MAX_MISSING_FRAC,
                "n_rows": int(len(df)),
                "n_numeric_features": int(len(numeric_cols)),
                "n_categorical_features": int(len(categorical_cols)),
                "models": str([m["name"] for m in MODEL_SPECS]),
            }
        )

        for m_cfg in MODEL_SPECS:
            model_name = str(m_cfg["name"]).lower().strip()
            params = dict(m_cfg.get("params", {}) or {})

            is_baseline = model_name in ("baseline_mean", "baseline_env_mean")

            if not is_baseline:
                model = make_model(model_name, params)
                profile = model.profile
                pre = make_preprocessor(
                    profile=profile,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )
                pipe = Pipeline(steps=[("pre", pre), ("est", model.estimator)])
            else:
                profile = "baseline"
                model = None
                pipe = None

            with mlflow.start_run(run_name=f"{model_name}", nested=True):
                mlflow.set_tags({"model": model_name, "profile": profile})
                mlflow.log_params(
                    {
                        "model": model_name,
                        "profile": profile,
                        **{f"model__{k}": v for k, v in (params or {}).items()},
                    }
                )

                fold_rows: List[Dict[str, Any]] = []
                pred_parts: List[pd.DataFrame] = []

                for sp in splits:
                    train_idx = sp.train_idx
                    test_idx = sp.test_idx

                    train_keep = miss_frac_all[train_idx] <= THRESHOLD
                    kept_train_idx = train_idx[train_keep]
                    kept_train_idx = kept_train_idx[~np.isnan(y_all[kept_train_idx])]

                    y_test = y_all[test_idx]
                    y_obs_mask = ~np.isnan(y_test)

                    fold_row: Dict[str, Any] = {
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
                        for subset in (
                            "all",
                            "ne",
                            "med",
                            "complete",
                            "medium",
                            "incomplete",
                        ):
                            fold_row.update(
                                {
                                    f"rmse_{subset}": float("nan"),
                                    f"mae_{subset}": float("nan"),
                                    f"pearson_r_{subset}": float("nan"),
                                    f"r2_{subset}": float("nan"),
                                    f"n_{subset}": 0.0,
                                }
                            )
                        for subset in ("all", "ne", "med"):
                            fold_row.update(
                                {
                                    f"spearman_variety_{subset}": float("nan"),
                                    f"n_varieties_{subset}": 0.0,
                                }
                            )
                        fold_rows.append(fold_row)
                        continue

                    if is_baseline:
                        y_train = y_all[kept_train_idx]
                        global_mean = float(np.nanmean(y_train))

                        if model_name == "baseline_mean":
                            y_pred = np.full(
                                shape=len(test_idx), fill_value=global_mean, dtype=float
                            )

                        elif model_name == "baseline_env_mean":
                            env_train = env_upper_all[kept_train_idx]
                            df_train = pd.DataFrame({"env": env_train, "y": y_train})
                            env_means = (
                                df_train.groupby("env", as_index=True)["y"]
                                .mean()
                                .to_dict()
                            )

                            env_test = env_upper_all[test_idx]
                            y_pred = np.array(
                                [env_means.get(e, global_mean) for e in env_test],
                                dtype=float,
                            )

                        else:
                            raise ValueError(model_name)

                    else:
                        X_train = df.iloc[kept_train_idx][feature_cols]
                        y_train = y_all[kept_train_idx]
                        X_test = df.iloc[test_idx][feature_cols]

                        pipe.fit(X_train, y_train)
                        y_pred = np.asarray(pipe.predict(X_test), dtype=float)

                    env_test = env_upper_all[test_idx]
                    miss_test = miss_frac_all[test_idx]

                    masks: Dict[str, np.ndarray] = {
                        "all": y_obs_mask,
                        "ne": y_obs_mask & (env_test == "NE"),
                        "med": y_obs_mask & (env_test == "MED"),
                        "complete": y_obs_mask
                        & _mask_missing_band(
                            miss_test, "complete", MEDIUM_MAX_MISSING_FRAC
                        ),
                        "medium": y_obs_mask
                        & _mask_missing_band(
                            miss_test, "medium", MEDIUM_MAX_MISSING_FRAC
                        ),
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
                                f"pearson_r_{subset}": mm["pearson_r"],
                                f"r2_{subset}": mm["r2"],
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

                    if LOG_FOLD_PREDICTIONS:
                        missing_band = np.where(
                            miss_test == 0.0,
                            "complete",
                            np.where(
                                miss_test <= MEDIUM_MAX_MISSING_FRAC,
                                "medium",
                                "incomplete",
                            ),
                        )
                        pred_df = pd.DataFrame(
                            {
                                "fold": int(sp.fold),
                                "row_id": row_id_all[test_idx],
                                "name1": name1_all[test_idx],
                                "env_type": env_test,
                                "y_true": y_test,
                                "y_pred": y_pred,
                                "y_observed": y_obs_mask,
                                "x_missing_frac": miss_test,
                                "missing_band": missing_band,
                            }
                        )
                        pred_parts.append(pred_df)

                fold_metrics_df = pd.DataFrame(fold_rows)

                rmse_all_mean, rmse_all_std, rmse_all_se, n_valid = mean_std_se(
                    fold_metrics_df["rmse_all"]
                )

                def _mean(col: str) -> float:
                    return float(fold_metrics_df[col].mean(skipna=True))

                model_summary = {
                    "model": model_name,
                    "profile": profile,
                    "rmse_all_mean": float(rmse_all_mean),
                    "rmse_all_std": float(rmse_all_std),
                    "rmse_all_se": float(rmse_all_se),
                    "rmse_all_n_folds_valid": float(n_valid),
                    "rmse_ne_mean": _mean("rmse_ne"),
                    "rmse_med_mean": _mean("rmse_med"),
                    "spearman_variety_all_mean": _mean("spearman_variety_all"),
                    "spearman_variety_ne_mean": _mean("spearman_variety_ne"),
                    "spearman_variety_med_mean": _mean("spearman_variety_med"),
                    "train_kept_frac_mean": float(
                        fold_metrics_df["train_kept_frac"].mean(skipna=True)
                    ),
                    "test_scored_n_mean": float(
                        fold_metrics_df["test_scored_n"].mean(skipna=True)
                    ),
                    "n_varieties_all_mean": _mean("n_varieties_all"),
                    "n_varieties_ne_mean": _mean("n_varieties_ne"),
                    "n_varieties_med_mean": _mean("n_varieties_med"),
                    "rmse_complete_mean": _mean("rmse_complete"),
                    "rmse_medium_mean": _mean("rmse_medium"),
                    "rmse_incomplete_mean": _mean("rmse_incomplete"),
                    "r2_all_mean": _mean("r2_all"),
                    "r2_ne_mean": _mean("r2_ne"),
                    "r2_med_mean": _mean("r2_med"),
                    "r2_complete_mean": _mean("r2_complete"),
                    "r2_medium_mean": _mean("r2_medium"),
                    "r2_incomplete_mean": _mean("r2_incomplete"),
                    "r2_all_mean": _mean("r2_all"),
                    "mae_ne_mean": _mean("mae_ne"),
                    "mae_med_mean": _mean("mae_med"),
                    "mae_complete_mean": _mean("mae_complete"),
                    "mae_medium_mean": _mean("mae_medium"),
                    "mae_incomplete_mean": _mean("mae_incomplete"),
                    "mae_ne_mean": _mean("mae_ne"),
                    "pearson_r_med_mean": _mean("pearson_r_med"),
                    "pearson_r_complete_mean": _mean("pearson_r_complete"),
                    "pearson_r_medium_mean": _mean("pearson_r_medium"),
                    "pearson_r_incomplete_mean": _mean("pearson_r_incomplete"),
                }

                mlflow.log_metrics(
                    {
                        k: float(v)
                        for k, v in model_summary.items()
                        if k not in ("model", "profile")
                    }
                )

                log_df_artifact(fold_metrics_df, f"exp2_fold_metrics__{model_name}.csv")

                if LOG_FOLD_PREDICTIONS:
                    fold_pred_df = (
                        pd.concat(pred_parts, ignore_index=True)
                        if pred_parts
                        else pd.DataFrame()
                    )
                    if not fold_pred_df.empty:
                        log_df_artifact(
                            fold_pred_df, f"exp2_fold_predictions__{model_name}.csv"
                        )

                model_summary_rows.append(model_summary)

        model_summary_df = pd.DataFrame(model_summary_rows).sort_values(
            ["rmse_all_mean", "spearman_variety_all_mean"],
            ascending=[True, False],
        )
        log_df_artifact(model_summary_df, "exp2_model_summary.csv")

        best_row = model_summary_df.iloc[0].to_dict()
        log_dict_artifact(
            {
                "selection_rule": "min rmse_all_mean, tie-break max spearman_variety_all_mean",
                "threshold": THRESHOLD,
                "selected_model": {
                    "model": best_row["model"],
                    "profile": best_row["profile"],
                    "rmse_all_mean": best_row["rmse_all_mean"],
                    "rmse_ne_mean": best_row["rmse_ne_mean"],
                    "rmse_med_mean": best_row["rmse_med_mean"],
                    "spearman_variety_all_mean": best_row["spearman_variety_all_mean"],
                    "spearman_variety_ne_mean": best_row["spearman_variety_ne_mean"],
                    "spearman_variety_med_mean": best_row["spearman_variety_med_mean"],
                    "rmse_complete_mean": best_row["rmse_complete_mean"],
                    "rmse_medium_mean": best_row["rmse_medium_mean"],
                    "rmse_incomplete_mean": best_row["rmse_incomplete_mean"],
                    "r2_all_mean": best_row["r2_all_mean"],
                    "r2_ne_mean": best_row["r2_ne_mean"],
                    "r2_med_mean": best_row["r2_med_mean"],
                    "r2_complete_mean": best_row["r2_complete_mean"],
                    "r2_medium_mean": best_row["r2_medium_mean"],
                    "r2_incomplete_mean": best_row["r2_incomplete_mean"],
                    "r2_all_mean": best_row["r2_all_mean"],
                    "mae_ne_mean": best_row["mae_ne_mean"],
                    "mae_med_mean": best_row["mae_med_mean"],
                    "mae_complete_mean": best_row["mae_complete_mean"],
                    "mae_medium_mean": best_row["mae_medium_mean"],
                    "mae_incomplete_mean": best_row["mae_incomplete_mean"],
                    "mae_ne_mean": best_row["mae_ne_mean"],
                    "pearson_r_med_mean": best_row["pearson_r_med_mean"],
                    "pearson_r_complete_mean": best_row["pearson_r_complete_mean"],
                    "pearson_r_medium_mean": best_row["pearson_r_medium_mean"],
                    "pearson_r_incomplete_mean": best_row["pearson_r_incomplete_mean"],
                },
            },
            "exp2_selected_model.json",
        )

    return model_summary_df


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
    df = load_dataframe(dataset_path)

    model_summary_df = run_exp2_best_model_cv1(
        df=df,
        dataset_path=dataset_path,
        target_col=target_col,
        key_cols=key_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    print("Exp 2 complete.")
    print("\nTop 10 models by (rmse_all_mean asc, spearman_variety_all_mean desc):")
    print(model_summary_df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
