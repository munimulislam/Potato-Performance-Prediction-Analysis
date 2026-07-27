"""
@File - imputation_threshold.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 26/07/2026
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact, log_dict_artifact
from ml.common.splits import group_kfold_splits
from ml.common.preprocess import make_preprocessor
from ml.common.models import make_model
from ml.common.metrics import compute_metrics, mean_std_se

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "oa/exp1_missingness_threshold_cv1"
PARENT_RUN_NAME = "exp1_missingness_threshold_cv1"

GROUP_COL = "name1"
N_SPLITS = 5

ENV_COL = "env_type"

MEDIUM_MAX_MISSING_FRAC = 0.5
THRESHOLDS = [i / 10 for i in range(1, 11)]
RANDOM_STATE = 42

THRESHOLD_SELECTION_STRATEGY = "avg_models_rmse"

MODELS = [
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

LOG_FOLD_PREDICTIONS = True


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


def run_exp1(
    *,
    df: pd.DataFrame,
    dataset_path: str,
    target_col: str,
    key_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:

    for col in [target_col, ENV_COL, GROUP_COL]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not present in dataset.")

    feature_cols = list(numeric_cols) + list(categorical_cols)

    required_cols = sorted(
        set([target_col, ENV_COL, GROUP_COL] + key_cols + feature_cols)
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required_cols].copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols:
        df[c] = df[c].astype("string")
    df[ENV_COL] = df[ENV_COL].astype("string")
    df[GROUP_COL] = df[GROUP_COL].astype("string")
    for c in key_cols:
        df[c] = df[c].astype("string")

    if df[target_col].isna().all():
        raise ValueError(f"Target '{target_col}' is all-null after coercion.")

    y_all = df[target_col].to_numpy(dtype=float)
    env_upper_all = _upper_str(df[ENV_COL]).to_numpy()
    miss_frac_all = _numeric_missing_frac(df, numeric_cols)
    row_id_all = _build_row_id(df, key_cols)

    splits = list(
        group_kfold_splits(
            df, target_col=target_col, group_col=GROUP_COL, n_splits=N_SPLITS
        )
    )

    model_threshold_rows: List[Dict[str, Any]] = []

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags(
            {
                "phase": "exp1",
                "cv": "cv1_groupkfold",
                "group_col": GROUP_COL,
            }
        )
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "env_col": ENV_COL,
                "group_col": GROUP_COL,
                "n_splits": N_SPLITS,
                "thresholds": str(THRESHOLDS),
                "medium_max_missing_frac": float(MEDIUM_MAX_MISSING_FRAC),
                "threshold_selection_strategy": THRESHOLD_SELECTION_STRATEGY,
                "n_rows": int(len(df)),
                "n_numeric_features": int(len(numeric_cols)),
                "n_categorical_features": int(len(categorical_cols)),
            }
        )

        for m_cfg in MODELS:
            model_name = str(m_cfg["name"]).lower().strip()
            params = dict(m_cfg.get("params", {}) or {})

            model = make_model(model_name, params)
            profile = model.profile

            pre = make_preprocessor(
                profile=profile,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            )

            for thr in THRESHOLDS:
                thr = float(thr)
                nested_run_name = f"{model_name}/thr={thr}"

                with mlflow.start_run(run_name=nested_run_name, nested=True):
                    mlflow.set_tags({"model": model_name, "profile": profile})
                    mlflow.log_params(
                        {
                            "model": model_name,
                            "profile": profile,
                            "threshold": thr,
                            **{f"model__{k}": v for k, v in model.params.items()},
                        }
                    )

                    pipe = Pipeline(steps=[("pre", pre), ("est", model.estimator)])

                    fold_rows: List[Dict[str, Any]] = []
                    pred_parts: List[pd.DataFrame] = []

                    for sp in splits:
                        train_idx = sp.train_idx
                        test_idx = sp.test_idx

                        train_keep = miss_frac_all[train_idx] <= thr
                        kept_train_idx = train_idx[train_keep]

                        kept_train_idx = kept_train_idx[
                            ~np.isnan(y_all[kept_train_idx])
                        ]

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
                            fold_rows.append(fold_row)
                            continue

                        X_train = df.iloc[kept_train_idx][feature_cols]
                        y_train = y_all[kept_train_idx]
                        X_test = df.iloc[test_idx][feature_cols]

                        pipe.fit(X_train, y_train)
                        y_pred_test = np.asarray(pipe.predict(X_test), dtype=float)

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
                            mm = _metrics_for_mask(
                                y_true=y_test, y_pred=y_pred_test, mask=mask
                            )
                            fold_row.update(
                                {
                                    f"rmse_{subset}": mm["rmse"],
                                    f"mae_{subset}": mm["mae"],
                                    f"pearson_r_{subset}": mm["pearson_r"],
                                    f"r2_{subset}": mm["r2"],
                                    f"n_{subset}": mm["n"],
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
                                    "group": df.iloc[test_idx][GROUP_COL]
                                    .astype("string")
                                    .to_numpy(),
                                    "env_type": env_upper_all[test_idx],
                                    "y_true": y_test,
                                    "y_pred": y_pred_test,
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

                    summary = {
                        "rmse_all_mean": float(rmse_all_mean),
                        "rmse_all_std": float(rmse_all_std),
                        "rmse_all_se": float(rmse_all_se),
                        "rmse_all_n_folds_valid": float(n_valid),
                        "train_kept_frac_mean": float(
                            fold_metrics_df["train_kept_frac"].mean(skipna=True)
                        ),
                        "test_scored_n_mean": float(
                            fold_metrics_df["test_scored_n"].mean(skipna=True)
                        ),
                        "rmse_ne_mean": _mean("rmse_ne"),
                        "rmse_med_mean": _mean("rmse_med"),
                        "mae_ne_mean": _mean("mae_ne"),
                        "mae_med_mean": _mean("mae_med"),
                        "rmse_complete_mean": _mean("rmse_complete"),
                        "rmse_medium_mean": _mean("rmse_medium"),
                        "rmse_incomplete_mean": _mean("rmse_incomplete"),
                        "mae_complete_mean": _mean("mae_complete"),
                        "mae_medium_mean": _mean("mae_medium"),
                        "mae_incomplete_mean": _mean("mae_incomplete"),
                        "n_ne_mean": _mean("n_ne"),
                        "n_med_mean": _mean("n_med"),
                        "n_complete_mean": _mean("n_complete"),
                        "n_medium_mean": _mean("n_medium"),
                        "n_incomplete_mean": _mean("n_incomplete"),
                    }

                    mlflow.log_metrics(summary)

                    log_df_artifact(
                        fold_metrics_df,
                        f"fold_metrics__cv1__{model_name}__thr{thr}.csv",
                    )

                    if LOG_FOLD_PREDICTIONS:
                        fold_pred_df = (
                            pd.concat(pred_parts, ignore_index=True)
                            if pred_parts
                            else pd.DataFrame()
                        )
                        if not fold_pred_df.empty:
                            log_df_artifact(
                                fold_pred_df,
                                f"fold_predictions__cv1__{model_name}__thr{thr}.csv",
                            )

                    model_threshold_rows.append(
                        {
                            "model": model_name,
                            "profile": profile,
                            "threshold": thr,
                            **summary,
                        }
                    )

        model_threshold_summary = pd.DataFrame(model_threshold_rows).sort_values(
            ["model", "threshold"]
        )

        log_df_artifact(model_threshold_summary, "exp1_model_threshold_summary.csv")

        thr_rows: List[Dict[str, Any]] = []

        for thr in THRESHOLDS:
            thr = float(thr)
            sub = model_threshold_summary[model_threshold_summary["threshold"] == thr]
            if sub.empty:
                continue

            avg_rmse = float(sub["rmse_all_mean"].mean(skipna=True))
            best_rmse = float(sub["rmse_all_mean"].min(skipna=True))
            best_model = str(sub.loc[sub["rmse_all_mean"].idxmin(), "model"])

            thr_rows.append(
                {
                    "threshold": thr,
                    "rmse_all_mean_avg_models": avg_rmse,
                    "rmse_all_mean_best_model": best_rmse,
                    "best_model_at_threshold": best_model,
                    "avg_train_kept_frac_mean": float(
                        sub["train_kept_frac_mean"].mean(skipna=True)
                    ),
                }
            )

        threshold_summary = pd.DataFrame(thr_rows).sort_values("threshold")
        log_df_artifact(threshold_summary, "exp1_threshold_summary.csv")

        if THRESHOLD_SELECTION_STRATEGY == "avg_models_rmse":
            score_col = "rmse_all_mean_avg_models"
        elif THRESHOLD_SELECTION_STRATEGY == "best_model_rmse":
            score_col = "rmse_all_mean_best_model"
        else:
            raise ValueError(
                f"Unknown THRESHOLD_SELECTION_STRATEGY: {THRESHOLD_SELECTION_STRATEGY}"
            )

        best_score = float(threshold_summary[score_col].min())
        best_candidates = threshold_summary[threshold_summary[score_col] == best_score]
        selected_threshold = float(best_candidates["threshold"].max())

        log_dict_artifact(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "group_col": GROUP_COL,
                "env_col": ENV_COL,
                "threshold_selection_strategy": THRESHOLD_SELECTION_STRATEGY,
                "score_col": score_col,
                "best_score": best_score,
                "selected_threshold": selected_threshold,
            },
            "exp1_selected_threshold.json",
        )

    return {
        "model_threshold_summary": model_threshold_summary,
        "threshold_summary": threshold_summary,
        "selected_threshold": selected_threshold,
    }


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

    res = run_exp1(
        df=df,
        dataset_path=dataset_path,
        target_col=target_col,
        key_cols=key_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    print("Exp 1 complete.")
    print(
        f"Selected threshold ({THRESHOLD_SELECTION_STRATEGY}): {res['selected_threshold']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
