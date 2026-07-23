"""
@File - runner.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass
from typing import Iterable

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .metrics import Metrics, compute_metrics
from .models import make_model
from .preprocess import (
    apply_train_threshold,
    build_preprocessor,
    usable_numeric_columns,
)
from .splits import Split, group_kfold_splits


@dataclass(frozen=True)
class FoldOutcome:
    fold: int
    n_train: int
    n_train_kept: int
    n_test: int
    metrics: Metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def run_cv1_sweep(
    *,
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    group_col: str,
    n_splits: int,
    thresholds: list[float],
    model_names: list[str],
    random_state: int,
) -> pd.DataFrame:

    required = set([target_col, group_col] + numeric_cols + categorical_cols)
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    y_all = pd.to_numeric(df[target_col], errors="raise")
    results_rows: list[dict] = []
    splits = list(group_kfold_splits(df, group_col=group_col, n_splits=n_splits))

    for model_name in model_names:
        model_spec = make_model(model_name, random_state=random_state)

        for threshold in thresholds:
            fold_outcomes: list[FoldOutcome] = []
            failed_reason = None

            for s in splits:
                train_df = df.iloc[s.train_idx].copy()
                test_df = df.iloc[s.test_idx].copy()

                tr = apply_train_threshold(
                    train_df, numeric_cols=numeric_cols, threshold=threshold
                )
                if tr.n_train_kept == 0:
                    failed_reason = "TRAIN_EMPTY_AFTER_THRESHOLD"
                    break

                train_kept_df = train_df.iloc[np.where(tr.train_kept_mask)[0]].copy()

                usable_num = usable_numeric_columns(train_kept_df, numeric_cols)
                if not usable_num:
                    failed_reason = "NO_USABLE_NUMERIC_FEATURES"
                    break

                pre = build_preprocessor(
                    numeric_cols=usable_num, categorical_cols=categorical_cols
                )
                pipe = Pipeline(steps=[("pre", pre), ("model", model_spec.estimator)])

                X_train = train_kept_df[usable_num + categorical_cols]
                y_train = pd.to_numeric(
                    train_kept_df[target_col], errors="raise"
                ).to_numpy()

                X_test = test_df[usable_num + categorical_cols]
                y_test = pd.to_numeric(test_df[target_col], errors="raise").to_numpy()

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                m = compute_metrics(y_test, y_pred)
                fold_outcomes.append(
                    FoldOutcome(
                        fold=s.fold,
                        n_train=tr.n_train,
                        n_train_kept=tr.n_train_kept,
                        n_test=len(test_df),
                        metrics=m,
                    )
                )

            row = {
                "model": model_spec.name,
                "threshold": float(threshold),
                "status": "OK" if failed_reason is None else "FAILED",
                "failed_reason": failed_reason,
                "n_folds": n_splits,
                "n_folds_completed": len(fold_outcomes),
            }

            if fold_outcomes:
                row["train_kept_frac_mean"] = float(
                    np.mean(
                        [
                            fo.n_train_kept / fo.n_train
                            for fo in fold_outcomes
                            if fo.n_train > 0
                        ]
                    )
                )
                row["n_train_kept_mean"] = float(
                    np.mean([fo.n_train_kept for fo in fold_outcomes])
                )
                row["n_test_mean"] = float(np.mean([fo.n_test for fo in fold_outcomes]))

                rmse_mean, rmse_std = mean_std(
                    [fo.metrics.rmse for fo in fold_outcomes]
                )
                mae_mean, mae_std = mean_std([fo.metrics.mae for fo in fold_outcomes])
                r2_mean, r2_std = mean_std([fo.metrics.r2 for fo in fold_outcomes])
                bias_mean, bias_std = mean_std(
                    [fo.metrics.bias_mean for fo in fold_outcomes]
                )
                acc05_mean, acc05_std = mean_std(
                    [fo.metrics.acc_within_0_5 for fo in fold_outcomes]
                )
                acc10_mean, acc10_std = mean_std(
                    [fo.metrics.acc_within_1_0 for fo in fold_outcomes]
                )
                sp_mean, sp_std = mean_std(
                    [fo.metrics.spearman_rho for fo in fold_outcomes]
                )

                row.update(
                    {
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "mae_std": mae_std,
                        "r2_mean": r2_mean,
                        "r2_std": r2_std,
                        "bias_mean": bias_mean,
                        "bias_std": bias_std,
                        "acc05_mean": acc05_mean,
                        "acc05_std": acc05_std,
                        "acc10_mean": acc10_mean,
                        "acc10_std": acc10_std,
                        "spearman_mean": sp_mean,
                        "spearman_std": sp_std,
                    }
                )

            results_rows.append(row)

    return pd.DataFrame(results_rows)


def log_sweep_to_mlflow(
    *,
    summary_df: pd.DataFrame,
    scenario: str,
    cv_type: str,
    split_axis: str,
    group_col: str,
    n_splits: int,
) -> None:

    mlflow.set_tag("scenario", scenario)
    mlflow.set_tag("cv_type", cv_type)
    mlflow.set_tag("split_axis", split_axis)
    mlflow.set_tag("group_col", group_col)
    mlflow.log_param("n_splits", n_splits)

    for _, r in summary_df.iterrows():
        run_name = f"{r['model']}__thr={r['threshold']}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("model_type", str(r["model"]))
            mlflow.log_param("threshold", float(r["threshold"]))
            mlflow.log_param("status", str(r.get("status", "")))
            if pd.notna(r.get("failed_reason")):
                mlflow.log_param("failed_reason", str(r.get("failed_reason")))

            for col in summary_df.columns:
                if col.endswith(("_mean", "_std")) or col in (
                    "train_kept_frac_mean",
                    "n_train_kept_mean",
                    "n_test_mean",
                ):
                    val = r.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    mlflow.log_metric(col, float(val))
