from __future__ import annotations

import argparse

import mlflow
import numpy as np
import pandas as pd

from .data import load_oa_dataframe
from .ml_config import load_ml_config
from .runner import log_sweep_to_mlflow, run_cv1_sweep


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run primary CV1 (global genotype holdout) experiment."
    )
    ap.add_argument("--config", default="src/ml/config/ml.yaml")
    args = ap.parse_args()

    cfg = load_ml_config(args.config)
    df = load_oa_dataframe(cfg)

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    parent_name = cfg.experiment.name
    with mlflow.start_run(run_name=parent_name):
        mlflow.log_param("dataset_source", cfg.dataset.source)
        mlflow.log_param(
            "relation", cfg.relations.oa_mart if cfg.dataset.source == "duckdb" else ""
        )
        mlflow.log_metric("n_rows", float(len(df)))
        mlflow.log_metric("n_locations", float(df["location"].nunique()))
        mlflow.log_metric("n_clones", float(df["name1"].nunique()))
        mlflow.log_param("models", ",".join(cfg.experiment.models))
        mlflow.log_param(
            "thresholds", ",".join(str(t) for t in cfg.experiment.thresholds)
        )

        summary = run_cv1_sweep(
            df=df,
            target_col=cfg.features.target,
            numeric_cols=cfg.features.numeric,
            categorical_cols=cfg.features.categorical,
            group_col=cfg.experiment.cv.group_col,
            n_splits=cfg.experiment.cv.n_splits,
            thresholds=cfg.experiment.thresholds,
            model_names=cfg.experiment.models,
            random_state=cfg.experiment.random_state,
        )

        log_sweep_to_mlflow(
            summary_df=summary,
            scenario="global_genotype",
            cv_type="CV1",
            split_axis="genotype",
            group_col=cfg.experiment.cv.group_col,
            n_splits=cfg.experiment.cv.n_splits,
        )

        # Print ranking (top 3 model families)
        ok = summary[(summary["status"] == "OK") & summary["rmse_mean"].notna()].copy()
        ok = ok.sort_values(
            ["rmse_mean", "rmse_std", "r2_mean"], ascending=[True, True, False]
        )

        print("\n=== CV1 leaderboard (top 15 rows) ===")
        print(
            ok[
                [
                    "model",
                    "threshold",
                    "rmse_mean",
                    "rmse_std",
                    "r2_mean",
                    "r2_std",
                    "train_kept_frac_mean",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

        best_per_model = (
            ok.sort_values(
                ["model", "rmse_mean", "rmse_std"], ascending=[True, True, True]
            )
            .groupby("model", as_index=False)
            .head(1)
            .sort_values(["rmse_mean", "rmse_std"], ascending=[True, True])
        )

        print("\n=== Best threshold per model (ranked) ===")
        print(
            best_per_model[
                [
                    "model",
                    "threshold",
                    "rmse_mean",
                    "rmse_std",
                    "r2_mean",
                    "train_kept_frac_mean",
                ]
            ].to_string(index=False)
        )

        top3 = best_per_model.head(3)
        print("\n=== TOP 3 models ===")
        print(
            top3[["model", "threshold", "rmse_mean", "rmse_std"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
