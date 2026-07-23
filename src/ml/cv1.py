"""
@File - cv1.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 23/07/2026
"""

# import argparse

# import mlflow
# import numpy as np
# import pandas as pd

# from .data import load_oa_dataframe
# from .config import load_ml_config
# from .runner import log_sweep_to_mlflow, run_cv1_sweep


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Run primary CV1 (global genotype holdout) experiment."
#     )
#     ap.add_argument("--config", default="src/ml/config/ml.yaml")
#     args = ap.parse_args()

#     cfg = load_ml_config(args.config)
#     df = load_oa_dataframe(cfg)

#     df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)

#     mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
#     mlflow.set_experiment(cfg.mlflow.experiment_name)

#     parent_name = cfg.experiment.name
#     with mlflow.start_run(run_name=parent_name):
#         mlflow.log_param("dataset_source", cfg.dataset.source)
#         mlflow.log_param(
#             "relation", cfg.relations.oa_mart if cfg.dataset.source == "duckdb" else ""
#         )
#         mlflow.log_metric("n_rows", float(len(df)))
#         mlflow.log_metric("n_locations", float(df["location"].nunique()))
#         mlflow.log_metric("n_clones", float(df["name1"].nunique()))
#         mlflow.log_param("models", ",".join(cfg.experiment.models))
#         mlflow.log_param(
#             "thresholds", ",".join(str(t) for t in cfg.experiment.thresholds)
#         )

#         summary = run_cv1_sweep(
#             df=df,
#             target_col=cfg.features.target,
#             numeric_cols=cfg.features.numeric,
#             categorical_cols=cfg.features.categorical,
#             group_col=cfg.experiment.cv.group_col,
#             n_splits=cfg.experiment.cv.n_splits,
#             thresholds=cfg.experiment.thresholds,
#             model_names=cfg.experiment.models,
#             random_state=cfg.experiment.random_state,
#         )

#         log_sweep_to_mlflow(
#             summary_df=summary,
#             scenario="global_genotype",
#             cv_type="CV1",
#             split_axis="genotype",
#             group_col=cfg.experiment.cv.group_col,
#             n_splits=cfg.experiment.cv.n_splits,
#         )

#         # Print ranking (top 3 model families)
#         ok = summary[(summary["status"] == "OK") & summary["rmse_mean"].notna()].copy()
#         ok = ok.sort_values(
#             ["rmse_mean", "rmse_std", "r2_mean"], ascending=[True, True, False]
#         )

#         print("\n=== CV1 leaderboard (top 15 rows) ===")
#         print(
#             ok[
#                 [
#                     "model",
#                     "threshold",
#                     "rmse_mean",
#                     "rmse_std",
#                     "r2_mean",
#                     "r2_std",
#                     "train_kept_frac_mean",
#                 ]
#             ]
#             .head(15)
#             .to_string(index=False)
#         )

#         best_per_model = (
#             ok.sort_values(
#                 ["model", "rmse_mean", "rmse_std"], ascending=[True, True, True]
#             )
#             .groupby("model", as_index=False)
#             .head(1)
#             .sort_values(["rmse_mean", "rmse_std"], ascending=[True, True])
#         )

#         print("\n=== Best threshold per model (ranked) ===")
#         print(
#             best_per_model[
#                 [
#                     "model",
#                     "threshold",
#                     "rmse_mean",
#                     "rmse_std",
#                     "r2_mean",
#                     "train_kept_frac_mean",
#                 ]
#             ].to_string(index=False)
#         )

#         top3 = best_per_model.head(3)
#         print("\n=== TOP 3 models ===")
#         print(
#             top3[["model", "threshold", "rmse_mean", "rmse_std"]].to_string(index=False)
#         )


# if __name__ == "__main__":
#     main()

import argparse
from dataclasses import asdict

import mlflow
import pandas as pd

from .data import load_oa_dataframe
from .config import load_ml_config
from .models import make_model
from .runner import evaluate_combo_on_splits
from .splits import group_kfold_splits


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run primary CV1 (global genotype holdout) experiment."
    )
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_ml_config(args.config)
    df = load_oa_dataframe(cfg)

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)

    required_cols = (
        [cfg.features.target]
        + cfg.features.keys
        + cfg.features.categorical
        + cfg.features.numeric
    )
    missing = sorted([c for c in required_cols if c not in df.columns])
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    splits = list(
        group_kfold_splits(
            df,
            group_col=cfg.experiment.cv.group_col,
            n_splits=cfg.experiment.cv.n_splits,
        )
    )

    # MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.set_tag("scenario", "global_genotype")
        mlflow.set_tag("cv_type", "CV1")
        mlflow.set_tag("split_axis", "genotype")
        mlflow.set_tag("group_col", cfg.experiment.cv.group_col)
        mlflow.log_param("n_splits", cfg.experiment.cv.n_splits)
        mlflow.log_param(
            "thresholds", ",".join(str(t) for t in cfg.experiment.thresholds)
        )
        mlflow.log_param("models", ",".join(cfg.experiment.models))
        mlflow.log_param(
            "use_missing_indicator_sweep",
            ",".join(str(b) for b in cfg.experiment.use_missing_indicator),
        )

        mlflow.log_metric("n_rows", float(len(df)))
        mlflow.log_metric("n_locations", float(df["location"].nunique()))
        mlflow.log_metric("n_clones", float(df["name1"].nunique()))

        all_rows = []

        for model_name in cfg.experiment.models:
            model_spec = make_model(
                model_name, random_state=cfg.experiment.random_state
            )

            for use_ind in cfg.experiment.use_missing_indicator:
                for threshold in cfg.experiment.thresholds:
                    combo = evaluate_combo_on_splits(
                        df=df,
                        splits=splits,
                        model_spec=model_spec,
                        threshold=threshold,
                        use_missing_indicator=use_ind,
                        target_col=cfg.features.target,
                        numeric_cols=cfg.features.numeric,
                        categorical_cols=cfg.features.categorical,
                    )
                    all_rows.append(asdict(combo))

                    run_name = f"{combo.model}__thr={combo.threshold}__ind={combo.use_missing_indicator}"
                    with mlflow.start_run(run_name=run_name, nested=True):
                        mlflow.set_tag("model_type", combo.model)
                        mlflow.log_param("threshold", combo.threshold)
                        mlflow.log_param(
                            "use_missing_indicator", combo.use_missing_indicator
                        )
                        mlflow.log_param("status", combo.status)
                        if combo.failed_reason:
                            mlflow.log_param("failed_reason", combo.failed_reason)

                        # metrics
                        for k, v in asdict(combo).items():
                            if k in (
                                "model",
                                "threshold",
                                "use_missing_indicator",
                                "status",
                                "failed_reason",
                            ):
                                continue
                            if v is None:
                                continue
                            mlflow.log_metric(k, float(v))

        results = pd.DataFrame(all_rows)

        ok = results[results["status"] == "OK"].copy()
        ok = ok.sort_values(
            ["rmse_mean", "rmse_std", "r2_mean"], ascending=[True, True, False]
        )

        print("\n=== CV1 leaderboard (top 20 rows) ===")
        cols = [
            "model",
            "use_missing_indicator",
            "threshold",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mape_mean",
            "r2_mean",
            "pearson_r_mean",
            "train_kept_frac_mean",
            "test_incomplete_frac_mean",
        ]
        print(ok[cols].head(20).to_string(index=False))

        ok["model_variant"] = (
            ok["model"].astype(str) + "__ind=" + ok["use_missing_indicator"].astype(str)
        )

        best_per_variant = (
            ok.sort_values(
                ["model_variant", "rmse_mean", "rmse_std"], ascending=[True, True, True]
            )
            .groupby("model_variant", as_index=False)
            .head(1)
            .sort_values(["rmse_mean", "rmse_std"], ascending=[True, True])
        )

        print("\n=== Best threshold per model variant (ranked) ===")
        print(
            best_per_variant[
                [
                    "model",
                    "use_missing_indicator",
                    "threshold",
                    "rmse_mean",
                    "rmse_std",
                    "r2_mean",
                ]
            ].to_string(index=False)
        )

        top3 = best_per_variant.head(3)
        print("\n=== TOP 3 model variants to carry forward ===")
        print(
            top3[
                ["model", "use_missing_indicator", "threshold", "rmse_mean", "rmse_std"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
