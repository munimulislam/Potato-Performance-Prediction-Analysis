import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def export_and_plot_experiment_runs(
    parent_run: str,
    tracking_uri: str = None,
):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    child_runs = mlflow.search_runs(
        experiment_names=["oa/missingness_threshold"],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run}'",
    )

    if child_runs.empty:
        print("No runs found in this experiment.")
        return None

    df = child_runs.copy()

    threshold_col = next(
        (c for c in df.columns if c == "params.threshold"),
        None,
    )

    model_col = next(
        (c for c in df.columns if c == "params.model"),
        None,
    )

    if threshold_col is None:
        raise ValueError("Could not find 'params.threshold' in MLflow runs.")

    if model_col is None:
        raise ValueError("Could not find 'params.model' in MLflow runs.")

    df["threshold"] = pd.to_numeric(
        df[threshold_col],
        errors="coerce",
    )

    df["model"] = df[model_col]

    df = (
        df.dropna(subset=["threshold", "model"])
        .sort_values(["model", "threshold"])
        .reset_index(drop=True)
    )

    metrics_config = {
        "metrics.rmse_all_mean": {
            "title": "RMSE vs Missingness Threshold",
            "ylabel": "RMSE",
        },
        "metrics.mae_all_mean": {
            "title": "MAE vs Missingness Threshold",
            "ylabel": "MAE",
        },
        "metrics.r2_all_mean": {
            "title": "R² vs Missingness Threshold",
            "ylabel": "R² Score",
        },
    }

    models = sorted(df["model"].unique())
    colors = sns.color_palette("tab10", n_colors=len(models))
    model_palette = {model: color for model, color in zip(models, colors)}

    print("Model colour mapping:")
    for model, color in model_palette.items():
        print(f"  {model}: {color}")

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    metrics_config = [
        (
            "metrics.rmse_all_mean",
            "RMSE",
            "RMSE (Lower is Better)",
        ),
        (
            "metrics.mae_all_mean",
            "MAE",
            "MAE (Lower is Better)",
        ),
        (
            "metrics.r2_all_mean",
            "R²",
            "R² (Higher is Better)",
        ),
    ]

    for ax, (metric_col, title, ylabel) in zip(axes, metrics_config):

        if metric_col not in df.columns:
            print(f"Skipping {metric_col}: column not found.")
            continue

        df[metric_col] = pd.to_numeric(
            df[metric_col],
            errors="coerce",
        )

        plot_df = df.dropna(subset=["threshold", "model", metric_col])

        sns.lineplot(
            data=plot_df,
            x="threshold",
            y=metric_col,
            hue="model",
            palette=model_palette,
            marker="o",
            markersize=6,
            linewidth=2,
            ax=ax,
            legend=False,
        )

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("Missingness Threshold", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        ax.set_xticks(sorted(plot_df["threshold"].unique()))

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

    handles = [
        mpatches.Patch(color=model_palette[model], label=model) for model in models
    ]

    fig.legend(
        handles=handles,
        labels=models,
        title="Model",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=10,
        title_fontsize=11,
    )

    fig.suptitle(
        "Model Performance vs Missingness Threshold",
        fontsize=15,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 0.90, 0.95])

    plt.show()
    return df


if __name__ == "__main__":

    df = export_and_plot_experiment_runs(
        parent_run="7ac89a6142934148af83e758482b09fd",
        tracking_uri="sqlite:///mlflow.db",
    )
