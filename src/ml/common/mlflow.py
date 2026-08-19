"""
@File - mlflow.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 25/07/2026
"""

import json
import tempfile
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import joblib


def init_mlflow(tracking_uri: Optional[str], experiment_name: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()

    try:
        experiments = client.search_experiments(
            filter_string=f"name = '{experiment_name}'", view_type=3
        )

        if experiments:
            exp = experiments[0]
            if exp.lifecycle_stage == "deleted":
                client.restore_experiment(exp.experiment_id)

        mlflow.set_experiment(experiment_name)
    except mlflow.MlflowException as e:
        print(f"Failed to initialize MLflow experiment: {e}")
        raise e


def log_df_artifact(df: pd.DataFrame, filename: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/{filename}"
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)


def log_dict_artifact(obj: Dict[str, Any], filename: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        mlflow.log_artifact(path)


def log_joblib_artifact(obj, filename: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/{filename}"
        joblib.dump(obj, path)
        mlflow.log_artifact(path)
