"""
@File - prod_model.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 28/07/2026
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_dict_artifact, log_joblib_artifact
from ml.common.preprocess import make_preprocessor

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"oa/prod_model"
RUN_NAME = f"oa_svr_prod_train_{datetime.now()}"

REGISTERED_MODEL_NAME = "oa_svr_prod"

THRESHOLD = 0.5

SVR_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,
}

SIGNATURE_SAMPLE_N = 200


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"dataset.yaml must parse to a dict, got {type(obj)}")
    return obj


def _numeric_missing_frac(df: pd.DataFrame, numeric_cols: List[str]) -> pd.Series:
    if not numeric_cols:
        return pd.Series(0.0, index=df.index)
    return df[numeric_cols].isna().mean(axis=1)


def _validate_feature_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")


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
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=RUN_NAME, help="MLflow run name")
    ap.add_argument(
        "--register",
        action="store_true",
        help="If set, register the model into MLflow Model Registry under REGISTERED_MODEL_NAME",
    )
    args = ap.parse_args()

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

    t0 = time.perf_counter()
    df = load_dataframe(dataset_path)

    needed_cols = feature_cols + [target_col]
    _validate_feature_columns_exist(df, needed_cols)
    df = df[needed_cols].copy()

    df = _coerce_types(
        df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    if df[target_col].isna().all():
        raise ValueError(
            f"Target '{target_col}' is all-null after coercion. Cannot train."
        )

    miss_frac = _numeric_missing_frac(df, numeric_cols=numeric_cols)
    train_mask = (miss_frac <= THRESHOLD) & (df[target_col].notna())

    df_train = df.loc[train_mask].copy()
    X_train = df_train[feature_cols]
    y_train = df_train[target_col].to_numpy(dtype=float)

    if len(df_train) == 0:
        raise ValueError(
            "No rows left after training filter. Check threshold policy and data."
        )

    pre = make_preprocessor(
        profile="linear", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )
    est = SVR(**SVR_PARAMS)
    pipe = Pipeline(steps=[("pre", pre), ("est", est)])

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "stage": "prod_train",
                "model_family": "svr",
                "preprocess_profile": "kernel",
                "target": target_col,
            }
        )

        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "threshold_train_missing_frac_numeric_leq": THRESHOLD,
                "n_rows_total": int(len(df)),
                "n_rows_train_used": int(len(df_train)),
                "train_kept_frac": (
                    float(len(df_train) / len(df)) if len(df) else float("nan")
                ),
                **{f"svr__{k}": v for k, v in SVR_PARAMS.items()},
            }
        )

        pipe.fit(X_train, y_train)

        sample_n = min(SIGNATURE_SAMPLE_N, len(df_train))
        X_sig = X_train.head(sample_n)
        y_sig_pred = pipe.predict(X_sig)

        signature = infer_signature(X_sig, y_sig_pred)
        register_name = REGISTERED_MODEL_NAME if args.register else None

        schema = {
            "target": target_col,
            "features": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "feature_cols_order": feature_cols,
            },
            "training_policy": {
                "train_row_filter": f"missing_frac_numeric <= {THRESHOLD} AND target_not_null",
                "inference_filter": "none (pipeline handles missingness)",
            },
        }

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="oa_prediction_model",
            signature=signature,
            input_example=X_sig,
            registered_model_name=register_name,
            tags={"profile": "kernel", "prediction": "oa"},
            params=SVR_PARAMS,
            serialization_format="cloudpickle",
        )

        log_dict_artifact(schema, "model_schema.json")
        log_joblib_artifact(pipe, "pipeline.joblib")

        elapsed = time.perf_counter() - t0
        print(f"Trained and logged production model in {_fmt(elapsed)}")
        print(f"MLflow run_id: {run_id}")
        print(f"Load by run:   mlflow.sklearn.load_model('runs:/{run_id}/model')")
        if args.register:
            print(f"Registered as: models:/{REGISTERED_MODEL_NAME}/latest")

    return 0


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}m"


if __name__ == "__main__":
    raise SystemExit(main())
