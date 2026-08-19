"""
@File - imoutation_strategy.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 12/08/2026
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import root_mean_squared_error

from ml.common.data import load_dataframe
from ml.common.mlflow import init_mlflow, log_df_artifact
from ml.common.splits import group_kfold_splits
from ml.common.metrics import mean_std_se
from ml.common.models import make_model

DATASET_YAML_PATH = Path(__file__).resolve().parent / "config" / "dataset.yaml"
TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "oa/imputation_strategy"
PARENT_RUN_NAME = "imputation_strategy_test"

GROUP_COL = "name1"
ENV_COL = "env_type"
N_SPLITS = 5

THRESHOLD = 1.0
RANDOM_STATE = 42

IMPUTE_STRATEGIES: List[Tuple[str, bool]] = [
    ("median", False),
    ("median", True),
    ("mean", False),
    ("knn", False),
    ("iterative", False),
]


IMPUTE_MODELS: List[Tuple[str, Dict[str, Any]]] = [
    ("ridge", {"alpha": 1.0}),
    ("rf", {"n_estimators": 500, "n_jobs": -1, "random_state": RANDOM_STATE}),
    ("svr", {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}),
    (
        "xgb",
        {
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
    ),
]

KNN_NEIGHBORS = 15
ITERATIVE_MAX_ITER = 10

XGB_REF_PARAMS = {
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
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("dataset.yaml must parse to a dict")
    return obj


def _upper_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def _numeric_missing_frac(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    if not numeric_cols:
        return np.zeros(len(df), dtype=float)
    return df[numeric_cols].isna().mean(axis=1).to_numpy(dtype=float)


def _make_onehot(handle_unknown: str = "ignore", dense: bool = True) -> OneHotEncoder:
    sparse_out = not dense
    return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_out)


def _to_float64(X):
    return X.astype(np.float64)


def _build_numeric_imputer(strategy: str):
    s = strategy.lower().strip()
    if s == "median":
        return SimpleImputer(strategy="median")
    if s == "mean":
        return SimpleImputer(strategy="mean")
    if s == "knn":
        return KNNImputer(n_neighbors=KNN_NEIGHBORS)
    if s == "iterative":
        return IterativeImputer(
            max_iter=ITERATIVE_MAX_ITER,
            random_state=RANDOM_STATE,
            initial_strategy="median",
        )
    raise ValueError(f"Unknown imputation strategy: {strategy}")


def _make_preprocessor_impute(
    *,
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_imputer: str,
    add_indicator: bool,
) -> Pipeline:

    num_values = Pipeline(
        steps=[
            ("impute", _build_numeric_imputer(numeric_imputer)),
            ("scale", StandardScaler()),
        ]
    )

    transformers = [("num_values", num_values, numeric_cols)]

    if add_indicator:
        transformers.append(
            (
                "num_miss",
                Pipeline(steps=[("miss", MissingIndicator(features="all"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        cat = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="__M__")),
                ("onehot", _make_onehot(dense=True)),
            ]
        )
        transformers.append(("cat", cat, categorical_cols))

    coltf = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("columns", coltf),
            ("to_float64", FunctionTransformer(_to_float64)),
        ]
    )


def _make_preprocessor_native_nan_booster(
    *,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:

    transformers = []
    if categorical_cols:
        cat = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="__M__")),
                ("onehot", _make_onehot(dense=True)),
            ]
        )
        transformers.append(("cat", cat, categorical_cols))

    coltf = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("columns", coltf),
            ("to_float64", FunctionTransformer(_to_float64)),
        ]
    )


def _variety_spearman(
    name1: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> float:
    if mask.sum() == 0:
        return float("nan")
    d = pd.DataFrame(
        {"name1": name1[mask], "yt": y_true[mask], "yp": y_pred[mask]}
    ).dropna()
    if d.empty:
        return float("nan")
    agg = d.groupby("name1", as_index=True).agg(yt=("yt", "mean"), yp=("yp", "mean"))
    if len(agg) < 2:
        return float("nan")
    return float(agg["yt"].corr(agg["yp"], method="spearman"))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def evaluate(
    *,
    df: pd.DataFrame,
    splits,
    target_col: str,
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    y_all: np.ndarray,
    name1_all: np.ndarray,
    env_all: np.ndarray,
    miss_all: np.ndarray,
    model_name: str,
    estimator,
    preprocessor: Pipeline,
) -> Dict[str, Any]:

    rmse_by_fold: Dict[int, float] = {}
    rmse_ne_by_fold: Dict[int, float] = {}
    rmse_med_by_fold: Dict[int, float] = {}
    spear_by_fold: Dict[int, float] = {}

    t0 = time.perf_counter()

    for sp in splits:
        fold = int(sp.fold)
        tri, tei = sp.train_idx, sp.test_idx

        keep = miss_all[tri] <= THRESHOLD
        kept = tri[keep]
        kept = kept[~np.isnan(y_all[kept])]
        if len(kept) == 0:
            continue

        y_test = y_all[tei]
        obs = ~np.isnan(y_test)
        if obs.sum() == 0:
            continue

        pipe = Pipeline([("pre", preprocessor), ("est", estimator)])
        pipe.fit(df.iloc[kept][feature_cols], y_all[kept])

        pred = np.asarray(pipe.predict(df.iloc[tei][feature_cols]), dtype=float)

        rmse_by_fold[fold] = _rmse(y_test[obs].astype(float), pred[obs].astype(float))
        spear_by_fold[fold] = _variety_spearman(name1_all[tei], y_test, pred, obs)

        env_test = env_all[tei]
        ne = obs & (env_test == "NE")
        med = obs & (env_test == "MED")

        rmse_ne_by_fold[fold] = (
            _rmse(y_test[ne].astype(float), pred[ne].astype(float))
            if ne.sum()
            else float("nan")
        )
        rmse_med_by_fold[fold] = (
            _rmse(y_test[med].astype(float), pred[med].astype(float))
            if med.sum()
            else float("nan")
        )

    folds = sorted(rmse_by_fold.keys())
    rmse_series = pd.Series([rmse_by_fold[f] for f in folds])
    rmse_mean, rmse_std, rmse_se, n_valid = mean_std_se(rmse_series)

    out = {
        "model": model_name,
        "rmse_mean": float(rmse_mean),
        "rmse_std": float(rmse_std),
        "rmse_se": float(rmse_se),
        "n_folds_used": int(n_valid),
        "rmse_ne_mean": (
            float(np.nanmean([rmse_ne_by_fold[f] for f in folds]))
            if folds
            else float("nan")
        ),
        "rmse_med_mean": (
            float(np.nanmean([rmse_med_by_fold[f] for f in folds]))
            if folds
            else float("nan")
        ),
        "spearman_variety_mean": (
            float(np.nanmean([spear_by_fold[f] for f in folds]))
            if folds
            else float("nan")
        ),
        "wall_time_sec": float(time.perf_counter() - t0),
        "_rmse_by_fold": rmse_by_fold,
    }
    return out


def main() -> int:
    cfg = _read_yaml(DATASET_YAML_PATH)
    ds = cfg["dataset"]
    feats = ds["features"]

    dataset_path = str(ds["path"])
    target_col = str(feats["target"])
    numeric_cols = list(feats.get("numeric", []))
    categorical_cols = list(feats.get("categorical", []))
    feature_cols = numeric_cols + categorical_cols

    init_mlflow(TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    df = load_dataframe(dataset_path)

    required = sorted(set([target_col, GROUP_COL, ENV_COL] + feature_cols))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required].copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols:
        df[c] = df[c].astype("string")
    df[GROUP_COL] = df[GROUP_COL].astype("string")
    df[ENV_COL] = df[ENV_COL].astype("string")

    y_all = df[target_col].to_numpy(dtype=float)
    name1_all = df[GROUP_COL].to_numpy()
    env_all = _upper_str(df[ENV_COL]).to_numpy()
    miss_all = _numeric_missing_frac(df, numeric_cols)

    splits = list(
        group_kfold_splits(
            df, target_col=target_col, group_col=GROUP_COL, n_splits=N_SPLITS
        )
    )

    results: List[Dict[str, Any]] = []

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags({"phase": "impute_strategy_test", "cv": "cv1_groupkfold"})
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "target_col": target_col,
                "threshold": THRESHOLD,
                "n_splits": N_SPLITS,
                "n_rows": int(len(df)),
                "knn_neighbors": KNN_NEIGHBORS,
                "iterative_max_iter": ITERATIVE_MAX_ITER,
            }
        )

        for model_name, model_params in IMPUTE_MODELS:
            for strat, add_ind in IMPUTE_STRATEGIES:
                strategy_label = strat + ("+indicator" if add_ind else "")
                run_name = f"{model_name}__{strategy_label}"

                print(f" {run_name} ...", flush=True)

                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "model": model_name,
                            "imputer": strat,
                            "add_indicator": str(add_ind),
                        }
                    )
                    mlflow.log_params(
                        {
                            "model": model_name,
                            "imputer": strat,
                            "add_indicator": add_ind,
                        }
                    )

                    ms = make_model(model_name, model_params)

                    pre = _make_preprocessor_impute(
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                        numeric_imputer=strat,
                        add_indicator=add_ind,
                    )

                    r = evaluate(
                        df=df,
                        splits=splits,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                        y_all=y_all,
                        name1_all=name1_all,
                        env_all=env_all,
                        miss_all=miss_all,
                        model_name=model_name,
                        estimator=ms.estimator,
                        preprocessor=pre,
                    )

                    r["strategy"] = strategy_label
                    results.append(r)

                    mlflow.log_metrics(
                        {
                            "rmse_mean": r["rmse_mean"],
                            "rmse_se": r["rmse_se"],
                            "rmse_ne_mean": r["rmse_ne_mean"],
                            "rmse_med_mean": r["rmse_med_mean"],
                            "n_folds_used": r["n_folds_used"],
                        }
                    )

                    print(
                        f"    rmse={r['rmse_mean']:.4f} (se {r['rmse_se']:.4f}) "
                        f"time={r['wall_time_sec']:.1f}s",
                        flush=True,
                    )

        print("xgb native_nan reference ...", flush=True)
        try:
            ms_xgb = make_model("xgb", XGB_REF_PARAMS)

            pre_native = _make_preprocessor_native_nan_booster(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            )

            rnb = evaluate(
                df=df,
                splits=splits,
                target_col=target_col,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                y_all=y_all,
                name1_all=name1_all,
                env_all=env_all,
                miss_all=miss_all,
                model_name="xgb",
                estimator=ms_xgb.estimator,
                preprocessor=pre_native,
            )
            rnb["strategy"] = "native_nan"
            results.append(rnb)

            print(
                f"    rmse={rnb['rmse_mean']:.4f} (se {rnb['rmse_se']:.4f})", flush=True
            )
        except Exception as e:
            print(f"    skipped xgb native_nan reference (reason: {e})", flush=True)

        summary = pd.DataFrame(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        )
        summary = summary.sort_values(
            ["rmse_mean", "spearman_variety_mean"], ascending=[True, False]
        )

        log_df_artifact(summary, "imputation_strategy_summary.csv")

        print("\n=== imputation strategy comparison (sorted by RMSE) ===")
        print(
            summary[
                [
                    "model",
                    "strategy",
                    "rmse_mean",
                    "rmse_se",
                    "spearman_variety_mean",
                    "wall_time_sec",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
