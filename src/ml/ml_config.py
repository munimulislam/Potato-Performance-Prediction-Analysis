"""
@File - ml_config.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _require_dict(d: Any, name: str) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"Expected '{name}' to be a dict in config")
    return d


def _require_list_str(x: Any, name: str) -> list[str]:
    if not isinstance(x, list) or not all(isinstance(v, str) for v in x):
        raise ValueError(f"Expected '{name}' to be a list[str]")
    return x


def _require_list_num(x: Any, name: str) -> list[float]:
    if not isinstance(x, list) or not all(isinstance(v, (int, float)) for v in x):
        raise ValueError(f"Expected '{name}' to be a list of numbers")
    return [float(v) for v in x]


@dataclass(frozen=True)
class DuckDbCfg:
    path: str


@dataclass(frozen=True)
class RelationsCfg:
    oa_mart: str


@dataclass(frozen=True)
class DatasetCfg:
    source: str
    path: str


@dataclass(frozen=True)
class FeaturesCfg:
    target: str
    keys: list[str]
    categorical: list[str]
    numeric: list[str]


@dataclass(frozen=True)
class CvCfg:
    strategy: str
    group_col: str
    n_splits: int


@dataclass(frozen=True)
class ExperimentCfg:
    name: str
    cv: CvCfg
    thresholds: list[float]
    models: list[str]
    random_state: int


@dataclass(frozen=True)
class MlflowCfg:
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class MlConfig:
    duckdb: DuckDbCfg
    relations: RelationsCfg
    features: FeaturesCfg
    dataset: DatasetCfg
    experiment: ExperimentCfg
    mlflow: MlflowCfg


def load_ml_config(config_path: str = "config/ml.yaml") -> MlConfig:
    path = Path(config_path)
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _require_dict(root, "root")

    duckdb = _require_dict(root.get("duckdb"), "duckdb")
    relations = _require_dict(root.get("relations"), "relations")
    dataset = _require_dict(root.get("dataset"), "dataset")
    features = _require_dict(root.get("features"), "features")
    experiment = _require_dict(root.get("experiment"), "experiment")
    cv = _require_dict(experiment.get("cv"), "experiment.cv")
    mlflow = _require_dict(root.get("mlflow"), "mlflow")

    cfg = MlConfig(
        duckdb=DuckDbCfg(path=str(duckdb["path"])),
        relations=RelationsCfg(oa_mart=str(relations["oa_mart"])),
        dataset=DatasetCfg(
            source=str(dataset.get("source", "duckdb")).lower(),
            path=str(dataset.get("parquet_path", "data/gold/o_a_mart.parquet")),
        ),
        features=FeaturesCfg(
            target=str(features["target"]),
            keys=_require_list_str(features["keys"], "features.keys"),
            categorical=_require_list_str(
                features["categorical"], "features.categorical"
            ),
            numeric=_require_list_str(features["numeric"], "features.numeric"),
        ),
        experiment=ExperimentCfg(
            name=str(experiment.get("name", "oa/global_genotype_cv1")),
            cv=CvCfg(
                strategy=str(cv.get("strategy", "group_kfold")),
                group_col=str(cv.get("group_col", "name1")),
                n_splits=int(cv.get("n_splits", 5)),
            ),
            thresholds=_require_list_num(
                experiment.get(
                    "thresholds",
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                ),
                "experiment.thresholds",
            ),
            models=_require_list_str(
                experiment.get("models", ["ridge", "rf"]), "experiment.models"
            ),
            random_state=int(experiment.get("random_state", 0)),
        ),
        mlflow=MlflowCfg(
            tracking_uri=str(mlflow.get("tracking_uri", "file:./mlruns")),
            experiment_name=str(mlflow.get("experiment_name", "oa/global_genotype")),
        ),
    )

    if cfg.dataset.source not in ("duckdb", "parquet"):
        raise ValueError("dataset.source must be one of: duckdb, parquet")
    if cfg.experiment.cv.strategy != "group_kfold":
        raise ValueError("Only cv.strategy=group_kfold implemented in this commit")
    if cfg.experiment.cv.n_splits < 2:
        raise ValueError("cv.n_splits must be >= 2")
    if not cfg.experiment.thresholds:
        raise ValueError("experiment.thresholds must be non-empty")
    if not cfg.experiment.models:
        raise ValueError("experiment.models must be non-empty")

    return cfg
