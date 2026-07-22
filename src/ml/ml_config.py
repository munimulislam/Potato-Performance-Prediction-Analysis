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


@dataclass(frozen=True)
class DuckDbCfg:
    path: str


@dataclass(frozen=True)
class RelationsCfg:
    oa_mart: str


@dataclass(frozen=True)
class OutputsCfg:
    oa_mart_parquet: str
    oa_mart_meta: str


@dataclass(frozen=True)
class MlConfig:
    duckdb: DuckDbCfg
    relations: RelationsCfg
    outputs: OutputsCfg


def load_ml_config(config_path: str = "config/ml.yaml") -> MlConfig:
    path = Path(config_path)
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _require_dict(root, "root")

    duckdb_dict = _require_dict(root.get("duckdb"), "duckdb")
    relations_dict = _require_dict(root.get("relations"), "relations")
    outputs_dict = _require_dict(root.get("outputs"), "outputs")

    cfg = MlConfig(
        duckdb=DuckDbCfg(path=str(duckdb_dict["path"])),
        relations=RelationsCfg(oa_mart=str(relations_dict["oa_mart"])),
        outputs=OutputsCfg(
            oa_mart_parquet=str(outputs_dict["oa_mart_parquet"]),
            oa_mart_meta=str(outputs_dict["oa_mart_meta"]),
        ),
    )
    return cfg
