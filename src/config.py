"""
@Author - MdMunimul.Islam@:teagasc.ie
@Create Date - 06/07/2026
"""

from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    incoming: str
    archive: str
    gold: str
    rejects: str
    artifacts_runs: str
    logs: str


@dataclass(frozen=True)
class BusinessKeyConfig:
    columns: list[str]  # ordered


@dataclass(frozen=True)
class ExcelConfig:
    extensions: list[str]
    default_sheet: int | str = 0


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig
    business_key: BusinessKeyConfig
    excel: ExcelConfig


def _require_dict(d: Any, name: str) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"Expected '{name}' to be a dict in config")

    return d


def load_business_key_config(root: dict) -> BusinessKeyConfig:
    business_key_dict = _require_dict(root.get("business_key"), "business_key")
    business_key_cols = business_key_dict.get("columns")

    if not isinstance(business_key_cols, list) or not all(
        isinstance(x, str) for x in business_key_cols
    ):
        raise ValueError("business_key columns must be a list of strings")

    business_key_cfg = BusinessKeyConfig(columns=business_key_dict["columns"])

    return business_key_cfg


def load_path_config(root: dict) -> PathConfig:
    paths_dict = _require_dict(root.get("paths"), "paths")

    paths_cfg = PathConfig(
        incoming=str(paths_dict["incoming"]),
        archive=str(paths_dict["archive"]),
        gold=str(paths_dict["gold"]),
        rejects=str(paths_dict["rejects"]),
        artifacts_runs=str(paths_dict["artifacts_runs"]),
        logs=str(paths_dict["logs"]),
    )

    paths = [
        paths_cfg.incoming,
        paths_cfg.archive,
        paths_cfg.gold,
        paths_cfg.rejects,
        paths_cfg.artifacts_runs,
        paths_cfg.logs,
    ]

    for d in paths:
        Path(d).mkdir(parents=True, exist_ok=True)

    return paths_cfg


def load_excel_config(root: dict) -> ExcelConfig:
    excel_dict = _require_dict(root.get("excel"), "excel")
    file_extensions = excel_dict.get("extensions")

    if not isinstance(file_extensions, list) or not all(
        isinstance(x, str) for x in file_extensions
    ):
        raise ValueError("file extensions must be a list of strings")

    excel_cfg = ExcelConfig(
        extensions=excel_dict["extensions"],
        default_sheet=excel_dict.get("default_sheet", 0),
    )

    return excel_cfg


def load_config(config_path: str = "config/pipeline.yaml") -> PipelineConfig:
    path = Path(config_path)
    root_dict = yaml.safe_load(path.read_text(encoding="utf-8"))
    root_dict = _require_dict(root_dict, "root")

    cfg = PipelineConfig(
        paths=load_path_config(root_dict),
        business_key=load_business_key_config(root_dict),
        excel=load_excel_config(root_dict),
    )

    return cfg
