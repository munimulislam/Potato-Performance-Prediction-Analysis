"""
@File - run_context.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 06/07/2026
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from .config import PathConfig, PipelineConfig
import yaml
import json


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: str
    start_time_utc: str
    config: PipelineConfig


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%d%m%Y_%H%M%S")


def create_pipeline_dirs(paths_cfg: PathConfig):
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


def publish_run_meta(context: RunContext, meta_dir: str):
    meta_path = Path(meta_dir) / "meta.yaml"
    meta = {
        "run_id": context.run_id,
        "start_time_utc": context.start_time_utc,
        "config": json.dumps(asdict(context.config)),
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def init_run(config: PipelineConfig) -> RunContext:
    create_pipeline_dirs(config.paths)
    run_id = create_run_id()
    run_dir = Path(config.paths.artifacts_runs) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    context = RunContext(
        run_id=run_id,
        run_dir=str(run_dir),
        start_time_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
    )

    publish_run_meta(context, str(run_dir))

    return context
