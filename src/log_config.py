"""
@File - log_config.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 09/07/2026
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .log_context import get_run_id

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | run_id=%(run_id)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = get_run_id()
        return True


def _formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)


def configure_logging(
    *, logs_dir: str, run_logs_dir: str, level: int = logging.DEBUG
) -> None:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    Path(run_logs_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    if getattr(root_logger, "_trials_pipeline_logging_configured", False):
        return

    root_logger.setLevel(level)

    run_filter = RunIdFilter()
    fmt = _formatter()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    console.addFilter(run_filter)
    console.name = "trials_pipeline_console"
    root_logger.addHandler(console)

    global_file = RotatingFileHandler(
        Path(logs_dir) / "pipeline.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    global_file.setLevel(level)
    global_file.setFormatter(fmt)
    global_file.addFilter(run_filter)
    global_file.name = "_trials_pipeline_global_file"
    root_logger.addHandler(global_file)

    run_file = logging.FileHandler(
        Path(run_logs_dir) / "pipeline.log", encoding="utf-8"
    )
    run_file.setLevel(level)
    run_file.setFormatter(_formatter())
    run_file.addFilter(RunIdFilter())
    run_file.name = "_trials_pipeline_run_file"
    root_logger.addHandler(run_file)

    root_logger._trials_pipeline_logging_configured = True
