"""
@File - logging_utils.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 06/07/2026
@Description - Description of the file.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def create_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | run_id=%(run_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_stream_handler(
    formatter: logging.Formatter,
    run_filter: logging.Filter,
) -> logging.Handler:

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(run_filter)
    return handler


def create_rotating_file_handler(
    log_path: Path,
    formatter: logging.Formatter,
    run_filter: logging.Filter,
) -> logging.Handler:

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.addFilter(run_filter)
    return handler


def create_file_handler(
    log_path: Path,
    formatter: logging.Formatter,
    run_filter: logging.Filter,
) -> logging.Handler:

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.addFilter(run_filter)
    return handler


def build_logger(run_id: str, global_logs_dir, run_logs_dir) -> logging.Logger:
    logger = logging.getLogger("trails_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = create_formatter()
    run_filter = RunIdFilter(run_id)

    stream_handler = create_stream_handler(formatter, run_filter)
    global_handler = create_rotating_file_handler(
        Path(global_logs_dir) / "data_pipeline.log", formatter, run_filter
    )
    run_handler = create_file_handler(
        Path(run_logs_dir) / "data_pipeline.log",
        formatter,
        run_filter,
    )

    logger.addHandler(stream_handler)
    logger.addHandler(global_handler)
    logger.addHandler(run_handler)

    return logger
