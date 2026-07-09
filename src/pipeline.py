"""
@File - pipeline.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from .log_context import get_run_id, set_run_id

from .config import load_config
from .run_context import RunContext, init_run
from .ingest import IngestSummary, ingest_incoming
from pathlib import Path
import json
from dataclasses import asdict
import logging
from .log_config import configure_logging

logger = logging.getLogger(__name__)


def publish_ingest_result(context: RunContext, result: list[IngestSummary]):
    data = asdict(context)
    data["result"] = [asdict(res) for res in result]
    file_path = Path(context.run_dir) / "ingest_result.json"
    file_path.write_text(json.dumps(data, indent=4))


def main():
    config = load_config()
    run_context = init_run(config)
    run_dir = Path(run_context.run_dir)

    set_run_id(run_context.run_id)
    configure_logging(
        logs_dir=config.paths.logs,
        run_logs_dir=f"{config.paths.artifacts_runs}/{run_context.run_id}",
    )
    logger = logging.getLogger(__name__)
    logger.info("Pipeline Started")

    valid, reject, ingest_summary_list = ingest_incoming(
        run_context.run_id, config.paths.incoming, config.excel.extensions
    )

    valid.to_csv(run_dir / "valid.csv")
    reject.to_csv(run_dir / "reject.csv")
    publish_ingest_result(run_context, ingest_summary_list)


if __name__ == "__main__":
    main()
