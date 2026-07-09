"""
@File - pipeline.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from .config import load_config
from .run_context import RunContext, init_run
from .ingest import IngestSummary, ingest_incoming
from pathlib import Path
import json
from dataclasses import asdict


def publish_run(context: RunContext, result: list[IngestSummary]):
    data = asdict(context)
    data["result"] = [asdict(res) for res in result]
    file_path = Path(context.run_dir) / "ingest_result.json"
    file_path.write_text(json.dumps(data, indent=4))


def main():
    config = load_config()
    run_context = init_run(config)
    run_dir = Path(run_context.run_dir)
    valid, reject, ingest_summary_list = ingest_incoming(
        run_context.run_id, config.paths.incoming, config.excel.extensions
    )

    valid.to_csv(run_dir / "valid.csv")
    reject.to_csv(run_dir / "reject.csv")
    publish_run(run_context, ingest_summary_list)


if __name__ == "__main__":
    main()
