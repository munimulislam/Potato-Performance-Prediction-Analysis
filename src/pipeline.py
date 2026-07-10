"""
@File - pipeline.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from .key import add_business_key_column, add_row_hash_column, split_duplicate_key_rows

from .log_context import get_run_id, set_run_id

from .config import load_config
from .run_context import RunContext, init_run
from .ingest import IngestBatchResult, SheetIngestResult, ingest_batch
from pathlib import Path
import json
from dataclasses import asdict
import logging
from .log_config import configure_logging
import pandas as pd


def get_ingest_summary(context: RunContext, result: IngestBatchResult) -> dict:
    out = asdict(context)
    out["n_sheet_processed"] = result.n_sheet_processed
    out["n_sheet_with_error"] = result.n_sheet_with_error
    out["sheets"] = []

    for sr in result.sheet_results:
        sr_dict = {
            k: v for k, v in asdict(sr).items() if k not in ["valid_df", "reject_df"]
        }
        out["sheets"].append(sr_dict)

    return out


def append_duplicate_rows_to_reject(
    duplicates_df: pd.DataFrame, reject_df: pd.DataFrame
):
    if not duplicates_df.empty:
        rejects = (
            pd.concat([duplicates_df, reject_df], ignore_index=True, sort=False)
            if not reject_df.empty
            else duplicates_df
        )


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

    batch_ingest_result = ingest_batch(
        run_context.run_id, config.paths.incoming, config.excel.extensions
    )

    summary = get_ingest_summary(run_context, batch_ingest_result)
    (run_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=4))

    if batch_ingest_result.valid_df.empty:
        logger.warning("PIPELINE | no valid rows; nothing to load")
    else:
        add_business_key_column(
            batch_ingest_result.valid_df, config.business_key.columns
        )
        add_row_hash_column(batch_ingest_result.valid_df)
        valid_df, duplicates_df = split_duplicate_key_rows(batch_ingest_result.valid_df)
        batch_ingest_result.valid_df = valid_df
        append_duplicate_rows_to_reject(duplicates_df, batch_ingest_result.reject_df)

        batch_ingest_result.valid_df.to_csv(run_dir / "valid.csv")
        batch_ingest_result.reject_df.to_csv(run_dir / "reject.csv")


if __name__ == "__main__":
    main()
