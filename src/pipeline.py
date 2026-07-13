"""
@File - pipeline.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from .duckdb_repository import DuckDBTrialsRepository, db_session
from .repository import RunRecord, SheetError
from .schema import get_schema_version

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
) -> pd.DataFrame:
    if duplicates_df.empty:
        return reject_df
    if reject_df.empty:
        return duplicates_df
    return pd.concat([duplicates_df, reject_df], ignore_index=True, sort=False)


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
        logger.warning("PIPELINE | no valid rows")
    else:
        batch_ingest_result.valid_df = add_business_key_column(
            batch_ingest_result.valid_df, config.business_key.columns
        )
        batch_ingest_result.valid_df = add_row_hash_column(batch_ingest_result.valid_df)
        valid_df, duplicates_df = split_duplicate_key_rows(batch_ingest_result.valid_df)
        batch_ingest_result.valid_df = valid_df

        if not duplicates_df.empty:
            batch_ingest_result.reject_df = append_duplicate_rows_to_reject(
                duplicates_df, batch_ingest_result.reject_df
            )

        with db_session(config.paths.duckdb) as con:
            repo = DuckDBTrialsRepository(con)
            repo.initialise()

            counts = repo.append_valids(
                batch_ingest_result.valid_df, run_context.run_id, get_schema_version()
            )
            n_rejected = repo.append_rejects(
                batch_ingest_result.reject_df, run_context.run_id
            )
            sheet_errors = [
                SheetError(s.source_file_name, s.source_sheet, s.sheet_error)
                for s in batch_ingest_result.sheet_results
                if s.sheet_error is not None
            ]
            n_sheet_errors = repo.append_sheet_errors(sheet_errors, run_context.run_id)

            repo.append_run(
                RunRecord(
                    run_id=run_context.run_id,
                    started_at_utc=run_context.start_time_utc,
                    status="OK",
                    n_files=len(batch_ingest_result.sheet_results),
                    n_rows_read=sum(
                        r.n_rows for r in batch_ingest_result.sheet_results
                    ),
                    n_rows_accepted=len(batch_ingest_result.valid_df),
                    n_rows_rejected=n_rejected,
                    n_sheet_errors=n_sheet_errors,
                    n_versions_inserted=counts.inserted,
                    n_versions_skipped_existing=counts.skipped_existing,
                )
            )

            print(con.execute("""
                SELECT run_id, n_versions_inserted, n_versions_skipped_existing
                FROM raw.pipeline_runs ORDER BY run_id
            """).fetchdf())

            print(
                con.execute("SELECT count(*) FROM raw.trials_rows_current").fetchone()
            )
            print(
                con.execute(
                    "SELECT count(DISTINCT business_key) FROM raw.trials_rows_versions"
                ).fetchone()
            )

        batch_ingest_result.valid_df.to_csv(run_dir / "valid.csv")
        batch_ingest_result.reject_df.to_csv(run_dir / "reject.csv")


if __name__ == "__main__":
    main()
