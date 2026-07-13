"""
@File - duckdb_repository.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 11/07/2026
@Description - Description of the file.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import re

import duckdb
import pandas as pd
from pathlib import Path

from .repository import InsertResult, RunRecord, SheetError

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")

CREATE_RAW_SCHEMA = "CREATE SCHEMA IF NOT EXISTS raw;"
CREATE_QUANRANTINE_SCHEMA = "CREATE SCHEMA IF NOT EXISTS quarantine;"
CREATE_VERSION_SEQ = "CREATE SEQUENCE IF NOT EXISTS raw.version_seq START 1;"
CREATE_TRIAL_DATA_TABLE = """
    CREATE TABLE IF NOT EXISTS raw.trials_rows_versions (
        business_key      VARCHAR   NOT NULL,
        row_hash          VARCHAR   NOT NULL,
        schema_version    VARCHAR   NOT NULL,
        version_ts_utc    TIMESTAMP NOT NULL,
        version_seq       BIGINT    NOT NULL,
        run_id            VARCHAR   NOT NULL,
        source_file_name  VARCHAR,
        source_sheet      VARCHAR,
        source_row_number INTEGER
    );
"""
UNIQUE_INDEX_VERSION = """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_trials_versions
    ON raw.trials_rows_versions (business_key, row_hash, schema_version);
"""

CREATE_CURRENT_TRIAL_DATA_VIEW = """
    CREATE OR REPLACE VIEW raw.trials_rows_current AS
    SELECT * EXCLUDE (rn) FROM (
        SELECT *, row_number() OVER (
            PARTITION BY business_key
            ORDER BY version_ts_utc DESC, version_seq DESC
        ) AS rn
        FROM raw.trials_rows_versions
    ) WHERE rn = 1;
"""

_CREATE_REJECT_ROWS_TABLE = """
    CREATE TABLE IF NOT EXISTS quarantine.reject_rows (
        run_id            VARCHAR,
        quarantined_at    TIMESTAMP,
        source_file_name  VARCHAR,
        source_sheet      VARCHAR,
        source_row_number INTEGER,
        _error            VARCHAR
    );
"""

CREATE_SHEET_ERROR_TABLE = """
    CREATE TABLE IF NOT EXISTS quarantine.sheet_errors (
        run_id           VARCHAR,
        quarantined_at   TIMESTAMP,
        source_file_name VARCHAR,
        source_sheet     VARCHAR,
        error_message    VARCHAR
    );
"""

CREATE_RUN_TABLE = """
    CREATE TABLE IF NOT EXISTS raw.pipeline_runs (
        run_id                      VARCHAR PRIMARY KEY,
        started_at_utc              VARCHAR,
        finished_at_utc             TIMESTAMP,
        status                      VARCHAR,
        n_files                     INTEGER,
        n_rows_read                 INTEGER,
        n_rows_accepted             INTEGER,
        n_rows_rejected             INTEGER,
        n_sheet_errors              INTEGER,
        n_versions_inserted         INTEGER,
        n_versions_skipped_existing INTEGER
    );
"""

_INSERT_RUN = """
    INSERT INTO raw.pipeline_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT (run_id) DO UPDATE SET
        finished_at_utc = excluded.finished_at_utc,
        status = excluded.status,
        n_files = excluded.n_files,
        n_rows_read = excluded.n_rows_read,
        n_rows_accepted = excluded.n_rows_accepted,
        n_rows_rejected = excluded.n_rows_rejected,
        n_sheet_errors = excluded.n_sheet_errors,
        n_versions_inserted = excluded.n_versions_inserted,
        n_versions_skipped_existing = excluded.n_versions_skipped_existing;
"""


@contextmanager
def db_session(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    try:
        yield con
    finally:
        con.close()


def _assert_safe_identifier(name: str) -> None:
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"unsafe column name for DDL: {name!r}")


class DuckDBTrialsRepository:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def initialise(self) -> None:
        con = self._con
        con.execute(CREATE_RAW_SCHEMA)
        con.execute(CREATE_QUANRANTINE_SCHEMA)
        con.execute(CREATE_VERSION_SEQ)
        con.execute(CREATE_TRIAL_DATA_TABLE)

        try:
            con.execute(UNIQUE_INDEX_VERSION)
        except Exception as exc:
            logger.warning(
                "DuckDB | unique index not created (%s) — relying on anti-join", exc
            )

        con.execute(CREATE_CURRENT_TRIAL_DATA_VIEW)
        con.execute(CREATE_SHEET_ERROR_TABLE)
        con.execute(_CREATE_REJECT_ROWS_TABLE)
        con.execute(CREATE_RAW_SCHEMA)

        logger.info("STORE | schema initialised")

    def _existing_columns(self, schema: str, table: str) -> set[str]:
        rows = self._con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ?",
            [schema, table],
        ).fetchall()
        return {r[0] for r in rows}

    def _evolve_columns(self, schema: str, table: str, df: pd.DataFrame) -> None:
        existing = self._existing_columns(schema, table)
        for col in df.columns:
            if col in existing:
                continue

            _assert_safe_identifier(col)

            self._con.execute(
                f'ALTER TABLE {schema}.{table} ADD COLUMN "{col}" VARCHAR;'
            )
            logger.info("STORE | %s.%s: added column %s", schema, table, col)

    def append_valids(
        self, rows: pd.DataFrame, run_id: str, schema_version: str
    ) -> InsertResult:
        if rows.empty:
            return InsertResult(0, 0, 0)

        con = self._con
        staged = rows.copy()
        staged["schema_version"] = schema_version
        staged["run_id"] = run_id
        staged["version_ts_utc"] = datetime.now(timezone.utc)
        staged["version_seq"] = [
            con.execute("SELECT nextval('raw.version_seq')").fetchone()[0]
            for _ in range(len(staged))
        ]

        self._evolve_columns("raw", "trials_rows_versions", staged)
        con.register("incoming", staged)
        try:
            skipped = con.execute("""
                SELECT count(*) FROM incoming i
                JOIN raw.trials_rows_versions v
                  ON v.business_key = i.business_key
                 AND v.row_hash = i.row_hash
                 AND v.schema_version = i.schema_version
            """).fetchone()[0]

            cols = ", ".join(f'"{c}"' for c in staged.columns)
            con.execute(f"""
                INSERT INTO raw.trials_rows_versions ({cols})
                SELECT {cols}
                FROM incoming i
                WHERE NOT EXISTS (
                    SELECT 1 FROM raw.trials_rows_versions v
                    WHERE v.business_key = i.business_key
                      AND v.row_hash = i.row_hash
                      AND v.schema_version = i.schema_version
                );
            """)
            inserted = len(staged) - skipped
        finally:
            con.unregister("incoming")

        counts = InsertResult(len(staged), inserted, skipped)
        logger.info(
            "VERSIONS | incoming=%d inserted=%d skipped_existing=%d",
            counts.incoming,
            counts.inserted,
            counts.skipped_existing,
        )
        return counts

    def append_rejects(self, rejects: pd.DataFrame, run_id: str) -> int:
        if rejects.empty:
            return 0

        con = self._con
        staged = rejects.copy()
        staged["run_id"] = run_id
        staged["quarantined_at"] = datetime.now(timezone.utc)

        self._evolve_columns("quarantine", "reject_rows", staged)

        con.register("q_in", staged)
        try:
            con.execute(
                "INSERT INTO quarantine.reject_rows BY NAME SELECT * FROM q_in;"
            )
        finally:
            con.unregister("q_in")

        logger.info("QUARANTINE | appended %d reject row(s)", len(staged))
        return len(staged)

    def append_sheet_errors(self, errors: list[SheetError], run_id: str) -> int:
        if not errors:
            return 0
        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "quarantined_at": now,
                    "source_file_name": e.source_file_name,
                    "source_sheet": e.source_sheet,
                    "error_message": e.error_message,
                }
                for e in errors
            ]
        )

        self._con.register("se_in", df)
        try:
            self._con.execute(
                "INSERT INTO quarantine.sheet_errors BY NAME SELECT * FROM se_in;"
            )
        finally:
            self._con.unregister("se_in")
        logger.warning("QUARANTINE | %d sheet(s) unprocessable", len(errors))
        return len(errors)

    def append_run(self, record: RunRecord) -> None:
        self._con.execute(
            """
            INSERT INTO raw.pipeline_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (run_id) DO UPDATE SET
                finished_at_utc = excluded.finished_at_utc,
                status = excluded.status,
                n_files = excluded.n_files,
                n_rows_read = excluded.n_rows_read,
                n_rows_accepted = excluded.n_rows_accepted,
                n_rows_rejected = excluded.n_rows_rejected,
                n_sheet_errors = excluded.n_sheet_errors,
                n_versions_inserted = excluded.n_versions_inserted,
                n_versions_skipped_existing = excluded.n_versions_skipped_existing;
        """,
            [
                record.run_id,
                record.started_at_utc,
                datetime.now(timezone.utc),
                record.status,
                record.n_files,
                record.n_rows_read,
                record.n_rows_accepted,
                record.n_rows_rejected,
                record.n_sheet_errors,
                record.n_versions_inserted,
                record.n_versions_skipped_existing,
            ],
        )
        logger.info("AUDIT | recorded run %s (%s)", record.run_id, record.status)

    def current_rows(self) -> pd.DataFrame:
        return self._con.execute("SELECT * FROM raw.trials_rows_current").fetchdf()
