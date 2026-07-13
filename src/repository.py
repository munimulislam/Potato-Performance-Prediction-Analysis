"""
@File - repository.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 10/07/2026
"""

from typing import Protocol
import pandas as pd
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class InsertResult:
    incoming: int
    inserted: int
    skipped_existing: int


@dataclass(frozen=True)
class SheetError:
    source_file_name: str
    source_sheet: str
    error_message: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    started_at_utc: str
    status: str
    n_files: int
    n_rows_read: int
    n_rows_accepted: int
    n_rows_rejected: int
    n_sheet_errors: int
    n_versions_inserted: int
    n_versions_skipped_existing: int


class TrialsRepository(Protocol):
    def initialise(self) -> None: ...

    def append_valids(
        self, rows: pd.DataFrame, run_id: str, schema_version: str
    ) -> InsertResult: ...

    def append_rejects(self, rejects: pd.DataFrame, run_id: str) -> int: ...

    def append_sheet_errors(self, errors: list[SheetError], run_id: str) -> int: ...

    def append_run(self, run) -> None: ...

    def current_rows(self) -> pd.DataFrame: ...
