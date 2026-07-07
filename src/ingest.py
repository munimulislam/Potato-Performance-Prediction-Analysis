"""
@File - ingest.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from collections import Counter
from .standardise import standardise_columns


class DuplicateColumnsError(Exception):
    def __init__(self, duplicates: list[str]):
        self.duplicates = duplicates
        msg = f"Duplicate column names: {self.duplicates}"
        super().__init__(msg)


@dataclass
class IngestResult:
    source_file_name: str
    source_sheet: str
    n_rows: int
    status: str
    error_message: str | None


def add_provenance(df: pd.DataFrame, provenance_dict: dict) -> pd.DataFrame:
    out = df.copy()
    for k, v in provenance_dict.items():
        out[k] = v
    return out


def discover_excel_files(source_dir: str, file_extensions: list[str]) -> list[Path]:
    incoming_dir = Path(source_dir)
    extensions = {e.lower() for e in file_extensions}
    files = [
        f
        for f in incoming_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return sorted(files)


def sheet_to_dataframe(file_path: str, sheet_name: int = 0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


def check_duplicate_columns(df: pd.DataFrame):
    counts = Counter(df.columns)
    duplicates = sorted([c for c, n in counts.items() if n > 1])
    return duplicates


def ingest_incoming(
    run_id: str, source_dir: str, file_extensions: list[str], sheet_name: int = 0
):
    files = discover_excel_files(source_dir, file_extensions)
    df_list: list[pd.DataFrame] = []
    results: list[IngestResult] = []

    for f in files:
        try:
            df = sheet_to_dataframe(str(f), sheet_name)
            df = standardise_columns(df)

            duplicates = check_duplicate_columns(df)
            if duplicates:
                raise DuplicateColumnsError(duplicates)

            provenance = {
                "run_id": run_id,
                "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_file_name": f.name,
                "source_sheet": sheet_name,
                "source_row_number": df.index.astype(int) + 2,
            }

            df = add_provenance(df, provenance)
            df_list.append(df)
            results.append(IngestResult(f.name, str(sheet_name), len(df), "OK", None))
        except Exception as e:
            results.append(IngestResult(f.name, str(sheet_name), 0, "Error", str(e)))

    for i, df in enumerate(df_list):
        if not df.index.is_unique:
            print(f"DataFrame at index {i} has duplicate ROW indices.")
        if not df.columns.is_unique:
            print(f"DataFrame at index {i} has duplicate COLUMN names.")

    dataframes = (
        pd.concat(df_list, ignore_index=True, sort=False) if df_list else pd.DataFrame()
    )

    return dataframes
