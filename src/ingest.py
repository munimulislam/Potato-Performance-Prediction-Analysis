"""
@File - ingest.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from .sheet_validator import SheetValidationResult, SheetValidator
from .standardise import standardise_columns


@dataclass
class IngestResult:
    source_file_name: str
    source_sheet: str
    n_rows: int
    status: str
    error_message: str | None


class SheetUnprocessable(Exception):
    pass


def add_provenance(
    df: pd.DataFrame, run_id: str, filename: str, sheet_name: str
) -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    out["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["source_file_name"] = filename
    out["source_sheet"] = sheet_name
    out["source_row_number"] = df.index.astype(int) + 2

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


def split_valid_reject(
    df: pd.DataFrame, result: SheetValidationResult
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if result.ok:
        return df.copy(), pd.DataFrame(columns=list(df.columns) + ["error_message"])

    if result.has_schema_errors:
        raise SheetUnprocessable("|".join(result.schema_errors))

    is_bad = df.index.isin(result.error_row_indices)
    valid_df = df[~is_bad].copy()
    reject_df = df[is_bad].copy()
    reject_df["error_message"] = reject_df.index.map(result.row_errors)
    return valid_df, reject_df


def ingest_incoming(
    run_id: str, source_dir: str, file_extensions: list[str], sheet_name: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = discover_excel_files(source_dir, file_extensions)
    valid_df_list = []
    reject_df_list = []
    ingestion_results: list[IngestResult] = []

    for f in files:
        try:
            print(str(f))
            df = sheet_to_dataframe(str(f), sheet_name)
            df = standardise_columns(df)
            df = add_provenance(df, run_id, f.name, str(sheet_name))

            validation_result = SheetValidator().validate(df)
            valid_df, reject_df = split_valid_reject(df, validation_result)

            valid_df_list.append(valid_df)
            reject_df_list.append(reject_df)

            ingestion_results.append(
                IngestResult(f.name, str(sheet_name), len(df), "OK", None)
            )
        except Exception as e:
            print(e)
            ingestion_results.append(
                IngestResult(f.name, str(sheet_name), 0, "Error", str(e))
            )

    valid_dataframe = (
        pd.concat(valid_df_list, ignore_index=True, sort=False)
        if valid_df_list
        else pd.DataFrame()
    )
    reject_dataframe = (
        pd.concat(reject_df_list, ignore_index=True, sort=False)
        if reject_df_list
        else pd.DataFrame()
    )

    return valid_dataframe, reject_dataframe
