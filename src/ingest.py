"""
@File - ingest.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

from .schema import ERROR_COLUMN_NAME, get_unknown_cols
from .sheet_validator import SheetValidationResult, SheetValidator
from .standardise import standardise_columns


@dataclass
class IngestBatchResult:
    n_sheet_processed: int
    n_sheet_with_error: int
    sheet_results: list[SheetIngestResult]
    valid_df: pd.DataFrame
    reject_df: pd.DataFrame


@dataclass
class SheetIngestResult:
    source_file_name: str
    source_sheet: str
    n_rows: int
    n_rows_rejected: int
    n_rows_accepted: int
    status: str
    sheet_error: str | None
    n_dropped_cols: int
    n_empty_cols: int
    n_unknown_cols: int
    dropped_cols: list[str] | None
    unknown_cols: list[str] | None
    empty_cols: list[str] | None
    rename_map: dict[str, str] | None
    valid_df: pd.DataFrame | None
    reject_df: pd.DataFrame | None


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


def split_valid_reject_dataframe(
    df: pd.DataFrame, result: SheetValidationResult
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if result.ok:
        return df.copy(), pd.DataFrame(columns=list(df.columns) + [ERROR_COLUMN_NAME])

    if result.has_schema_errors:
        raise SheetUnprocessable("|".join(result.schema_errors))

    is_bad = df.index.isin(result.error_row_indices)
    valid_df = df[~is_bad].copy()
    reject_df = df[is_bad].copy()
    reject_df["error_message"] = reject_df.index.map(result.row_errors)
    return valid_df, reject_df


def ingest_sheet(file_path: str, run_id: str, sheet_name: int = 0):
    try:
        df = sheet_to_dataframe(file_path, sheet_name)
        std_result = standardise_columns(df)
        df = std_result.dataframe
        df = add_provenance(df, run_id, Path(file_path).name, str(sheet_name))

        empty_cols = df.columns[df.isna().all()].tolist()
        dropped_cols = std_result.dropped_columns
        unknown_cols = get_unknown_cols(df)

        validation_result = SheetValidator().validate(df)
        valid_df, reject_df = split_valid_reject_dataframe(df, validation_result)

        return SheetIngestResult(
            source_file_name=Path(file_path).name,
            source_sheet=str(sheet_name),
            n_rows=len(df),
            n_rows_rejected=len(reject_df),
            n_rows_accepted=len(valid_df),
            status="OK",
            sheet_error=None,
            n_dropped_cols=len(dropped_cols),
            n_empty_cols=len(empty_cols),
            n_unknown_cols=len(unknown_cols),
            dropped_cols=dropped_cols,
            unknown_cols=unknown_cols,
            rename_map=std_result.column_rename_map,
            empty_cols=empty_cols,
            valid_df=valid_df,
            reject_df=reject_df,
        )

    except SheetUnprocessable as e:
        return SheetIngestResult(
            source_file_name=Path(file_path).name,
            source_sheet=str(sheet_name),
            n_rows=0,
            n_rows_rejected=0,
            n_rows_accepted=0,
            status="SHEET_ERROR",
            sheet_error=str(e),
            n_dropped_cols=0,
            n_empty_cols=0,
            n_unknown_cols=0,
            dropped_cols=None,
            unknown_cols=None,
            rename_map=None,
            empty_cols=None,
            valid_df=None,
            reject_df=None,
        )


def ingest_batch(
    run_id: str, source_dir: str, file_extensions: list[str], sheet_name: int = 0
) -> IngestBatchResult:
    files = discover_excel_files(source_dir, file_extensions)
    valid_df_list: list[pd.DataFrame] = []
    reject_df_list: list[pd.DataFrame] = []
    sheet_result_list: list[SheetIngestResult] = []

    for file in files:
        sheet_result = ingest_sheet(str(file), run_id)
        valid = sheet_result.valid_df
        reject = sheet_result.reject_df

        if valid is not None:
            valid_df_list.append(valid)

        if reject is not None:
            reject_df_list.append(reject)

        sheet_result_list.append(sheet_result)

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

    result = IngestBatchResult(
        n_sheet_processed=len(sheet_result_list),
        n_sheet_with_error=len(
            [r for r in sheet_result_list if r.sheet_error is not None]
        ),
        valid_df=valid_dataframe,
        reject_df=reject_dataframe,
        sheet_results=sheet_result_list,
    )

    return result
