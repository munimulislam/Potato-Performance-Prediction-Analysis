"""
@File - key.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 10/07/2026
"""

import logging
import hashlib
import pandas as pd
from schema import ERROR_COLUMN_NAME, get_schema_columns

logger = logging.getLogger(__name__)


def _normalize_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def add_business_key_column(df: pd.DataFrame, key_columns: list[str]):
    missing_key_cols = [c for c in key_columns if c not in df.columns]
    if missing_key_cols:
        raise ValueError(f"key columns absent in data: {missing_key_cols}")

    key_columns = sorted(key_columns)
    df["business_key"] = df.apply(
        lambda r: _digest("|".join([_normalize_value(r[c]) for c in key_columns])),
        axis=1,
    )


def add_row_hash_column(df: pd.DataFrame, exclude_columns: list[str] = []):
    cols = sorted([c for c in get_schema_columns() if c not in exclude_columns])

    df["row_hash"] = df.apply(
        lambda r: _digest("|".join([_normalize_value(r[c]) for c in cols])),
        axis=1,
    )
    return df


def split_duplicate_key_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "business_key" not in df.columns:
        raise ValueError("business key not present")

    duplicate_mask = df.duplicated("business_key", keep=False)
    if not duplicate_mask.any():
        return df, df.iloc[0:0].copy()

    duplicates = df[duplicate_mask].copy()
    unique = df[~duplicate_mask].copy()

    logger.warning(
        "DUPLICATE_KEY | %d rows across %d colliding keys — business_key is not unique",
        len(duplicates),
        duplicates["business_key"].nunique(),
    )
    return unique, duplicates
