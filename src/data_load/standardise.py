"""
@File - standardise.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

import pandas as pd
import re
from dataclasses import dataclass
from collections import Counter


@dataclass
class StandardisationResult:
    dataframe: pd.DataFrame
    column_rename_map: dict[str, str]
    dropped_columns: list[str]


def clean_column_name(name: str) -> str:
    c = str(name).strip().lower()
    c = re.sub(r"\s+", " ", c)

    c = c.replace(">=", "gte")
    c = c.replace("<=", "lte")
    c = c.replace(">", "gt")
    c = c.replace("<", "lt")
    c = c.replace("%", "percent")
    c = c.replace("#", "hash")

    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")

    if not c or not c[0].isalpha():
        c = f"col_{c}" if c else "col"

    return c


def get_duplicate_column_names(column_names: list[str]) -> list[str] | None:
    if len(column_names) != len(set(column_names)):
        duplicates = [c for c, n in Counter(column_names).items() if n > 1]
        return duplicates
    return None


def standardise_columns(
    df: pd.DataFrame,
    drop_columns: list[str] | None = None,
    column_name_aliases: dict[str, str] | None = None,
) -> StandardisationResult:
    df = df.copy()
    drop_columns = list(drop_columns or [])
    column_name_aliases = dict(column_name_aliases or {})

    original_col_names = list(df.columns)
    clean_col_names = [clean_column_name(c) for c in df.columns]
    final_col_names = [column_name_aliases.get(c, c) for c in clean_col_names]

    duplicate_columns = get_duplicate_column_names(final_col_names)
    if duplicate_columns:
        raise ValueError(
            f"Duplicate column names after standardisation: {duplicate_columns}"
        )

    df.columns = final_col_names

    rename_map = {
        str(original): str(final)
        for original, final in zip(original_col_names, final_col_names)
        if str(original) != str(final)
    }

    drop_cols = [c for c in df.columns if c in drop_columns]
    keep_cols = [c for c in df.columns if c not in drop_cols]

    df = df[keep_cols]

    result = StandardisationResult(df, rename_map, drop_cols)

    return result
