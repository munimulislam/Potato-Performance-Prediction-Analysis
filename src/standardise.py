"""
@File - standardise.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

import pandas as pd
import re
from dataclasses import dataclass
from .schema import ALWAYS_DROP_COLS

COLUMN_ALIASES: dict[str, str] = {}


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
    c = c.replace("%", "parcent")
    c = c.replace("#", "hash")

    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")

    if not c or not c[0].isalpha():
        c = f"col_{c}" if c else "col"

    return c


def standardise_columns(df: pd.DataFrame) -> StandardisationResult:
    df = df.copy()

    original_col_names = list(df.columns)
    clean_col_names = [clean_column_name(c) for c in df.columns]
    final_col_names = [COLUMN_ALIASES.get(c, c) for c in clean_col_names]
    df.columns = final_col_names

    rename_map = {
        str(original): str(final)
        for original, final in zip(original_col_names, final_col_names)
        if str(original) != str(final)
    }

    drop_cols = ALWAYS_DROP_COLS
    keep_cols = [c for c in df.columns if c not in drop_cols]

    df = df[keep_cols]

    result = StandardisationResult(df, rename_map, drop_cols)

    return result
