"""
@File - standardise.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

import pandas as pd
import re

from .schema import ALWAYS_DROP_COLS

COLUMN_ALIASES: dict[str, str] = {}


def clean_column_name(name: str) -> str:
    c = str(name).strip().lower()
    c = re.sub(r"\s+", " ", c)
    c = (
        c.replace("lenght", "length")
        .replace(" (", "(")
        .replace(" )", ")")
        .replace(" %", "%")
        .replace(" #", "#")
        .replace(" ", "_")
    )

    return c


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    clean_col_names = [clean_column_name(c) for c in df.columns]
    final_col_names = [COLUMN_ALIASES.get(c, c) for c in clean_col_names]
    df.columns = final_col_names

    keep_cols = [c for c in df.columns if c not in ALWAYS_DROP_COLS]

    df = df[keep_cols]

    return df
