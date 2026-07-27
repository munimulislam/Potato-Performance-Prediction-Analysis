"""
@File - data.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from pathlib import Path
import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Parquet snapshot not found: {p}")
    return pd.read_parquet(p)
