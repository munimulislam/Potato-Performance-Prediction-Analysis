"""
@File - data.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

from pathlib import Path

import duckdb
import pandas as pd

from .ml_config import MlConfig


def load_oa_dataframe(cfg: MlConfig) -> pd.DataFrame:
    if cfg.dataset.source == "parquet":
        p = Path(cfg.dataset.path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Parquet snapshot not found: {p}")
        return pd.read_parquet(p)

    db_path = Path(cfg.duckdb.path).expanduser().resolve()
    con = duckdb.connect(str(db_path))

    try:
        return con.execute(f"SELECT * FROM {cfg.relations.oa_mart};").fetchdf()
    finally:
        con.close()
