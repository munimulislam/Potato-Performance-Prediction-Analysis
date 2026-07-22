"""
@File - export_oa_snapshot.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .ml_config import load_ml_config


@dataclass(frozen=True)
class SnapshotMeta:
    relation: str
    duckdb_path: str
    exported_at_utc: str
    parquet_path: str
    n_rows: int
    columns: list[str]
    min_year: int | None
    max_year: int | None
    n_locations: int | None
    n_clones: int | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    try:
        return con.execute(sql).fetchone()[0]
    except Exception:
        return None


def export_snapshot(
    *, duckdb_path: Path, relation: str, out_parquet: Path, out_meta: Path | None
) -> SnapshotMeta:
    duckdb_path = duckdb_path.expanduser().resolve()
    out_parquet = out_parquet.expanduser().resolve()
    _ensure_parent(out_parquet)

    if out_meta is not None:
        out_meta = out_meta.expanduser().resolve()
        _ensure_parent(out_meta)

    try:
        con = duckdb.connect(str(duckdb_path), read_only=True)
    except TypeError:
        con = duckdb.connect(str(duckdb_path))

    try:
        con.execute(f"SELECT 1 FROM {relation} LIMIT 1;")

        escaped = str(out_parquet).replace("'", "''")
        con.execute(f"""
            COPY (SELECT * FROM {relation})
            TO '{escaped}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """)

        n_rows = con.execute(f"SELECT COUNT(*) FROM {relation};").fetchone()[0]
        cols = [
            r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {relation};").fetchall()
        ]

        min_year = _safe_scalar(con, f"SELECT MIN(year) FROM {relation};")
        max_year = _safe_scalar(con, f"SELECT MAX(year) FROM {relation};")
        n_locations = _safe_scalar(
            con, f"SELECT COUNT(DISTINCT location) FROM {relation};"
        )
        n_clones = _safe_scalar(con, f"SELECT COUNT(DISTINCT name1) FROM {relation};")

        meta = SnapshotMeta(
            relation=relation,
            duckdb_path=str(duckdb_path),
            exported_at_utc=_utc_now_iso(),
            parquet_path=str(out_parquet),
            n_rows=int(n_rows),
            columns=list(cols),
            min_year=int(min_year) if min_year is not None else None,
            max_year=int(max_year) if max_year is not None else None,
            n_locations=int(n_locations) if n_locations is not None else None,
            n_clones=int(n_clones) if n_clones is not None else None,
        )

        if out_meta is not None:
            out_meta.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

        return meta

    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export OA modelling dataset snapshot from DuckDB using config."
    )
    p.add_argument(
        "--config", default="src/ml/config/ml.yaml", help="Path to ML config yaml."
    )

    p.add_argument("--duckdb-path", default=None, help="Override DuckDB path.")
    p.add_argument(
        "--relation", default=None, help="Override relation, e.g. gold.o_a_mart."
    )
    p.add_argument("--out-parquet", default=None, help="Override output parquet path.")
    p.add_argument(
        "--out-meta",
        default=None,
        help="Override output metadata JSON path ('' to disable).",
    )

    args = p.parse_args()

    cfg = load_ml_config(args.config)

    duckdb_path = Path(args.duckdb_path) if args.duckdb_path else Path(cfg.duckdb.path)
    relation = args.relation if args.relation else cfg.relations.oa_mart
    out_parquet = (
        Path(args.out_parquet)
        if args.out_parquet
        else Path(cfg.outputs.oa_mart_parquet)
    )

    if args.out_meta == "":
        out_meta = None
    else:
        out_meta = (
            Path(args.out_meta) if args.out_meta else Path(cfg.outputs.oa_mart_meta)
        )

    meta = export_snapshot(
        duckdb_path=duckdb_path,
        relation=relation,
        out_parquet=out_parquet,
        out_meta=out_meta,
    )

    print("Snapshot export complete:")
    print(json.dumps(asdict(meta), indent=2))


if __name__ == "__main__":
    main()
