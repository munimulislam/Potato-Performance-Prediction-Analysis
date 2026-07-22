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


def export_snapshot(
    *,
    duckdb_path: Path,
    relation: str,
    out_parquet: Path,
    out_meta: Path | None,
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

        con.execute(
            f"""
            COPY (
                SELECT * FROM {relation}
            )
            TO '{str(out_parquet).replace("'", "''")}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        n_rows = con.execute(f"SELECT COUNT(*) FROM {relation};").fetchone()[0]

        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {relation};").fetchall()]

        def safe_scalar(sql: str) -> Any:
            try:
                return con.execute(sql).fetchone()[0]
            except Exception:
                return None

        min_year = safe_scalar(f"SELECT MIN(year) FROM {relation};")
        max_year = safe_scalar(f"SELECT MAX(year) FROM {relation};")
        n_locations = safe_scalar(f"SELECT COUNT(DISTINCT location) FROM {relation};")
        n_clones = safe_scalar(f"SELECT COUNT(DISTINCT name1) FROM {relation};")

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
    p = argparse.ArgumentParser(description="Export DVC snapshot for OA modelling dataset from DuckDB.")
    p.add_argument("--duckdb-path", required=True, help="Path to DuckDB file (same one used by dlt/dbt).")
    p.add_argument("--relation", default="gold.o_a_mart", help='DuckDB relation, e.g. "gold.o_a_mart".')
    p.add_argument("--out-parquet", default="data/gold/o_a_mart.parquet", help="Output parquet path.")
    p.add_argument("--out-meta", default="data/gold/o_a_mart.meta.json", help="Output metadata JSON path.")
    args = p.parse_args()

    meta = export_snapshot(
        duckdb_path=Path(args.duckdb_path),
        relation=args.relation,
        out_parquet=Path(args.out_parquet),
        out_meta=Path(args.out_meta) if args.out_meta else None,
    )

    print("Snapshot export complete:")
    print(json.dumps(asdict(meta), indent=2))


if __name__ == "__main__":
    main()