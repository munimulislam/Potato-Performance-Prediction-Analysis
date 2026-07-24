"""
@File - exporter.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 22/07/2026
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
import duckdb


@dataclass(frozen=True)
class DataSnapshotExportConfig:
    db_path: str
    schema: str
    tables: dict[str, str]
    out_folder: str


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


def load_config(config_path: str) -> DataSnapshotExportConfig:
    path = Path(f"{config_path}")
    root = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(root, dict):
        raise ValueError(f"Expected root to be a dict in config")

    return DataSnapshotExportConfig(
        db_path=root["db_path"],
        schema=root["schema"],
        tables=root["tables"],
        out_folder=root["out_folder"],
    )


def export_snapshot(
    *, db_path: Path, schema: str, table: str, out_folder: Path
) -> SnapshotMeta:
    out_parquret = out_folder / table / f"{table}.parquet"
    out_parquret = out_parquret.expanduser().resolve()

    out_meta = out_folder / table / f"{table}.meta.json"
    out_meta = out_meta.expanduser().resolve()

    db_path = db_path.expanduser().resolve()

    _ensure_parent(out_parquret)
    _ensure_parent(out_meta)

    relation = f"{schema}.{table}"
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        escaped = str(out_parquret).replace("'", "''")
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
            duckdb_path=str(db_path),
            exported_at_utc=_utc_now_iso(),
            parquet_path=str(out_parquret),
            n_rows=int(n_rows),
            columns=list(cols),
            min_year=int(min_year) if min_year is not None else None,
            max_year=int(max_year) if max_year is not None else None,
            n_locations=int(n_locations) if n_locations is not None else None,
            n_clones=int(n_clones) if n_clones is not None else None,
        )

        out_meta.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

        return meta

    finally:
        con.close()


def main() -> None:
    cfg = load_config("src/data_export/config.yaml")

    p = argparse.ArgumentParser(description="Export dataset snapshot from Datasource.")
    p.add_argument("--db_path", default=None)
    p.add_argument("--schema", default=None)
    p.add_argument("--table", required=True, choices=cfg.tables.keys())
    p.add_argument("--out_folder", default=None)

    args = p.parse_args()

    db_path = Path(args.db_path) if args.db_path else Path(cfg.db_path)
    schema = args.schema if args.schema else cfg.schema
    table = cfg.tables[args.table]
    out_folder = Path(args.out_folder) if args.out_folder else Path(cfg.out_folder)

    meta = export_snapshot(
        db_path=db_path,
        schema=schema,
        table=table,
        out_folder=out_folder,
    )

    print("OA Snapshot export complete:")
    print(json.dumps(asdict(meta), indent=2))


if __name__ == "__main__":
    main()
