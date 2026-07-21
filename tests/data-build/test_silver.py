"""
@File - test_silver.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 19/07/2026
"""

import duckdb

_DB = "data/trials.duckdb"
_KEY = "(year, location, experiment_name, name1, plot)"


def test_silver_matches_bronze_current():
    con = duckdb.connect(_DB, read_only=True)
    try:
        silver = con.execute("SELECT count(*) FROM main.silver_trials").fetchone()[0]
        bronze = con.execute(
            f"SELECT count(DISTINCT {_KEY}) "
            "FROM trial_data.valid WHERE _dlt_valid_to IS NULL"
        ).fetchone()[0]
    finally:
        con.close()
    assert (
        silver == bronze
    ), f"silver has {silver} rows but bronze has {bronze} distinct current "


def test_no_unknown_environments():
    con = duckdb.connect(_DB, read_only=True)
    try:
        n = con.execute(
            "SELECT count(*) FROM main.silver_trials WHERE env_type = 'UNKNOWN'"
        ).fetchone()[0]
    finally:
        con.close()
    assert n == 0, f"{n} rows have unmapped locations — add them to location_env.csv"
