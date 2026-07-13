"""
@File - schema.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
@Description - Description of the file.
"""

import hashlib

import pandera.pandas as pa
import pandas as pd

ALWAYS_DROP_COLS = [
    "archived",
    "barcode_39",
    "boxnos",
    "block",
    "bn",
    "cross_name",
    "check",
    "clone",
    "cone",
    "description1",
    "entry",
    "female",
    "female_pedigree",
    "for_yield",
    "fn",
    "filler",
    "images",
    "identity",
    "harvester",
    "loc",
    "locn",
    "male",
    "male_pedigree",
    "method",
    "original_name",
    "pass",
    "planter",
    "plot_range",
    "plot_size_uk",
    "plot_column",
    "selection",
    "tyduk",
    "yearfact",
]

PROVENANCE_COLS = [
    "run_id",
    "ingested_at_utc",
    "source_file_name",
    "source_sheet",
    "source_row_number",
]

ERROR_COLUMN_NAME = ["error_message"]

EMPTY_STRING_CHECK = pa.Check(
    lambda s: len(s) > 0,
    element_wise=True,
    error="Empty String",
)

NUMBER_0_9_RANGE_CHECK = pa.Check(
    lambda x: x >= 0.0 and x <= 9.0,
    element_wise=True,
    error="value must be between 0 - 9",
)

NUMBER_NON_NEGATIVE_CHECK = pa.Check(
    lambda x: x > 0, element_wise=True, error="value must be positive number"
)


def build_sheet_schema() -> pa.DataFrameSchema:
    schema = pa.DataFrameSchema(
        name="trial_data_sheet",
        title="trial_data_sheet",
        strict=False,
        coerce=True,
        unique_column_names=True,
        report_duplicates="all",
        columns={
            "appearance": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "acb": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "acb24": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "comments": pa.Column(
                dtype=str,
                required=True,
                nullable=True,
                coerce=True,
            ),
            "crisp": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "coflscol": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "chipcolour": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "cook_com": pa.Column(
                dtype=str,
                required=True,
                nullable=True,
                coerce=True,
            ),
            "dryness": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "drymatter": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "dsntgrtn": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "eveness": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "experiment_name": pa.Column(
                dtype=str,
                required=True,
                nullable=False,
                coerce=True,
                checks=[EMPTY_STRING_CHECK],
            ),
            "eyedepth": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "ffdefects": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "ffscab": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "ffhollowh": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "ff_irs": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "fleshcolou": pa.Column(
                dtype=str,
                required=True,
                nullable=True,
                coerce=True,
            ),
            "flavour": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "hunterl": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_NON_NEGATIVE_CHECK],
            ),
            "huntera": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_NON_NEGATIVE_CHECK],
            ),
            "hunterb": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_NON_NEGATIVE_CHECK],
            ),
            "location": pa.Column(
                dtype=str,
                required=True,
                nullable=False,
                coerce=True,
                checks=[EMPTY_STRING_CHECK],
            ),
            "mealiness": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "name1": pa.Column(
                dtype=str,
                required=True,
                nullable=False,
                coerce=True,
                checks=[EMPTY_STRING_CHECK],
            ),
            "o_a_score": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "origin": pa.Column(
                dtype=str,
                required=True,
                nullable=True,
                coerce=True,
            ),
            "ovall_cook": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "plot": pa.Column(
                dtype=int,
                required=True,
                nullable=False,
                coerce=True,
                checks=[NUMBER_NON_NEGATIVE_CHECK],
            ),
            "skincolour": pa.Column(
                dtype=str,
                required=True,
                nullable=True,
                coerce=True,
            ),
            "tubersize": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "tubnumbers": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "tubshape": pa.Column(
                dtype=str,
                required=True,
                nullable=True,
                coerce=True,
            ),
            "uniformity": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "utilisatio": pa.Column(
                dtype=float,
                required=True,
                nullable=True,
                coerce=True,
                checks=[NUMBER_0_9_RANGE_CHECK],
            ),
            "year": pa.Column(dtype=int, required=True, nullable=False, coerce=True),
        },
    )

    return schema


def get_unknown_cols(df: pd.DataFrame):
    return list(
        set(df.columns)
        - build_sheet_schema().columns.keys()
        - set(PROVENANCE_COLS)
        - set(ERROR_COLUMN_NAME)
    )


def get_schema_columns() -> list[str]:
    return sorted(list(build_sheet_schema().columns.keys()))


def get_schema_version() -> str:
    cols = get_schema_columns()
    return hashlib.sha256("|".join(cols).encode("utf-8")).hexdigest()
