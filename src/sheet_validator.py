"""
@File - sheet_validator.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 08/07/2026
"""

from dataclasses import dataclass, field

from .schema import build_sheet_schema
import pandas as pd
import pandera.pandas as pa


@dataclass
class SheetValidationResult:
    ok: bool
    row_errors: dict[int, str] = field(default_factory=dict)
    schema_errors: list[str] = field(default_factory=list)

    @property
    def has_schema_errors(self) -> bool:
        return len(self.schema_errors) > 0

    @property
    def has_row_errors(self) -> bool:
        return len(self.row_errors) > 0

    @property
    def error_row_indices(self) -> list[int]:
        return list(self.row_errors.keys())


class SheetValidator:
    def __init__(self):
        self.schema = build_sheet_schema()

    def extract_row_errors(self, failure_cases: pd.DataFrame) -> dict[int, str]:
        row_errors: dict[int, str] = {}
        cases = failure_cases[failure_cases["index"].notna()]

        if len(cases):
            grouped = cases.groupby("index")
            for idx, group in grouped:
                parts = [
                    f"{row['column']}: {row['check']}" for _, row in group.iterrows()
                ]
                row_errors[int(idx)] = " | ".join(parts)

        return row_errors

    def extract_schema_errors(self, failure_cases: pd.DataFrame) -> list[str]:
        cases = failure_cases[failure_cases["index"].isna()]

        schema_errors = [
            f"{r['check']}: {r['failure_case']}" for _, r in cases.iterrows()
        ]

        return schema_errors

    def validate(self, df: pd.DataFrame, lazy=True) -> SheetValidationResult:
        try:
            self.schema.validate(df, lazy=lazy)
            return SheetValidationResult(ok=True, row_errors={}, schema_errors=[])
        except pa.errors.SchemaErrors as err:
            fc = err.failure_cases
            row_errors = self.extract_row_errors(fc)
            schema_errors = self.extract_schema_errors(fc)
            ok = len(schema_errors) == 0 and len(row_errors) == 0

            return SheetValidationResult(
                ok=ok, row_errors=row_errors, schema_errors=schema_errors
            )
