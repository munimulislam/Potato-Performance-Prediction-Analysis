"""
@File - trial_model.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 20/07/2026
"""

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    model_validator,
)


class Trial(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    experiment_name: str = Field(min_length=1)
    location: str = Field(min_length=1)
    name1: str = Field(min_length=1)
    plot: int = Field(gt=0)
    year: int
    source_file: str = Field(min_length=1)
    source_row: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def clean_all_nans(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data

        cleaned = {}

        for key, val in data.items():
            if val is None or pd.isna(val):
                cleaned[key] = None
            elif isinstance(val, str):
                stripped = val.strip()
                if stripped == "":
                    cleaned[key] = None
                else:
                    try:
                        num = float(stripped)
                    except ValueError:
                        cleaned[key] = stripped
                    else:
                        cleaned[key] = None if num < 0 else stripped
            elif isinstance(val, bool):
                cleaned[key] = val
            elif isinstance(val, (int, float)) and val < 0:
                cleaned[key] = None
            elif isinstance(val, int) and key not in ["year", "plot"]:
                cleaned[key] = float(val)
            else:
                cleaned[key] = val
        return cleaned
