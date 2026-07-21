"""
@File - test_standardise.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 20/07/2026
"""

import pandas as pd
import pytest

from src.data_load.standardise import standardise_columns


def test_standardise_raises_on_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["75-80%", "75 80%"])
    with pytest.raises(ValueError) as e:
        standardise_columns(df)

    assert "Duplicate column names after standardisation" in str(e.value)
