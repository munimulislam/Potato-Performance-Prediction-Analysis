import pytest
from pydantic import ValidationError

from src.data_load.models.trial_model import Trial


def test_required_identifiers_not_empty():
    with pytest.raises(ValidationError):
        Trial(experiment_name=" ", location="x", name1="y", plot=1, year=2024)

    with pytest.raises(ValidationError):
        Trial(experiment_name="exp", location=" ", name1="y", plot=1, year=2024)

    with pytest.raises(ValidationError):
        Trial(experiment_name="exp", location="loc", name1=" ", plot=1, year=2024)


def test_minus9_is_treated_as_missing():
    data = {
        "experiment_name": "exp",
        "location": "loc",
        "name1": "clone",
        "plot": 1,
        "year": 2024,
        "o_a_score": "-9",
    }
    t = Trial(**data).model_dump()

    assert t["o_a_score"] is None
