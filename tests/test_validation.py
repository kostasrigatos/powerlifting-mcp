"""Bad inputs must fail at the schema boundary with clear, located errors."""

import pytest
from pydantic import ValidationError

from powerlifting_mcp.model import lift_balance, predict_total
from powerlifting_mcp.schemas import (
    BW_MAX,
    Confidence,
    LiftBalanceInput,
    LiftBalanceOutput,
    PredictTotalInput,
    StrengthStandingInput,
    TotalPredictionOutput,
)

VALID_PREDICT = {"sex": "M", "bodyweight_kg": 90.6, "age": 34}


def test_predict_input_applies_defaults() -> None:
    params = PredictTotalInput.model_validate(VALID_PREDICT)
    assert params.federation == "HPF"
    assert params.confidence is Confidence.P95


def test_confidence_accepts_plain_number() -> None:
    params = PredictTotalInput.model_validate({**VALID_PREDICT, "confidence": 0.68})
    assert params.confidence is Confidence.P68


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bodyweight_kg", -80.0),
        ("bodyweight_kg", BW_MAX + 1),
        ("age", 200),
        ("sex", "X"),
        ("federation", "IPF"),
        ("confidence", 0.90),
    ],
)
def test_predict_input_rejects_bad_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        PredictTotalInput.model_validate({**VALID_PREDICT, field: value})
    assert exc_info.value.errors()[0]["loc"] == (field,)


def test_predict_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictTotalInput.model_validate({**VALID_PREDICT, "weight": 90})


@pytest.mark.parametrize("bad_total", [0.0, -5.0])
def test_strength_standing_input_rejects_non_positive_total(bad_total: float) -> None:
    with pytest.raises(ValidationError):
        StrengthStandingInput.model_validate(
            {"sex": "F", "bodyweight_kg": 63, "total_kg": bad_total}
        )


def test_lift_balance_input_rejects_zero_lift() -> None:
    with pytest.raises(ValidationError):
        LiftBalanceInput.model_validate(
            {"sex": "M", "squat_kg": 200, "bench_kg": 0, "deadlift_kg": 230}
        )


def test_outputs_build_from_model_results() -> None:
    prediction = TotalPredictionOutput.model_validate(
        predict_total("M", 90.6, 34, "HPF"), from_attributes=True
    )
    assert prediction.predicted_total_kg == pytest.approx(568.8663, abs=1e-3)

    balance = LiftBalanceOutput.model_validate(
        lift_balance("M", 205, 135, 230), from_attributes=True
    )
    assert [share.lift for share in balance.lifts] == ["squat", "bench", "deadlift"]
