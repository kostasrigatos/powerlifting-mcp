"""Golden-value and property tests for the model layer.

Golden numbers come from the notebook's end-to-end run (the export cell's assertions).
"""

import numpy as np
import pandas as pd
import pytest

from powerlifting_mcp.model import (
    ARTIFACT_DIR,
    Federation,
    Sex,
    dots,
    lift_balance,
    load_artifacts,
    predict_total,
    strength_standing,
)

# ---------- artifacts ----------


def test_artifacts_load_with_four_confidence_levels() -> None:
    assert set(load_artifacts().levels) == {0.50, 0.68, 0.80, 0.95}


# ---------- predict_total ----------


def test_predict_total_phase8_profile() -> None:
    result = predict_total("M", 90.6, 34, "HPF", confidence=0.95)
    assert result.predicted_total_kg == pytest.approx(568.8663, abs=1e-3)
    assert result.lower_kg == pytest.approx(406.4941, abs=1e-3)
    assert result.upper_kg == pytest.approx(731.2386, abs=1e-3)
    assert result.validation_coverage_pct == pytest.approx(94.54, abs=0.01)
    assert result.validation_coverage_female_pct == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("confidence", "half_width"),
    [(0.50, 47.1813), (0.68, 81.1194), (0.80, 103.5140), (0.95, 162.3722)],
)
def test_predict_total_half_widths(confidence: float, half_width: float) -> None:
    result = predict_total("M", 90.6, 34, "HPF", confidence=confidence)
    assert result.half_width_kg == pytest.approx(half_width, abs=1e-3)
    assert result.upper_kg - result.lower_kg == pytest.approx(2 * half_width, abs=1e-3)


@pytest.mark.parametrize(
    ("sex", "bodyweight_kg", "age", "federation", "expected_total"),
    [
        ("F", 63.0, 27, "HPF", 321.81),
        ("M", 83.0, 25, "ESDT", 518.81),
        ("M", 74.0, 30, "Other", 534.22),
        ("F", 57.0, 22, "GPL", 283.39),
    ],
)
def test_predict_total_covers_both_sexes_and_all_federations(
    sex: Sex, bodyweight_kg: float, age: float, federation: Federation, expected_total: float
) -> None:
    result = predict_total(sex, bodyweight_kg, age, federation)
    assert result.predicted_total_kg == pytest.approx(expected_total, abs=0.01)


def test_predict_total_rejects_unknown_confidence() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        predict_total("M", 90.6, 34, "HPF", confidence=0.90)


# ---------- strength_standing ----------


def test_dots_matches_openpowerlifting_column() -> None:
    cohort = pd.read_csv(ARTIFACT_DIR / "cohort.csv").dropna(
        subset=["Sex", "BodyweightKg", "TotalKg", "Dots"]
    )
    assert set(cohort["Sex"]) <= {"M", "F"}

    sexes: tuple[Sex, ...] = ("M", "F")
    for sex in sexes:
        rows = cohort[cohort["Sex"] == sex]
        computed = np.array(
            [
                dots(float(total), float(bodyweight), sex)
                for total, bodyweight in zip(rows["TotalKg"], rows["BodyweightKg"], strict=True)
            ]
        )
        max_diff = np.abs(computed - rows["Dots"].to_numpy()).max()
        assert max_diff <= 0.0051, f"{sex}: max |diff| = {max_diff:.4f}"


def test_strength_standing_extremes_and_cohort_size() -> None:
    strongest_total = float(load_artifacts().totals_by_sex["M"].max())
    top = strength_standing("M", 93.0, strongest_total)
    bottom = strength_standing("M", 93.0, 1.0)
    assert top.total_percentile == pytest.approx(100.0)
    assert bottom.total_percentile == pytest.approx(0.0)
    assert bottom.dots_percentile == pytest.approx(0.0)
    assert top.cohort_size == 2108


def test_strength_standing_is_monotonic_in_total() -> None:
    lower = strength_standing("F", 63.0, 300.0)
    higher = strength_standing("F", 63.0, 400.0)
    assert higher.dots > lower.dots
    assert higher.dots_percentile >= lower.dots_percentile
    assert higher.total_percentile >= lower.total_percentile


# ---------- lift_balance ----------


def test_lift_balance_shares_sum_to_100() -> None:
    result = lift_balance("M", 205.0, 135.0, 230.0)
    assert result.total_kg == pytest.approx(570.0)
    assert sum(share.share_pct for share in result.lifts) == pytest.approx(100.0)


def test_lift_balance_median_lifter_is_typical() -> None:
    shares = load_artifacts().shares_by_sex["F"]
    medians = {lift: float(np.median(shares[lift])) for lift in ("squat", "bench", "deadlift")}
    result = lift_balance(
        "F", 300 * medians["squat"], 300 * medians["bench"], 300 * medians["deadlift"]
    )
    assert all(share.assessment == "typical" for share in result.lifts)


def test_lift_balance_flags_bench_heavy_lifter() -> None:
    result = lift_balance("M", 150.0, 200.0, 180.0)
    bench = next(share for share in result.lifts if share.lift == "bench")
    assert bench.assessment == "high"


def test_lift_balance_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="positive total"):
        lift_balance("M", 0.0, 0.0, 0.0)
