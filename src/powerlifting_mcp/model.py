"""
Model layer: loads the exported artifacts and implements the three tools.

Plain Python with no MCP dependency. It can be tested and re-used on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

Sex = Literal["M", "F"]
Federation = Literal["HPF", "ESDT", "GPL", "Other"]
Assessment = Literal["low", "typical", "high"]
FloatArray = npt.NDArray[np.float64]

DOTS_COEFFS: dict[str, tuple[float, ...]] = {
    "M": (-307.75076, 24.0900756, -0.1918759221, 0.0007391293, -0.0000010930),
    "F": (-57.96288, 13.6175032, -0.1126655495, 0.0005158568, -0.0000010706),
}
DOTS_BW_CLAMP: dict[str, tuple[float, float]] = {"M": (40.0, 210.0), "F": (40.0, 150.0)}

FEDERATION_DUMMY: dict[str, str | None] = {
    "HPF": None,
    "ESDT": "Fed_ESDT",
    "GPL": "Fed_GPL",
    "Other": "Fed_Other Federations",
}

EXPECTED_FEATURES = {
    "Sex_binary",
    "BodyweightKg",
    "Age",
    *(d for d in FEDERATION_DUMMY.values() if d),
}

LIFT_COLUMNS = {"squat": "Best3SquatKg", "bench": "Best3BenchKg", "deadlift": "Best3DeadliftKg"}
LOW_PERCENTILE, HIGH_PERCENTILE = 10.0, 90.0


@dataclass(frozen=True)
class Artifacts:
    booster: xgb.Booster
    feature_names: list[str]
    levels: dict[float, dict[str, float]]
    dots_by_sex: dict[str, FloatArray]
    totals_by_sex: dict[str, FloatArray]
    shares_by_sex: dict[str, dict[str, FloatArray]]


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    """Load model, card, and cohort from disc once and pre-compute the cohort arrays."""
    # Model: a plain Booster --- no scikit-learn dependency
    booster = xgb.Booster()
    booster.load_model(ARTIFACT_DIR / "model_age_total.json")

    # Card: feature order and conformal levels, checked against what the code expects
    card: dict[str, Any] = json.loads((ARTIFACT_DIR / "model_card.json").read_text())
    feature_names = list(card["model"]["feature_names"])
    if set(feature_names) != EXPECTED_FEATURES or booster.num_features() != len(feature_names):
        raise RuntimeError(f"Model card features {feature_names} do not match the code")
    levels = {float(level["confidence"]): level for level in card["conformal"]["levels"]}

    cohort = pd.read_csv(ARTIFACT_DIR / "cohort.csv")
    dots_by_sex: dict[str, FloatArray] = {}
    totals_by_sex: dict[str, FloatArray] = {}
    shares_by_sex: dict[str, dict[str, FloatArray]] = {}
    for sex in ("M", "F"):
        rows = cohort[cohort["Sex"] == sex]
        dots_by_sex[sex] = np.asarray(rows["Dots"].dropna(), dtype=np.float64)
        totals_by_sex[sex] = np.asarray(rows["TotalKg"].dropna(), dtype=np.float64)
        lifts = rows[list(LIFT_COLUMNS.values())].dropna()
        lifts = lifts[(lifts > 0).all(axis=1)]
        lifts_sum = lifts.sum(axis=1)
        shares_by_sex[sex] = {
            lift: np.asarray(lifts[column] / lifts_sum, dtype=np.float64)
            for lift, column in LIFT_COLUMNS.items()
        }

    return Artifacts(booster, feature_names, levels, dots_by_sex, totals_by_sex, shares_by_sex)


def dots(total_kg: float, bodyweight_kg: float, sex: Sex) -> float:
    """DOTS score; bodyweight clamped to the valid range of the formula. Same as OpenPowerlifting"""
    low, high = DOTS_BW_CLAMP[sex]
    bw = min(max(bodyweight_kg, low), high)
    denominator = sum(c * bw**power for power, c in enumerate(DOTS_COEFFS[sex]))
    dots_score_calc = total_kg * 500.0 / denominator
    return dots_score_calc


def build_feature_row(
    sex: Sex, bodyweight_kg: float, age: float, federation: Federation
) -> FloatArray:
    """One-row feature matrix in the exact column order that the model was trained with."""
    values = {name: 0.0 for name in EXPECTED_FEATURES}
    values["Sex_binary"] = 1.0 if sex == "M" else 0.0
    values["BodyweightKg"] = float(bodyweight_kg)
    values["Age"] = float(age)
    dummy = FEDERATION_DUMMY[federation]
    if dummy is not None:
        values[dummy] = 1.0
    feature_names = load_artifacts().feature_names
    return np.array([[values[name] for name in feature_names]], dtype=np.float64)


def percentile_of(value: float, reference: FloatArray) -> float:
    """Percentage of reference values at or below `value` --- the `weak` convention."""
    pct_ref_values = float(np.mean(reference <= value) * 100.0)
    return pct_ref_values


@dataclass(frozen=True)
class TotalPrediction:
    predicted_total_kg: float
    lower_kg: float
    upper_kg: float
    half_width_kg: float
    confidence: float
    validation_coverage_pct: float
    validation_coverage_male_pct: float
    validation_coverage_female_pct: float


@dataclass(frozen=True)
class StrengthStanding:
    dots: float
    dots_percentile: float
    total_percentile: float
    cohort_size: int


@dataclass(frozen=True)
class LiftShare:
    lift: str
    kg: float
    share_pct: float
    cohort_median_share_pct: float
    cohort_percentile: float
    assessment: Assessment


@dataclass(frozen=True)
class LiftBalance:
    total_kg: float
    lifts: tuple[LiftShare, ...]
    cohort_size: int


def predict_total(
    sex: Sex, bodyweight_kg: float, age: float, federation: Federation, confidence: float = 0.95
) -> TotalPrediction:
    """Point prediction of the total plus its split-conformal interval."""
    artifacts = load_artifacts()
    level = artifacts.levels.get(confidence)
    if level is None:
        raise ValueError(f"Confidence must be one of {sorted(artifacts.levels)}")
    x = build_feature_row(sex, bodyweight_kg, age, federation)
    point = float(artifacts.booster.inplace_predict(x)[0])
    half_width = float(level["half_width_kg"])
    return TotalPrediction(
        predicted_total_kg=point,
        lower_kg=point - half_width,
        upper_kg=point + half_width,
        half_width_kg=half_width,
        confidence=confidence,
        validation_coverage_pct=float(level["coverage_overall_pct"]),
        validation_coverage_male_pct=float(level["coverage_male_pct"]),
        validation_coverage_female_pct=float(level["coverage_female_pct"]),
    )


def strength_standing(sex: Sex, bodyweight_kg: float, total_kg: float) -> StrengthStanding:
    """DOTS of the given total, and where it ranks among the Greek raw lifters of the same sex."""
    artifacts = load_artifacts()
    score = dots(total_kg, bodyweight_kg, sex)
    return StrengthStanding(
        dots=score,
        dots_percentile=percentile_of(score, artifacts.dots_by_sex[sex]),
        total_percentile=percentile_of(total_kg, artifacts.totals_by_sex[sex]),
        cohort_size=len(artifacts.dots_by_sex[sex]),
    )


def lift_balance(sex: Sex, squat_kg: float, bench_kg: float, deadlift_kg: float) -> LiftBalance:
    """Each lift's share of the total, compared with lifters of the same sex in the cohort."""
    artifacts = load_artifacts()
    lifts_kg = {"squat": squat_kg, "bench": bench_kg, "deadlift": deadlift_kg}
    total = squat_kg + bench_kg + deadlift_kg
    if total <= 0:
        raise ValueError("The three lifts must sum to a positive total.")

    shares = []
    for lift, kg in lifts_kg.items():
        reference = artifacts.shares_by_sex[sex][lift]
        percentile = percentile_of(kg / total, reference)
        assessment: Assessment = (
            "low"
            if percentile < LOW_PERCENTILE
            else "high"
            if percentile > HIGH_PERCENTILE
            else "typical"
        )
        shares.append(
            LiftShare(
                lift=lift,
                kg=kg,
                share_pct=100 * kg / total,
                cohort_median_share_pct=100.0 * float(np.median(reference)),
                cohort_percentile=percentile,
                assessment=assessment,
            )
        )
    return LiftBalance(
        total_kg=total,
        lifts=tuple(shares),
        cohort_size=len(artifacts.shares_by_sex[sex]["squat"]),
    )
