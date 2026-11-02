"""Pydantic models for the inputs and outputs of the tool.

These are the contract an agent sees: field types, bounds and descriptions become
the JSON schema and each MCP tool.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from powerlifting_mcp.model import ARTIFACT_DIR, Assessment, Federation, Sex

_CARD: dict[str, Any] = json.loads((ARTIFACT_DIR / "model_card.json").read_text())
BW_MIN, BW_MAX = _CARD["model"]["training_ranges"]["BodyweightKg"]
AGE_MIN, AGE_MAX = _CARD["model"]["training_ranges"]["Age"]


class Confidence(float, Enum):
    """Confidence levels that have a calibrated conformal half-width."""

    P50 = 0.50
    P68 = 0.68
    P80 = 0.80
    P95 = 0.95


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictTotalInput(_Input):
    sex: Sex = Field(description="'M' or 'F'")
    bodyweight_kg: float = Field(ge=BW_MIN, le=BW_MAX, description="Body weight in kg")
    age: float = Field(ge=AGE_MIN, le=AGE_MAX, description="Age in years")
    federation: Federation = Field(
        default="HPF", description="Greek federation: HPF, ESDT or GPL; 'Other' for any other"
    )
    confidence: Confidence = Field(
        default=Confidence.P95, description="Confidence level of the prediction interval"
    )


class StrengthStandingInput(_Input):
    sex: Sex = Field(description="'M' or 'F'")
    bodyweight_kg: float = Field(gt=0, le=250, description="Bodyweight in kg")
    total_kg: float = Field(
        gt=0, le=1500, description="Competition total (squat + bench + deadlift) in kg"
    )


class LiftBalanceInput(_Input):
    sex: Sex = Field(description="'M' or 'F'")
    squat_kg: float = Field(gt=0, le=600, description="Best squat in kg")
    bench_kg: float = Field(gt=0, le=450, description="Best benchmark in kg")
    deadlift_kg: float = Field(gt=0, le=450, description="Best deadlift in kg")


class _Output(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TotalPredictionOutput(_Output):
    predicted_total_kg: float = Field(description="Point prediction of the competition total")
    lower_kg: float
    upper_kg: float
    half_width_kg: float = Field(description="Fixed split-conformal half-width at this confidence")
    confidence: float
    validation_coverage_pct: float = Field(
        description="Share of held-out lifters whose real total fell inside the interval"
    )
    validation_coverage_male_pct: float
    validation_coverage_female_pct: float


class StrengthStandingOutput(_Output):
    dots: float = Field(description="DOTS score of a given total")
    dots_percentile: float = Field(
        description="Percent of same-sex Greek raw entries at or below this DOTS score"
    )
    total_percentile: float = Field(
        description="Percent of same-sex entries at or below this raw total; BW not adjusted"
    )
    cohort_size: int


class LiftShareOutput(_Output):
    lift: str
    kg: float
    share_pct: float = Field(description="This lift as a percent of the three-lift total")
    cohort_median_share_pct: float
    cohort_percentile: float
    assessment: Assessment = Field(
        description="'low' below the 10th percentile, 'high' above the 90th"
    )


class LiftBalanceOutput(_Output):
    total_kg: float
    lifts: list[LiftShareOutput]
    cohort_size: int
