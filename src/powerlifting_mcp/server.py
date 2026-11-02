"""MCP server: exposes the model layer as three tools over stdio.

It is thin by design. Each tool takes a validated input schema, calls the model layer, and
returns an output schema. All logic and numbers live in model.py
"""

import dataclasses
import logging

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from powerlifting_mcp import model
from powerlifting_mcp.schemas import (
    LiftBalanceInput,
    LiftBalanceOutput,
    PredictTotalInput,
    StrengthStandingInput,
    StrengthStandingOutput,
    TotalPredictionOutput,
)

logger = logging.getLogger(__name__)

server = MCPServer(
    "powerlifting-mcp",
    instructions=(
        "Tools for Greek raw powerlifting; full-power meets, OpenPowerlifting data. "
        "predict_total returns a calibrated prediction interval: always report the range, "
        "not only the point prediction. strength_standing and lift_balance compare a lifter "
        "with Greek raw lifters of the same sex."
    ),
)


@server.tool()
def predict_total(params: PredictTotalInput) -> TotalPredictionOutput:
    """Predict a raw full-power total with a split-conformal prediction interval.

    The interval width is fixed per confidence level. Its coverage was checked on held-out
    lifters and differs by sex: see the validation_coverage fields.
    """
    try:
        result = model.predict_total(
            params.sex, params.bodyweight_kg, params.age, params.federation, params.confidence.value
        )
    except ValueError as error:
        raise ToolError(str(error)) from error
    return TotalPredictionOutput.model_validate({"inputs": params, **dataclasses.asdict(result)})


@server.tool()
def strength_standing(params: StrengthStandingInput) -> StrengthStandingOutput:
    """DOTS of a total and its percentile among same-sex Greek raw lifters; all federations."""
    result = model.strength_standing(params.sex, params.bodyweight_kg, params.total_kg)
    return StrengthStandingOutput.model_validate(result, from_attributes=True)


@server.tool()
def lift_balance(params: LiftBalanceInput) -> LiftBalanceOutput:
    """Each lift's share of the total, compared with same-sex Greek raw lifters."""
    try:
        result = model.lift_balance(
            params.sex, params.squat_kg, params.bench_kg, params.deadlift_kg
        )
    except ValueError as error:
        raise ToolError(str(error)) from error
    return LiftBalanceOutput.model_validate(result, from_attributes=True)


@server.resource(
    "powerlifting://model-card",
    mime_type="application/json",
    description="Model card: training data, metrics, conformal coverage and limitations",
)
def model_card() -> str:
    return (model.ARTIFACT_DIR / "model_card.json").read_text()


def main() -> None:
    # stdio transport: stdout carries the protocol, so logs must go to stderr (the default)
    logging.basicConfig(level=logging.INFO)
    model.load_artifacts()
    logger.info("powerlifting-mcp ready")
    server.run()
