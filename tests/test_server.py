"""End-to-end tests through the MCP layer: what an agent actually sees."""

import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, InputRequiredResult

from powerlifting_mcp.server import server

pytestmark = pytest.mark.anyio


async def test_server_lists_three_tools() -> None:
    names = {tool.name for tool in await server.list_tools()}
    assert names == {"predict_total", "strength_standing", "lift_balance"}


async def test_predict_total_through_mcp() -> None:
    result = await server.call_tool(
        "predict_total", {"params": {"sex": "M", "bodyweight_kg": 90.6, "age": 34}}
    )
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    assert result.structured_content["predicted_total_kg"] == pytest.approx(568.8663, abs=1e-3)
    assert result.structured_content["confidence"] == pytest.approx(0.95)
    assert result.structured_content["inputs"]["federation"] == "HPF"


async def test_invalid_input_returns_located_error() -> None:
    with pytest.raises(ToolError, match="bodyweight_kg"):
        await server.call_tool(
            "predict_total", {"params": {"sex": "M", "bodyweight_kg": -1, "age": 34}}
        )


async def test_model_card_resource() -> None:
    contents = await server.read_resource("powerlifting://model-card")
    assert not isinstance(contents, InputRequiredResult)
    card = json.loads(next(iter(contents)).content)
    assert card["model"]["target"] == "TotalKg"
