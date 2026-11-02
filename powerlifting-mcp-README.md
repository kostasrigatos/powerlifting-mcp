# powerlifting-mcp

![CI](https://github.com/kostasrigatos/powerlifting-mcp/actions/workflows/ci.yml/badge.svg)

An [MCP](https://modelcontextprotocol.io) server that exposes my
[Greek raw powerlifting model](https://github.com/kostasrigatos/Greek-powerlifting-ML)
as tools an AI agent can call: a tuned XGBoost model with split-conformal prediction
intervals, plus cohort comparisons built on OpenPowerlifting data.

The point of the project is the interface, not the model: typed, validated tool inputs,
structured outputs that carry their own uncertainty, and tests that tie the server to the
numbers verified in the original analysis.

## Tools

| Tool | Input | Output |
|------|-------|--------|
| `predict_total` | sex, bodyweight, age, federation, confidence (50 / 68 / 80 / 95%) | Predicted total with its split-conformal interval, the validation coverage at that level (overall, men, women), and the inputs actually used, defaults included |
| `strength_standing` | sex, bodyweight, total | DOTS score and its percentile among same-sex Greek raw lifters; raw-total percentile as a secondary field |
| `lift_balance` | sex, squat, bench, deadlift | Each lift's share of the total against the same-sex cohort distribution, flagged `low` / `typical` / `high` |

The server also exposes the model card as a resource, `powerlifting://model-card`
(training data, metrics, conformal coverage and limitations).

## Example: Claude Desktop

> **Prompt:** I'm a 34-year-old male lifter, 90.6 kg, competing in HPF. What total should I
> expect? Then, if I total 645 kg, how do I rank, and is my lift balance normal with
> 230/150/265?

Claude called all three tools and answered (condensed):

> **Expected total:** about 569 kg, with a 95% interval of 406–731 kg. The interval is wide
> (±162 kg), because age, bodyweight and federation alone say little about an individual lifter.
> In validation it held 93.1% of male lifters, slightly under the 95% target.
>
> **At 645 kg:** DOTS 415.7, about the 82nd percentile of male Greek raw lifters (2,108 entries).
> This comparison covers all Greek federations, not just HPF.
>
> **Lift balance:** all three lifts typical. Squat 35.7% of total (cohort median 35.3%),
> bench 23.3% (23.4%), deadlift 41.1% (41.3%).

**What agent testing changed.** In the first version, Claude answered correctly but added that
it could not tell whether the federation input had been used. Federation is a model feature,
but HPF is also the default, and the output did not show which inputs produced the prediction.
`predict_total` now echoes its resolved inputs, defaults included, and the descriptions say that
the comparison cohort spans all federations. On a re-run, the doubt was gone.

## Design

```
src/powerlifting_mcp/
├── model.py      # plain Python, no MCP: loads artifacts once, implements the three tools
├── schemas.py    # Pydantic inputs and outputs: the contract the agent sees
├── server.py     # thin MCP layer: schema in -> model -> schema out, over stdio
└── artifacts/    # exported from the analysis notebook
    ├── model_age_total.json   # XGBoost native JSON (no pickle)
    ├── model_card.json        # feature order, conformal half-widths, metrics, limitations
    └── cohort.csv             # reference cohort for percentiles, no identifiers
```

- **The server never trains anything.** The notebook's export cell writes the three
  artifacts, then reloads them from disk and asserts them against golden values from an
  earlier end-to-end run (test-split fingerprint, predictions, interval bounds, all four
  half-widths). A fresh run reproducing those values is what makes the artifacts trustworthy.
- **`model.py` is testable without MCP.** `server.py` is a wrapper of about 80 lines. A REST
  API would be one more thin file next to it, with nothing else changing.
- **Validation at the boundary.** `predict_total` accepts only bodyweights and ages inside
  the model's training range, read from the model card, so the model is never asked to
  extrapolate. Unknown fields are rejected (`extra="forbid"`), so a misspelled argument fails
  loudly instead of silently falling back to a default. Errors name the offending field, so an
  agent can correct its call.
- **Two kinds of error.** A bad argument becomes a `ToolError`, whose message reaches the
  agent. Anything else is treated as a crash, and its details are not sent to the client.
- **Load-time checks.** If the model card's features don't match what the code builds, the
  server refuses to start.

## Testing

35 tests, run in CI together with `ruff format --check`, `ruff check` and `mypy --strict`:

- **Golden values** (`test_tools.py`): `predict_total` reproduces the notebook's numbers for
  both sexes and all four federation paths, and the conformal half-widths at all four
  confidence levels. My DOTS implementation matches OpenPowerlifting's stored `Dots` column
  to within its 2-decimal rounding.
- **Properties**, for the tools with no notebook reference: extremes map to the 0th and
  100th percentiles, a higher total never ranks lower, lift shares sum to 100%, the median
  lifter is "typical", and an obvious outlier is flagged.
- **Validation** (`test_validation.py`): out-of-range values, unknown federations,
  unsupported confidence levels and unknown fields all fail with a located error.
- **Through the MCP layer** (`test_server.py`): tool listing, a full `call_tool` round
  trip, error propagation, and the model-card resource.

## Run it

Requires [uv](https://docs.astral.sh/uv/). On macOS, XGBoost also needs OpenMP:
`brew install libomp`.

```bash
git clone https://github.com/kostasrigatos/powerlifting-mcp.git
cd powerlifting-mcp
uv sync
uv run pytest
```

**MCP Inspector** (needs Node.js):

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/powerlifting-mcp run powerlifting-mcp
```

**Claude Desktop:** Settings → Desktop app → Local MCP servers → Edit config, then add:

```json
{
  "mcpServers": {
    "powerlifting": {
      "command": "/absolute/path/to/uv",
      "args": ["--directory", "/path/to/powerlifting-mcp", "run", "powerlifting-mcp"]
    }
  }
}
```

Use the absolute path from `which uv`: Claude Desktop does not use your shell's `PATH`.
Restart the app fully (Cmd+Q) after editing.

## Model and limitations

- **Model:** XGBoost tuned with Optuna on Greek raw full-power (SBD) entries since 2015
  (OpenPowerlifting), with age-augmented features. Test RMSE 78.6 kg, 5-fold CV RMSE 79.9 kg.
- **Intervals:** split conformal, with an absolute-residual score and the finite-sample
  quantile, calibrated on half of the test set and validated on the other half (183 rows
  each). At 95% nominal: 94.5% coverage overall, 93.1% for men, 100% for women.
- **Fixed-width, marginal coverage.** One half-width per confidence level for everyone, so
  women are over-covered and men slightly under-covered. The pooled quantile is dominated
  by the larger male group.
- **Small validation set:** 183 rows, 38 of them women, so the coverage figures carry wide
  sampling error.
- **Optimistic test RMSE:** Optuna's CV search used the full cohort, including the test rows.
- **Scope:** Greek raw full-power lifting only; not equipped lifting, single-lift meets or
  other countries. Percentiles compare against all entries, not best-per-lifter.

Full analysis, data cleaning and the comparison with a Bayesian alternative:
[Greek-powerlifting-ML](https://github.com/kostasrigatos/Greek-powerlifting-ML).

