# Fantasy Football Prediction

> [!IMPORTANT]
> This project is undergoing a major overhaul. The current code, commands, and
> documentation may continue to change.

The original experiment is described in the paper
[Fantasy Football Prediction](https://arxiv.org/abs/1505.06918). The exact code
associated with the paper is preserved in the
[`legacy-2015`](https://github.com/romanlutz/fantasy-football-prediction/tree/legacy-2015)
tag.

This project builds reproducible, leakage-safe quarterback fantasy-football
datasets from [nflverse](https://github.com/nflverse/nflverse-data), then trains
support-vector and multilayer-perceptron regressors.

## Requirements and setup

The supported runtime is CPython 3.11 through 3.13. Install the locked runtime
and development environment with [uv](https://docs.astral.sh/uv/):

```console
uv sync --all-groups
```

## Commands

Build the historical 2010-2013 training set and 2014 test set:

```console
uv run ffpred build-dataset
```

Build a more recent experiment:

```console
uv run ffpred build-dataset \
  --history-start 2018 \
  --train-start 2019 \
  --test-year 2025 \
  --output-dir artifacts
```

Enable nflreadpy's filesystem cache for repeated builds:

```powershell
$env:FFPRED_CACHE_MODE = "filesystem"
uv run ffpred build-dataset
```

Train and evaluate either model:

```console
uv run ffpred train-svr --train train.parquet --test test.parquet
uv run ffpred train-mlp --train train.parquet --test test.parquet
uv run ffpred evaluate svr-predictions.parquet
```

Launch the graphical war room with one or more prediction artifacts:

```console
uv run ffpred-dashboard --predictions svr-predictions.parquet
uv run ffpred-dashboard \
  --predictions svr-predictions.parquet \
  --predictions mlp-predictions.parquet
```

The **Draft Board** aggregates weekly rows into season totals. **Weekly
Decisions** keeps single-game comparisons separate, and **Model Room** shows
error and bias when actual results are present. The dashboard automatically
discovers `*-predictions.parquet` files in the current directory and
`artifacts/`; files can also be uploaded in the browser.

The current training pipeline is quarterback-only, so existing artifacts show
only QB in the position filter. Artifacts with a `position` column automatically
add their positions to the interface. Historical artifacts include completed
game outcomes and are labeled as backtests, not live future projections.

Build a point-in-time season forecast without using any target-season game
statistics:

```console
# Historical replay: 2025 predictions frozen before the season, with actuals
# attached afterward for evaluation.
uv run ffpred build-forecast \
  --history-start 2018 \
  --train-start 2019 \
  --history-through 2024 \
  --target-year 2025 \
  --include-actuals \
  --output-dir artifacts/2025

# Upcoming season: 2026 schedule and preseason QB1 depth charts, with all model
# features frozen after the completed 2025 season.
uv run ffpred build-forecast \
  --history-start 2018 \
  --train-start 2019 \
  --history-through 2025 \
  --target-year 2026 \
  --output-dir artifacts/2026

uv run ffpred project-svr \
  --train artifacts/2026/training.parquet \
  --forecast artifacts/2026/forecast.parquet \
  --predictions artifacts/2026/svr-predictions.parquet
uv run ffpred project-mlp \
  --train artifacts/2026/training.parquet \
  --forecast artifacts/2026/forecast.parquet \
  --predictions artifacts/2026/mlp-predictions.parquet
```

Forecast builds use the latest QB1 depth-chart snapshot available no later than
the forecast date. A historical replay defaults to the last depth chart before
that season's first regular-season game; an upcoming forecast defaults to the
current date. The generated manifest records the cutoff, as-of date, source
hashes, and output hashes. Changing a team's projected starter requires
rebuilding after nflverse publishes an updated depth chart.

Every command writes machine-readable JSON to standard output. Diagnostics use
Python logging on standard error. Environment defaults are available as
`FFPRED_OUTPUT_DIR`, `FFPRED_HISTORY_START`, `FFPRED_TRAIN_START`,
`FFPRED_TEST_YEAR`, `FFPRED_CACHE_MODE`, and `FFPRED_LOG_LEVEL`; explicit CLI
arguments take precedence. Set `FFPRED_CACHE_DIR` to choose the filesystem cache
location.

## Architecture

The installable package uses explicit dependency direction:

```text
cli -> training/evaluation -> datasets -> features -> acquisition -> providers
                                      \-> domain <-/
```

- `domain`: immutable identifiers, game records, histories, and scoring rules.
- `providers`: the `NflDataProvider` protocol plus real and in-memory adapters.
- `acquisition`: runtime-validated Polars schemas and normalized domain records.
- `features`: named, typed, rolling features with explicit history lineage.
- `datasets`: atomic Parquet IO and versioned provenance manifests.
- `training`: deterministic, scaled SVR and MLP library APIs.
- `evaluation`: shared metrics, chronological splits, cohorts, and plots.
- `cli`: the composition root; provider choice and process behavior stay here.

Only the nflreadpy adapter imports `nflreadpy`. Polars DataFrames are validated
at acquisition and feature boundaries because static type checkers cannot track
column-level schemas. Internal domain state uses typed dataclasses and strong
identifier types instead of heterogeneous dictionaries.

## Artifacts and reproducibility

Dataset builds produce:

- `train.parquet` and `test.parquet`, with named columns and preserved dtypes.
- `dataset-manifest.json`, schema version 2.

The manifest records the package and provider versions, scoring/build
parameters, feature-schema hash, output hashes, and Arrow-IPC content hashes and
schemas for every source frame. Preserve the manifest and Parquet files
together. nflverse release assets can be corrected after publication, so source
hashes are necessary to distinguish upstream revisions.

Feature rows include target identity and the latest quarterback and defense
history period used. Those lineage columns are never model inputs; tests assert
that every history period strictly precedes its target.

## Quality checks

Run the complete offline quality suite:

```console
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -m "not live" --cov
```

The offline suite includes unit, property, mocked-provider, schema-contract,
integration, golden snapshot, leakage, model, plot, and CLI tests with a 90%
branch-aware coverage gate.

Live nflreadpy contracts pull the completed 2025 season from nflverse:

```console
uv run pytest -m "live and not live_slow" --no-cov
uv run pytest -m live_slow --no-cov
```

The second command includes the larger play-by-play download. CI runs the core
live contracts weekly and allows the play-by-play tier through manual dispatch.

## Data licensing

`nflreadpy` is MIT-licensed. Most nflverse data is CC BY 4.0; specialized FTN
data uses CC BY-SA 4.0. This project uses standard player, team, schedule,
roster, and play-by-play datasets rather than FTN charting.
