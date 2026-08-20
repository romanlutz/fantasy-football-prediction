# Fantasy Football Prediction

> [!IMPORTANT]
> This project is undergoing a major overhaul. The current code, commands, and
> documentation may continue to change.

The original experiment is described in the paper
[Fantasy Football Prediction](https://arxiv.org/abs/1505.06918). The exact code
associated with the paper is preserved in the
[`legacy-2015`](https://github.com/romanlutz/fantasy-football-prediction/tree/legacy-2015)
tag.

This project builds reproducible, leakage-safe fantasy-football datasets from
[nflverse](https://github.com/nflverse/nflverse-data), then trains
support-vector and multilayer-perceptron regressors for QB, RB, WR, TE, K, and
team defense/special teams (DST).

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
`artifacts/`. Prediction-file selection stays out of the interface; use repeated
`--predictions` arguments when launching against explicit artifacts.

The dashboard reads the complete point-in-time archive from `artifacts/`, so the
season selector covers 2010 through the upcoming season and the position filter
offers QB, RB, WR, TE, K, and DST. Historical artifacts include completed game
outcomes for comparison and remain frozen to information available before their
target season.

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

Build the complete standard-scoring archive:

```console
uv run ffpred build-forecast-archive \
  --history-start 1999 \
  --first-target-year 2010 \
  --last-target-year 2026 \
  --output-dir artifacts
```

The archive uses week-one historical depth charts through 2024 and preseason
daily depth charts from 2025 onward. It selects fantasy-relevant depth at each
position, adds one DST entry per team, and freezes every feature at the previous
completed season. If a source depth chart omits a team-position slot, the builder
logs the gap, falls back to that team's top prior-season producer, and records
team-position coverage in the manifest. Historical outcomes are attached only
after the frozen features are built. For completed target seasons, nflverse
injury reports and weekly roster status identify scheduled absences backed by an
Out or reserve-list designation. These outcome fields never enter model features.
Run both model projections for each generated season:

```powershell
foreach ($year in 2010..2026) {
  uv run ffpred project-svr `
    --train "artifacts\$year\training.parquet" `
    --forecast "artifacts\$year\forecast.parquet" `
    --predictions "artifacts\$year\svr-predictions.parquet"
  uv run ffpred project-mlp `
    --train "artifacts\$year\training.parquet" `
    --forecast "artifacts\$year\forecast.parquet" `
    --predictions "artifacts\$year\mlp-predictions.parquet"
}
```

Player scoring is standard non-PPR. Kicker scoring awards 3 points through 39
yards, 4 from 40–49, 5 from 50+, and 1 per extra point. DST scoring includes
sacks, takeaways, touchdowns, safeties, blocked kicks, and points allowed.

The dashboard places projected and actual season totals side by side. It shows
two availability-adjusted results: `actual points + injury games missed x
projected points per game` and `actual points + injury games missed x actual
points per game`. Their gap percentages show how close each estimate lands to
the original preseason projection. Questionable, healthy-inactive, suspended,
and otherwise unclassified absences are not counted as injury misses.

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

Feature rows include target identity and the latest player and opponent history
season used. Those lineage columns are never model inputs; tests assert that
every history period strictly precedes its target.

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
