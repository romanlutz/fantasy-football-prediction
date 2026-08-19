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
[nflverse](https://github.com/nflverse/nflverse-data), then trains support-
vector and multilayer-perceptron regressors. Quarterback and team defense/
special-teams (D/ST) prediction are implemented today; additional positions
are being added incrementally.

## Requirements and setup

The supported runtime is CPython 3.11 through 3.13. Install the locked runtime
and development environment with [uv](https://docs.astral.sh/uv/):

```console
uv sync --all-groups
```

## Commands

Build the default quarterback dataset (the historical 2010-2013 training set
and 2014 test set):

```console
uv run ffpred build-dataset
```

Build the equivalent team defense/special-teams (D/ST) dataset:

```console
uv run ffpred build-dst-dataset
```

Build the equivalent kicker dataset:

```console
uv run ffpred build-kicker-dataset
```

Build the equivalent RB/WR/TE dataset (`--position` selects `rb`, `wr`, `te`,
or `all` for a combined table; default `all`):

```console
uv run ffpred build-receiving-dataset --position wr
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

Train and evaluate either model. Pass `--position dst`, `--position k`,
`--position rb`, `--position wr`, or `--position te` to train on that
dataset instead of the default quarterback dataset (`--manual-features` is
quarterback-only):

```console
uv run ffpred train-svr --train train.parquet --test test.parquet
uv run ffpred train-mlp --train train.parquet --test test.parquet
uv run ffpred train-svr --position dst --train train.parquet --test test.parquet
uv run ffpred train-mlp --position k --train train.parquet --test test.parquet
uv run ffpred train-svr --position wr --train train.parquet --test test.parquet
uv run ffpred evaluate svr-predictions.parquet
```

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

Feature rows include target identity and the latest history period used for
that row's own group (a quarterback and their upcoming opponent's defense, a
team's own D/ST unit, a kicker's own history, or an RB/WR/TE and their
upcoming opponent's defense). Those lineage columns are never model inputs;
tests assert that every history period strictly precedes its target.

## Positions

| Position | Status | Scoring |
|---|---|---|
| QB | Implemented | Standard passing/rushing, configurable via `ScoringConfig` |
| Team D/ST | Implemented | Sacks, interceptions, fumble recoveries, defensive/ST touchdowns, safeties, blocked kicks, and tiered points-allowed, configurable via `DstScoringConfig` |
| K | Implemented | Field goals grouped into 0-39/40-49/50+ yard bands, PATs, configurable via `KickerScoringConfig` (kicker's own debut games are dropped rather than backed by a rookie-cohort fallback) |
| RB / WR / TE | Implemented | Rushing/receiving yards and touchdowns, fumbles, and a single configurable `reception` weight that expresses standard (0), half-PPR (0.5), or full-PPR (1.0) scoring via `ReceivingScoringConfig` (debut games are dropped, as with K; two-point conversion *attempts* are not tracked, only makes) |
| IDP | Not yet implemented | — |

All implemented positions share the same acquisition/features/datasets
architecture below; adding a position means adding a domain stats type, a
scoring config, an acquisition contract and normalize function, a feature
schema/builder module pair, and a dataset-build function, then registering
the new position's feature columns/identity columns/validator with the CLI's
`POSITION_FEATURE_COLUMNS`/`POSITION_IDENTITY_COLUMNS`/`POSITION_VALIDATORS`
maps in `cli/app.py`.

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
