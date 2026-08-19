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
vector, multilayer-perceptron, and Explainable Boosting Machine regressors. QB,
team defense/special-teams (D/ST), kicker, RB/WR/TE, and individual defensive
player (IDP) prediction are all implemented; see the Positions table below for
scope and caveats.

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

Build the equivalent IDP dataset (defaults to a 2010+ history window; see
the Positions table below for why):

```console
uv run ffpred build-idp-dataset
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

## See when a player got injured, and how it affected their stats

`injury-report` compares what a player was **on pace for** (their trailing
average fantasy score over recent games they actually played) against what
actually happened for every week they appeared on the official NFL injury
report -- whether they sat out entirely, and if they played, how their score
compared to their pre-injury pace. It also tracks how many consecutive games
they had already missed, so a multi-week absence and eventual return are both
visible:

```console
uv run ffpred injury-report --output injuries.csv --start-season 2022 --end-season 2023 --positions rb
```

`--positions` accepts `qb`, `rb`, `wr`, `te`, or `all` (default). This means a
prediction that looks wrong for a given week can be checked against whether
the player was actually injured rather than a modeling failure, and their
eventual return can be compared against their own pre-injury trajectory.
Injury-report data is only available for the 2009-2024 seasons (nflverse's
source was retired after the 2024 season with no replacement announced);
requests outside that range return no events rather than raising.

For seasons after that -- where the pace comparison above has no historical
data to draw on -- `current-injuries` fetches **today's** report from ESPN's
public (unofficial) injuries endpoint as an operational supplement, and
crosswalks it to GSIS player IDs:

```console
uv run ffpred current-injuries --output current-injuries.csv
```

This is a live snapshot only, not a source of historical training features:
ESPN's endpoint is undocumented, unofficial, and exposes no reliable way to
query a past week or season, so it cannot backfill the leakage-safe
`injury-report`/dataset pipeline above -- it's meant for sanity-checking a
prediction for an upcoming game against today's actual report.

Train and evaluate any model. Pass `--position dst`, `--position k`,
`--position rb`, `--position wr`, `--position te`, or `--position idp` to
train on that dataset instead of the default quarterback dataset
(`--manual-features` is quarterback-only):

```console
uv run ffpred train-svr --train train.parquet --test test.parquet
uv run ffpred train-mlp --train train.parquet --test test.parquet
uv run ffpred train-ebm --train train.parquet --test test.parquet
uv run ffpred train-svr --explanations svr-explanations.json
uv run ffpred train-mlp --explanations mlp-explanations.json
uv run ffpred train-svr --position dst --train train.parquet --test test.parquet
uv run ffpred train-mlp --position k --train train.parquet --test test.parquet
uv run ffpred train-ebm --position wr --train train.parquet --test test.parquet
uv run ffpred evaluate svr-predictions.parquet
```

`train-ebm` writes `ebm-explanations.json` and calibrated
`prediction_lower`/`prediction_upper` columns by default. Its versioned
artifact contains native EBM term importance, learned main-effect and
interaction shapes with uncertainty bounds, and a complete additive
decomposition of every test prediction. It also includes:

- finite-sample split-conformal interval metadata, calibrated only on the
  latest held-out training periods;
- accumulated local effects (ALE) curves for every numeric feature;
- held-out permutation importance that shuffles within seasons;
- model-agnostic permutation SHAP values; and
- residual cohorts for week, position, experience, and rolling opponent
  points allowed when those columns exist for the selected position.

Each local explanation includes the row's player or team identity, so future
visualizations can link model behavior back to a concrete fantasy projection.
Use `--explanations` to choose a different path, `--interactions 0` to fit main
effects only, and `--interval-coverage` to change interval coverage.

SVR and MLP expose the same ALE, temporal permutation, SHAP, and cohort report
when `--explanations PATH` is supplied, making their diagnostics directly
comparable with EBM. Their existing training behavior is unchanged when the
option is omitted. Controls such as `--ale-bins`, `--permutation-repeats`,
`--shap-background`, and `--shap-samples` bound diagnostic cost.

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

Artifacts with a `position` column automatically add their positions to the
interface. Historical artifacts include completed game outcomes and are labeled
as backtests, not live future projections.

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
- `providers`: the `NflDataProvider` protocol plus real and in-memory adapters,
  and a standalone ESPN injuries client outside that protocol (see below).
- `acquisition`: runtime-validated Polars schemas and normalized domain records.
- `features`: named, typed, rolling features with explicit history lineage.
- `datasets`: atomic Parquet IO and versioned provenance manifests.
- `training`: deterministic SVR, MLP, and explainable EBM library APIs.
- `evaluation`: shared metrics, chronological splits, cohorts, plots, and the
  `injury_impact` pace-versus-actual report.
- `cli`: the composition root; provider choice and process behavior stay here.

Only the nflreadpy adapter imports `nflreadpy`. Polars DataFrames are validated
at acquisition and feature boundaries because static type checkers cannot track
column-level schemas. Internal domain state uses typed dataclasses and strong
identifier types instead of heterogeneous dictionaries.

`providers/espn.py` is deliberately *not* an `NflDataProvider` implementation:
it fetches a different shape of data (a live JSON snapshot, not a seasonal
Parquet/CSV release) from a different, unofficial service, and is consumed
directly by the `current-injuries` CLI command rather than by the
acquisition/features/datasets chain.

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
team's own D/ST unit, a kicker's own history, an RB/WR/TE and their upcoming
opponent's defense, or an IDP's own history). Those lineage columns are never
model inputs; tests assert that every history period strictly precedes its
target.

## Positions

| Position | Status | Scoring |
|---|---|---|
| QB | Implemented | Standard passing/rushing, configurable via `ScoringConfig` |
| Team D/ST | Implemented | Sacks, interceptions, fumble recoveries, defensive/ST touchdowns, safeties, blocked kicks, and tiered points-allowed, configurable via `DstScoringConfig` |
| K | Implemented | Field goals grouped into 0-39/40-49/50+ yard bands, PATs, configurable via `KickerScoringConfig` (kicker's own debut games are dropped rather than backed by a rookie-cohort fallback) |
| RB / WR / TE | Implemented | Rushing/receiving yards and touchdowns, fumbles, and a single configurable `reception` weight that expresses standard (0), half-PPR (0.5), or full-PPR (1.0) scoring via `ReceivingScoringConfig` (debut games are dropped, as with K; two-point conversion *attempts* are not tracked, only makes) |
| IDP | Implemented (lower-confidence) | Solo/assisted tackles, sacks, interceptions, passes defended, forced fumbles, and touchdowns, configurable via `IdpScoringConfig`. Nflverse tackle attribution is less consistently officiated than offensive box scores and some advanced defensive charting is missing in early seasons, so `build-idp-dataset` defaults to a 2010+ history window and should be evaluated separately from the other positions rather than blended into one headline metric |

All implemented positions share the same acquisition/features/datasets
architecture below; adding a position means adding a domain stats type, a
scoring config, an acquisition contract and normalize function, a feature
schema/builder module pair, and a dataset-build function, then registering
the new position's feature columns/identity columns/validator with the CLI's
`POSITION_FEATURE_COLUMNS`/`POSITION_IDENTITY_COLUMNS`/`POSITION_VALIDATORS`
maps in `cli/app.py`.

Injury-report data (used by `injury-report`, not by any `build-*-dataset`
feature schema above) is acquired the same way, via
`acquire_injury_reports`, but is only available for the 2009-2024 seasons;
nflverse retired the source after the 2024 season with no replacement
announced. `current-injuries` supplements this for later seasons via
ESPN's public (unofficial) endpoint (`providers/espn.py`), deliberately kept
outside the acquisition/features/datasets chain: it only exposes a current
snapshot, not queryable historical weeks, so it cannot produce leakage-safe
training rows the way `acquire_injury_reports` does.

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

Live nflreadpy contracts pull the completed 2025 season from nflverse for
every implemented position:

```console
uv run pytest -m "live and not live_slow" --no-cov
uv run pytest -m live_slow --no-cov
```

The second command includes the larger play-by-play download (only needed by
QB acquisition, for two-point attempts). CI runs the core live contracts
weekly and allows the play-by-play tier through manual dispatch. The same
`live` tier also includes a check against ESPN's public injuries endpoint
(`tests/live/test_espn_live.py`); being unofficial and undocumented, it may
change shape or become unavailable without notice, independent of nflverse's
own release schedule.

## Data licensing

`nflreadpy` is MIT-licensed. Most nflverse data is CC BY 4.0; specialized FTN
data uses CC BY-SA 4.0. This project uses standard player, team, schedule,
roster, and play-by-play datasets rather than FTN charting. ESPN's injuries
endpoint (used only by `current-injuries`) is an unofficial, undocumented
public API with no stated license terms; use it accordingly.
