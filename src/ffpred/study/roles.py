"""Historical team-role selection and forecasting on separate evaluation populations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from ffpred.datasets.io import file_sha256
from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider
from ffpred.study.benchmark import backtest, scorecard
from ffpred.study.defense import defense_snapshots
from ffpred.study.methods import compare_methods, role_error_slices
from ffpred.study.panel import (
    PLAYER_WEEK,
    box_scores,
    candidates,
    player_panel,
    schedule_sides,
)
from ffpred.study.role_evidence import (
    DEPTH_CHANGE,
    ENTRY,
    NEWS_FILE,
    SIDE,
    depth_evidence,
    game_clocks,
    injury_evidence,
    official_news,
    read_news,
)
from ffpred.study.role_selection import (
    LEAGUE,
    SLOTS,
    documented_absences,
    histories,
    league_reference,
    role_outcomes,
    role_pool,
    select_roles,
)
from ffpred.study.run import HISTORY_START, LAST_COMPLETE, PAPER_YEAR, TRAIN_START


def selection_scores(
    games: pl.DataFrame,
    selected: pl.DataFrame,
    actual: pl.DataFrame,
    *,
    by_year: bool,
) -> pl.DataFrame:
    c = pl.col
    keys = ["season", "position"] if by_year else ["position"]
    planned = games.join(
        pl.DataFrame({"position": list(SLOTS), "slots": list(SLOTS.values())}),
        how="cross",
    )
    possible = planned.group_by(keys).agg(c("slots").sum().alias("planned"))
    targets = actual.group_by(keys).agg(
        pl.len().alias("realized_roles"), c("injury_return").sum().alias("return_roles")
    )
    outcomes = actual.select(*ENTRY, "position", "injury_return")
    hits = (
        selected.join(outcomes, on=[*ENTRY, "position"], how="left")
        .group_by(keys)
        .agg(
            pl.len().alias("selected"),
            c("injury_return").count().alias("correct"),
            c("injury_return").sum().alias("correct_returns"),
            c("announced").sum().alias("announced_qbs"),
            c("depth_at").count().alias("with_dated_chart"),
            c("report_at").count().alias("with_timed_report"),
        )
    )
    return (
        possible.join(targets, on=keys, how="left")
        .join(hits, on=keys, how="left")
        .fill_null(0)
        .with_columns(
            (c("correct") / c("selected").replace(0, None)).alias("precision"),
            (c("correct_returns") / c("return_roles").replace(0, None)).alias(
                "return_recall"
            ),
        )
        .sort(keys)
    )


def run_roles(
    output: Path,
    end_year: int,
    provider: NflDataProvider,
    *,
    news_file: Path = NEWS_FILE,
    compare_models: bool = False,
) -> dict[str, object]:
    if not PAPER_YEAR <= end_year <= LAST_COMPLETE:
        raise ConfigurationError("Weekly role study end year must be in 2014..2025")
    c = pl.col
    sources = ProvenanceProvider(provider)
    seasons = tuple(range(HISTORY_START, end_year + 1))
    schedule = sources.load_schedules(seasons)
    sides = schedule_sides(schedule)
    clocks = game_clocks(sides, schedule)
    boxes = box_scores(sources.load_player_stats(seasons))
    if sides.join(
        boxes.select("game_id", c("actual_team").alias("team")).unique(),
        on=SIDE,
        how="anti",
    ).height:
        raise DataAcquisitionError(
            "Missing player box scores for a completed team-game"
        )
    rosters = sources.load_rosters_weekly(seasons)
    injuries = sources.load_injuries(seasons)
    base = candidates(sides, rosters, boxes)
    old_depth = sources.load_depth_charts(tuple(y for y in seasons if y < DEPTH_CHANGE))
    daily = (
        sources.load_depth_charts([DEPTH_CHANGE]) if end_year >= DEPTH_CHANGE else None
    )
    depth = depth_evidence(clocks, old_depth, daily)
    reports = injury_evidence(clocks, injuries)
    news = official_news(clocks, read_news(news_file))
    absences = documented_absences(rosters, injuries, boxes)
    pool = role_pool(clocks, base, depth, news, reports)
    selection = select_roles(pool, reports, histories(base, boxes, absences))
    defense = defense_snapshots(sides, sources.load_team_stats(seasons))
    base_panel = player_panel(base, boxes).join(
        defense, on=["game_id", "opponent"], validate="m:1"
    )
    selection = selection.join(league_reference(base_panel), on=ENTRY, how="left")
    evaluate = selection.filter(
        (c("season") >= PAPER_YEAR) & (c("team_role") | c("league_rank").is_not_null())
    )
    # Preserve the baseline training rows and their histories. New candidates
    # receive their own strictly lagged features, including pre-injury history.
    new_rows = (
        evaluate.select(PLAYER_WEEK)
        .unique()
        .join(base.select(PLAYER_WEEK), on=PLAYER_WEEK, how="anti")
    )
    # Construct missing identity rows only for the feature lookup. Don't count an
    # uncertain candidate's inferred zero as part of another week's baseline.
    ref = player_panel(pl.concat([base.select(PLAYER_WEEK), new_rows]), boxes)
    from_base = base_panel.drop("game_id", "team", "position", "opponent").select(
        *PLAYER_WEEK, *[col for col in ref.columns if col not in PLAYER_WEEK]
    )
    extra = ref.join(new_rows, on=PLAYER_WEEK, how="semi")
    feature_rows = pl.concat([from_base, extra], how="vertical_relaxed")
    forecast_input = (
        evaluate.join(feature_rows, on=PLAYER_WEEK, validate="m:1")
        .join(defense, on=["game_id", "opponent"], validate="m:1")
        .sort("season", "week", "team", "position", "player_id")
    )
    preds = backtest(
        base_panel.filter(c("season") >= TRAIN_START),
        range(PAPER_YEAR, end_year + 1),
        forecast_panel=forecast_input,
    )
    preds = preds.join(
        evaluate.select(*ENTRY, "team_role", "league_rank"), on=ENTRY, validate="1:1"
    )
    outcomes = role_outcomes(clocks, schedule, boxes, rosters, absences).filter(
        c("season") >= PAPER_YEAR
    )
    observed_games = clocks.filter(c("season") >= PAPER_YEAR)
    selections = evaluate.filter(c("team_role"))
    files = {
        "candidates.parquet": selection.filter(c("season") >= PAPER_YEAR),
        "predictions.parquet": preds,
        "realized-roles.parquet": outcomes,
    }
    if compare_models:
        files.update(
            compare_methods(
                base_panel.filter(c("season") >= TRAIN_START), forecast_input, preds
            )
        )
        files["methods-role-audit.csv"] = role_error_slices(
            files["method-predictions.parquet"], selections, outcomes, absences
        )
    for by_year, suffix in ((True, "by-season"), (False, "pooled")):
        files[f"selection-{suffix}.csv"] = selection_scores(
            observed_games, selections, outcomes, by_year=by_year
        )
        files[f"scores-{suffix}.csv"] = pl.concat(
            [
                scorecard(preds.filter(mask), by_year=by_year).with_columns(
                    pl.lit(label).alias("cohort")
                )
                for label, mask in (
                    ("team_roles", c("team_role")),
                    ("league_caps", c("league_rank").is_not_null()),
                )
            ]
        )
    output.mkdir(exist_ok=True, parents=True)
    for filename, frame in files.items():
        if filename.endswith(".csv"):
            frame.write_csv(output / filename)
        else:
            frame.write_parquet(output / filename)
    manifest = {
        "study": "weekly-team-roles-v1",
        "test_years": [PAPER_YEAR, end_year],
        "cutoff_minutes_before_kickoff": 60,
        "team_slots": SLOTS,
        "league_caps": LEAGUE,
        "official_news_sha256": file_sha256(news_file),
        "sources": {k: asdict(v) for k, v in sources.artifacts.items()},
        "provider": dict(provider.metadata()),
        "outputs": {
            k: {"rows": v.height, "sha256": file_sha256(output / k)}
            for k, v in files.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "directory": str(output),
        "team_selections": selections.height,
        "represented_team_games": observed_games.height,
        "league_selections": preds["league_rank"].count(),
    }
