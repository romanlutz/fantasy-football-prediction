"""One-command reproducible study build; legacy repository commands remain unchanged."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import polars as pl

from ffpred.datasets.io import file_sha256
from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider
from ffpred.study.benchmark import BLOCKS, backtest, scorecard
from ffpred.study.bridge import (
    ARCHIVE_HASH,
    PAPER_YEAR,
    historical_passers,
    reconciliation,
)
from ffpred.study.defense import (
    HALF_LIFE,
    OFFSEASON_RETENTION,
    PRIOR_PLAYS,
    defense_snapshots,
)
from ffpred.study.panel import (
    PLAYER_WEEK,
    POSITIONS,
    SCORING,
    box_scores,
    candidates,
    player_panel,
    schedule_sides,
)

HISTORY_START = 2009
TRAIN_START = 2010
LAST_COMPLETE = 2025


def run_study(
    output: Path,
    end_year: int,
    provider: NflDataProvider,
    *,
    paper_source: Callable[[], pl.DataFrame] = historical_passers,
    exact_svr: bool = False,
) -> dict[str, object]:
    if not PAPER_YEAR <= end_year <= LAST_COMPLETE:
        raise ConfigurationError("Weekly study end year must be in 2014..2025")
    sources = ProvenanceProvider(provider)
    seasons = tuple(range(HISTORY_START, end_year + 1))
    sides = schedule_sides(sources.load_schedules(seasons))
    boxes = box_scores(sources.load_player_stats(seasons))
    if sides.join(
        boxes.select("game_id", pl.col("actual_team").alias("team")).unique(),
        on=["game_id", "team"],
        how="anti",
    ).height:
        raise DataAcquisitionError(
            "Missing player box scores for a completed team-game"
        )
    pool = candidates(sides, sources.load_rosters_weekly(seasons), boxes)
    defense = defense_snapshots(sides, sources.load_team_stats(seasons))
    panel = (
        player_panel(pool, boxes)
        .join(defense, on=["game_id", "opponent"], validate="m:1")
        .filter(pl.col("season") >= TRAIN_START)
    )
    audit = reconciliation(panel, boxes, paper_source())
    predictions = backtest(panel, range(PAPER_YEAR, end_year + 1), exact_svr=exact_svr)
    annual = scorecard(predictions, by_year=True)
    pooled = scorecard(predictions, by_year=False)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "player-games.parquet": panel,
        "predictions.parquet": predictions,
        "scores-by-season.csv": annual,
        "scores-pooled.csv": pooled,
        "reconciliation-2014.csv": audit,
        "coverage.csv": _coverage(panel, boxes),
    }
    for filename, frame in files.items():
        path = output / filename
        if path.suffix == ".csv":
            frame.write_csv(path)
        else:
            frame.write_parquet(path)
    manifest = {
        "study_version": 1,
        "history_start": HISTORY_START,
        "train_start": TRAIN_START,
        "test_years": [PAPER_YEAR, end_year],
        "scoring": asdict(SCORING),
        "candidate_rule": (
            "prior completed in-season team-game roster: A01/I01/I02; "
            "prior player box if the player has no weekly status code on that team"
        ),
        "nflgame_source_sha256": ARCHIVE_HASH,
        "svr_solver": "libsvm" if exact_svr else "liblinear",
        "features": BLOCKS,
        "defense_half_life_games": HALF_LIFE,
        "defense_prior_plays": PRIOR_PLAYS,
        "defense_offseason_retention": OFFSEASON_RETENTION,
        "versions": {p: version(p) for p in ("scikit-learn", "polars", "numpy")},
        "provider": dict(provider.metadata()),
        "sources": {k: asdict(v) for k, v in sources.artifacts.items()},
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
        "panel_rows": panel.height,
        "test_rows": predictions.height,
        "legacy_2014_rows": audit["paper_points"].count(),
    }


def _coverage(panel: pl.DataFrame, boxes: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    by = ["season", "position"]
    candidates_count = panel.group_by(by).agg(
        pl.len().alias("candidate_weeks"),
        c("player_id").n_unique().alias("players"),
        (c("roster_source") == "prior_box").sum().alias("prior_box_candidates"),
        (~c("has_box")).sum().alias("no_box"),
        (c("points") == 0).sum().alias("zero_scores"),
        (c("team") != c("actual_team")).sum().alias("changed_team_with_box"),
    )
    missed = (
        boxes.with_columns(c("box_position").replace({"FB": "RB", "HB": "RB"}))
        .filter(
            (c("season") >= TRAIN_START)
            & c("box_position").is_in(POSITIONS)
            & (c("points") != 0)
        )
        .join(panel.select(PLAYER_WEEK), on=PLAYER_WEEK, how="anti")
        .rename({"box_position": "position"})
    )
    return (
        candidates_count.join(
            missed.group_by(by).agg(
                (c("week") == 1).sum().alias("week1_nonzero_outside_pool"),
                (c("week") != 1).sum().alias("later_nonzero_outside_pool"),
            ),
            on=by,
            how="left",
        )
        .fill_null(0)
        .sort(by)
    )
