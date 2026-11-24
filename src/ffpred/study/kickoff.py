"""Independent pregame forecasts and strict game-side scorecards, not lineup picks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from ffpred.datasets.io import file_sha256
from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider, fingerprint_frame
from ffpred.study.club_rosters import FACTS, REVISIONS
from ffpred.study.historical_availability import (
    AVAILABILITY,
    apply_historical_availability,
)
from ffpred.study.matchup_eval import intervals
from ffpred.study.matchup_models import NAMES, project
from ffpred.study.matchup_pool import id_map
from ffpred.study.matchups import KICKOFF_NEWS, matchup_sources, prepare_board
from ffpred.study.methods import method_scores
from ffpred.study.opening import SOURCES
from ffpred.study.opportunities import CARRY, rb_opportunities
from ffpred.study.opportunity_study import edition_inputs, opportunity_sources
from ffpred.study.panel import POSITIONS
from ffpred.study.role_evidence import ENTRY, NEWS_FILE
from ffpred.study.run import HISTORY_START, LAST_COMPLETE, PAPER_YEAR
from ffpred.study.transactions import LEDGER, read_transactions

c = pl.col
CLAIMS = Path(__file__).with_name("kickoff_claims.csv")
ARMS = ("weekly", "kickoff", "kickoff-audited")
TRADE_WEEKS = [(2018, 7, "00-0031045"), (2022, 7, "00-0033280")]


def ledger_hashes(paths: list[Path]) -> dict[str, str]:
    """Compare the committed evidence across Windows and LF checkouts."""
    return {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in paths
    }


def strict_tables(
    forecasts: pl.DataFrame, data: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    selected = data["board"].filter(c("team_role") & (c("season") >= PAPER_YEAR))
    missing = selected.join(forecasts, on=ENTRY, how="anti")
    if (
        missing.height
        or forecasts.height != selected.height
        or forecasts.select(ENTRY).is_duplicated().any()
    ):
        raise DataAcquisitionError("Forecasts do not cover the issued matchup roles")
    verified = forecasts.drop_nulls("points")
    if verified.filter(~c("outcome_status").str.starts_with("same_")).height:
        raise DataAcquisitionError(
            "Attempted to score a superseded or unverifiable matchup"
        )
    annual, scores = method_scores(verified, NAMES)
    judged = data["judged"].join(
        forecasts.select(*ENTRY, "points"), on=ENTRY, validate="1:1"
    )
    role = judged.group_by("position", "opener").agg(
        pl.len().alias("issued"),
        c("points").count().alias("scored"),
        c("correct").filter(c("points").is_not_null()).sum().alias("correct"),
        c("correct").filter(c("points").is_not_null()).mean().alias("precision"),
        (c("points") == 0).sum().alias("zeros"),
        (c("points") < 0).sum().alias("negatives"),
    )
    cases = (
        data["board"]
        .join(
            pl.DataFrame(
                TRADE_WEEKS, schema=["season", "week", "player_id"], orient="row"
            ),
            on=["season", "week", "player_id"],
            how="semi",
        )
        .select(
            "season",
            "week",
            *ENTRY,
            "name",
            "opponent",
            "cutoff",
            "roster_source",
            "identity_at",
            "kind",
            "news_at",
            "ruled_out",
            "team_rank",
            "team_role",
            "outcome_status",
            "actual_team",
            "roster_team",
            "played",
            c("points").alias("matchup_points"),
        )
        .join(
            forecasts.select(*ENTRY, "ensemble", c("points").alias("scored_points")),
            on=ENTRY,
            how="left",
            validate="1:1",
        )
    )
    return {
        "forecast.parquet": forecasts,
        "scores.csv": scores,
        "annual.csv": annual,
        "intervals.csv": intervals(verified),
        "roles.csv": role.sort("opener", "position"),
        "evidence.csv": forecasts.group_by("position", "opener", "outcome_status")
        .len("rows")
        .sort("opener", "position", "outcome_status"),
        "unscored.csv": forecasts.filter(c("points").is_null())
        .select(
            "season",
            "week",
            *ENTRY,
            "name",
            "position",
            "opponent",
            "cutoff",
            "outcome_status",
            "actual_team",
            "actual_game_id",
            "snap_team",
            "roster_team",
            "ensemble",
        )
        .sort("season", "week", *ENTRY),
        "trade-weeks.csv": cases.sort("season", *ENTRY),
    }


def run_kickoff(
    output: Path,
    end_year: int,
    provider: NflDataProvider,
    *,
    arm: str,
) -> dict[str, object]:
    if not PAPER_YEAR <= end_year <= LAST_COMPLETE or arm not in ARMS:
        raise ConfigurationError("Expected a weekly/kickoff arm ending in 2014..2025")
    source = ProvenanceProvider(provider)
    raw = matchup_sources(end_year)
    years = tuple(range(HISTORY_START, end_year + 1))
    rosters, players = provider.load_rosters_weekly(years), provider.load_players()
    _, editions, metadata = edition_inputs(
        end_year, players, rosters, raw["draft"], directory_only=False
    )
    tx = read_transactions()
    inputs = [NEWS_FILE, KICKOFF_NEWS, SOURCES, REVISIONS, FACTS, LEDGER]
    if arm == "kickoff-audited":
        inputs.extend([CLAIMS, AVAILABILITY])
        tx = pl.concat([tx, read_transactions(CLAIMS)], how="vertical_relaxed")
    data = prepare_board(
        source,
        end_year,
        **raw,
        editions=editions,
        transactions=tx,
        strict_clock="weekly" if arm == "weekly" else "kickoff",
        pregame_filter=apply_historical_availability
        if arm == "kickoff-audited"
        else None,
    )
    teams, players_pbp, hashes = opportunity_sources(
        provider, years, raw["snaps"], id_map(players, rosters)
    )
    board = data["board"]
    rb = rb_opportunities(board, teams, players_pbp)
    pred = pl.concat(
        [
            project(
                rb if pos == "RB" else board,
                end_year,
                positions=(pos,),
                extra_features=CARRY if pos == "RB" else (),
            )
            for pos in POSITIONS
        ]
    ).sort("season", "week", *ENTRY)
    files = strict_tables(pred, data)
    directory = output / arm
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in files.items():
        if name.endswith(".csv"):
            frame.write_csv(directory / name, float_precision=8)
        else:
            frame.write_parquet(directory / name)
    manifest = {
        "arm": arm,
        "test_years": [PAPER_YEAR, end_year],
        "cutoff_inclusive": "first weekly kickoff minus 60 minutes"
        if arm == "weekly"
        else "own kickoff minus one microsecond",
        "target": "verified game + team + player, otherwise withheld",
        "pbp_sha256_by_season": hashes,
        "source_sha256": {
            **{k: v.sha256 for k, v in source.artifacts.items()},
            **{k: fingerprint_frame(k, v).sha256 for k, v in raw.items()},
        },
        "ledger_sha256_lf": ledger_hashes(inputs),
        "opening_editions": metadata["editions"]
        .select("season", "url", "sha256")
        .to_dicts(),
        "outputs": {
            k: {"rows": v.height, "sha256": file_sha256(directory / k)}
            for k, v in files.items()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "directory": str(directory),
        "forecasts": pred.height,
        "scored": pred["points"].count(),
    }
