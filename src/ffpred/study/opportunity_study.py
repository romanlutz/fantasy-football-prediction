"""Separate opening-role selection from paired RB opportunity-feature comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ffpred.datasets.io import file_sha256
from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider, fingerprint_frame
from ffpred.study.club_rosters import FACTS, REVISIONS, club_editions
from ffpred.study.matchup_eval import starter_scores
from ffpred.study.matchup_models import NAMES, project
from ffpred.study.matchup_pool import id_map
from ffpred.study.matchups import matchup_sources, prepare_board
from ffpred.study.opening import SOURCES, opening_editions
from ffpred.study.opportunities import (
    CARRY,
    COUNTS,
    GOAL,
    count_opportunities,
    include_snap_zeros,
    rb_opportunities,
)
from ffpred.study.panel import PLAYER_WEEK, TEAM_CODES
from ffpred.study.role_evidence import ENTRY, NEWS_FILE, SIDE
from ffpred.study.run import HISTORY_START, LAST_COMPLETE, PAPER_YEAR
from ffpred.study.transactions import LEDGER, read_transactions

c = pl.col
PBP_COLUMNS = (
    "season_type",
    "game_id",
    "season",
    "week",
    "posteam",
    "play_type",
    "rush_attempt",
    "qb_kneel",
    "rusher_player_id",
    "receiver_player_id",
    "yardline_100",
    "play_deleted",
    "two_point_attempt",
)


def opportunity_sources(
    provider: NflDataProvider,
    years: tuple[int, ...],
    snaps: pl.DataFrame,
    ids: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, str]]:
    teams: list[pl.DataFrame] = []
    players: list[pl.DataFrame] = []
    hashes: dict[str, str] = {}
    for year in years:
        raw = provider.load_pbp(year).select(PBP_COLUMNS)
        hashes[str(year)] = fingerprint_frame(str(year), raw).sha256
        team, player = count_opportunities(raw)
        teams.append(team)
        players.append(player)
    team = pl.concat(teams).sort("season", "week", "team")
    player = include_snap_zeros(pl.concat(players), team, snaps, ids).sort(
        "season", "week", "team", "player_id"
    )
    return team, player, hashes


def observed_affiliations(
    board: pl.DataFrame, snaps: pl.DataFrame, ids: pl.DataFrame
) -> pl.DataFrame:
    observed = (
        snaps.filter(
            (c("game_type") == "REG")
            & (pl.sum_horizontal("offense_snaps", "defense_snaps", "st_snaps") > 0)
        )
        .join(ids, left_on="pfr_player_id", right_on="pfr_id")
        .select(*PLAYER_WEEK, c("team").replace(TEAM_CODES).alias("snap_team"))
        .unique()
        .group_by(PLAYER_WEEK)
        .agg(c("snap_team").first(), pl.len().alias("snap_teams"))
    )
    # Box and snap teams are evaluation labels, never sources for the picks.
    category = (
        pl.when((c("snap_teams") > 1) | (c("actual_team") != c("snap_team")))
        .then(pl.lit("conflicting_records"))
        .when(c("actual_team") == c("team"))
        .then(pl.lit("same_box"))
        .when(c("actual_team").is_not_null())
        .then(pl.lit("other_box"))
        .when(c("snap_team") == c("team"))
        .then(pl.lit("same_snap_only"))
        .when(c("snap_team").is_not_null())
        .then(pl.lit("other_snap_only"))
        .otherwise(pl.lit("unverified"))
    )
    return (
        board.filter(c("team_role") & (c("season") >= PAPER_YEAR))
        .join(observed, on=PLAYER_WEEK, how="left", validate="1:1")
        .with_columns(category.alias("evidence"))
    )


def affiliation_audit(
    board: pl.DataFrame, snaps: pl.DataFrame, ids: pl.DataFrame
) -> pl.DataFrame:
    return (
        observed_affiliations(board, snaps, ids)
        .group_by("season", "opener", "evidence")
        .len("picks")
    )


def roster_audit(observed: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    weekly = (
        rosters.filter(c("game_type") == "REG")
        .select(
            "season",
            "week",
            c("gsis_id").alias("player_id"),
            c("team").replace(TEAM_CODES).alias("roster_team"),
            "status_description_abbr",
        )
        .drop_nulls(["player_id", "roster_team"])
        .unique()
        .group_by(PLAYER_WEEK)
        .agg(
            c("roster_team").n_unique().alias("roster_teams"),
            c("roster_team").unique().sort().str.join("|"),
            c("status_description_abbr")
            .drop_nulls()
            .unique()
            .sort()
            .str.join("|")
            .alias("roster_codes"),
        )
    )
    return observed.join(
        weekly, on=PLAYER_WEEK, how="left", validate="1:1"
    ).with_columns(
        pl.when(c("evidence") != "unverified")
        .then(c("evidence"))
        .when(c("roster_teams") > 1)
        .then(pl.lit("conflicting_rosters"))
        .when(c("roster_team") == c("team"))
        .then(pl.lit("same_roster_only"))
        .when(c("roster_team").is_not_null())
        .then(pl.lit("other_roster_only"))
        .otherwise(pl.lit("no_record"))
        .alias("evidence")
    )


def transaction_tables(
    data: dict[str, dict[str, pl.DataFrame]],
    snaps: pl.DataFrame,
    ids: pl.DataFrame,
    rosters: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    observed = pl.concat(
        [
            roster_audit(observed_affiliations(group["board"], snaps, ids), rosters)
            .select(
                *PLAYER_WEEK,
                *SIDE,
                "opener",
                "name",
                "position",
                "cutoff",
                "evidence",
                "actual_team",
                "snap_team",
                "roster_team",
                "roster_codes",
            )
            .with_columns(pl.lit(label).alias("board"))
            for label, group in data.items()
        ]
    )
    after = observed.filter(c("board") == "transactions").select(
        *PLAYER_WEEK, c("team").alias("selected_after")
    )
    cases = observed.filter(
        (c("board") == "club_only") & c("evidence").str.starts_with("other_")
    ).join(after, on=PLAYER_WEEK, how="left", validate="1:1")
    return {
        "affiliation-tiers": observed.group_by("board", "opener", "evidence")
        .len("picks")
        .sort("opener", "board", "evidence"),
        "roster-status": observed.filter(c("evidence").str.ends_with("roster_only"))
        .group_by("board", "opener", "evidence", "roster_codes")
        .len("picks")
        .sort("board", "opener", "evidence", "roster_codes"),
        "reviewed-mismatches": cases.drop("board").sort(*PLAYER_WEEK),
        "unresolved": observed.filter(
            c("evidence").is_in(
                ["no_record", "conflicting_rosters", "conflicting_records"]
            )
        ).sort("board", *PLAYER_WEEK),
    }


def roster_tables(
    data: dict[str, dict[str, pl.DataFrame]], covered: list[int]
) -> dict[str, pl.DataFrame]:
    rows = []
    a, b = data
    for key, counterpart in ((a, b), (b, a)):
        group = data[key]
        board = group["board"]
        other = data[counterpart]["judged"].select(*ENTRY, "position")
        rows.append(
            group["judged"]
            .join(board.select(*ENTRY, "week", "actual_team"), on=ENTRY, validate="1:1")
            .with_columns(
                pl.lit(key).alias("board"), c("season").is_in(covered).alias("covered")
            )
            .join(
                other.with_columns(pl.lit(True).alias("same_pick")),
                on=[*ENTRY, "position"],
                how="left",
            )
        )
    all_rows = pl.concat(rows)
    measures = [
        pl.len().alias("picks"),
        c("correct").sum(),
        c("correct").mean().alias("precision"),
        (c("actual_team") == c("team")).sum().alias("same_team_box"),
        (c("actual_team") != c("team")).sum().alias("other_team_box"),
        c("actual_team").is_null().sum().alias("no_box"),
        c("same_pick").is_null().sum().alias("changed_picks"),
        (c("same_pick").is_null() & c("correct")).sum().alias("changed_correct"),
    ]
    return {
        "selection": all_rows.group_by("board", "opener", "position")
        .agg(measures)
        .sort("opener", "position", "board"),
        "opening-annual": all_rows.filter(c("opener"))
        .group_by("season", "covered", "board", "position")
        .agg(measures)
        .sort("season", "position", "board"),
    }


def rb_tables(pred: pl.DataFrame, oracle: pl.DataFrame) -> dict[str, pl.DataFrame]:
    targets = [(k, "points") for k in NAMES]
    targets.extend(
        (f"{label}_{k}", k)
        for label in ("linear", "forest")
        for k in ("carries", "rushing_tds", "rushing_yards")
    )
    long = pl.concat(
        [
            pred.select(
                "season",
                "opener",
                pl.lit(k).alias("prediction"),
                pl.lit(v).alias("target"),
                (c(k) - c(v)).alias("err"),
            )
            for k, v in targets
        ]
    )
    if not long["err"].is_finite().all() or long["err"].null_count():
        raise DataAcquisitionError("Incomplete RB comparison")
    metric = [
        pl.len().alias("rows"),
        c("err").abs().mean().alias("mae"),
        c("err").pow(2).mean().sqrt().alias("rmse"),
        c("err").mean().alias("bias"),
    ]
    weekly, starters = starter_scores(pred, oracle)
    return {
        "scores": long.group_by("target", "prediction").agg(metric),
        "annual": long.group_by("season", "target", "prediction").agg(metric),
        "opener-scores": long.group_by("opener", "target", "prediction").agg(metric),
        "starters": starters,
        "starter-annual": weekly.group_by("season", "model", "oracle").agg(
            pl.len().alias("weeks"),
            c("hits", "regret", "picked_points", "oracle_points").mean(),
        ),
    }


def _repair_tables(
    earlier: dict[str, pl.DataFrame],
    repaired: dict[str, pl.DataFrame],
    directory_years: list[int],
    snaps: pl.DataFrame,
    ids: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    comparison = roster_tables(
        {"original": earlier, "dated": repaired}, directory_years
    )
    boards = {"directory_only": earlier, "repaired": repaired}
    return {
        "roster-repair": comparison["opening-annual"]
        .rename({"covered": "had_directory"})
        .with_columns(
            c("board").replace({"original": "directory_only", "dated": "repaired"})
        ),
        "affiliation-audit": pl.concat(
            [
                affiliation_audit(v["board"], snaps, ids).with_columns(
                    pl.lit(k).alias("board")
                )
                for k, v in boards.items()
            ]
        ).sort("season", "opener", "board", "evidence"),
    }


def edition_inputs(
    end_year: int,
    players: pl.DataFrame,
    rosters: pl.DataFrame,
    draft: pl.DataFrame,
    *,
    directory_only: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, pl.DataFrame]]:
    directory, meta = opening_editions(end_year)
    tables = {"editions": meta}
    if directory_only:
        return directory, directory, tables
    club, tables["club-revisions"] = club_editions(end_year, players, rosters, draft)
    previous = directory.with_columns(pl.lit(False).alias("club_roster"))
    editions = pl.concat(
        [previous, club.select(previous.columns)], how="vertical_relaxed"
    )
    return directory, editions, tables


def run_opportunity(
    output: Path,
    end_year: int,
    provider: NflDataProvider,
    *,
    directory_only: bool = False,
    transactions: bool = False,
) -> dict[str, object]:
    if not PAPER_YEAR <= end_year <= LAST_COMPLETE:
        raise ConfigurationError("Opportunity holdouts must end in 2014..2025")
    if transactions and directory_only:
        raise ConfigurationError(
            "Transaction comparison needs the published club board"
        )
    output.mkdir(parents=True, exist_ok=True)
    source = ProvenanceProvider(provider)
    raw = matchup_sources(end_year)
    years = tuple(range(HISTORY_START, end_year + 1))
    players_index = provider.load_players()
    roster_index = provider.load_rosters_weekly(years)
    ids = id_map(players_index, roster_index)
    directory, editions, tables = edition_inputs(
        end_year,
        players_index,
        roster_index,
        raw["draft"],
        directory_only=directory_only,
    )
    pairs = (
        (("club_only", editions, None), ("transactions", editions, read_transactions()))
        if transactions
        else (("original", None, None), ("dated", editions, None))
    )
    data = {
        label: prepare_board(
            source,
            end_year,
            raw["trades"],
            raw["draft"],
            raw["snaps"],
            editions=ed,
            transactions=tx,
        )
        for label, ed, tx in pairs
    }
    teams, players, pbp_hashes = opportunity_sources(provider, years, raw["snaps"], ids)
    if data[pairs[0][0]]["sides"].join(teams, on=SIDE, how="anti").height:
        raise DataAcquisitionError("Missing play-by-play for a completed team-game")
    tables.update(roster_tables(data, editions["season"].unique().to_list()))
    if transactions:
        tables.update(transaction_tables(data, raw["snaps"], ids, roster_index))
    elif not directory_only:
        earlier = prepare_board(
            source,
            end_year,
            raw["trades"],
            raw["draft"],
            raw["snaps"],
            editions=directory,
        )
        tables.update(
            _repair_tables(
                earlier,
                data["dated"],
                directory["season"].unique().to_list(),
                raw["snaps"],
                ids,
            )
        )
    tables["play-coverage"] = (
        teams.group_by("season")
        .agg(
            pl.len().alias("team_games"),
            c(COUNTS).sum(),
            (c("inside5_runs") == 0).sum().alias("zero_inside5_rush_games"),
        )
        .sort("season")
    )
    results: dict[str, list[pl.DataFrame]] = {}
    outputs = {}
    coverage = []
    for label, group in data.items():
        selected = (
            group["board"]
            .filter(c("team_role") & (c("season") >= PAPER_YEAR))
            .select(*PLAYER_WEEK, "position", "points")
        )
        universe = pl.concat(
            [
                selected,
                group["universe"]
                .join(selected, on=PLAYER_WEEK, how="anti")
                .select(selected.columns),
            ],
            how="vertical_relaxed",
        ).filter((c("position") == "RB") & (c("season") >= PAPER_YEAR))
        board = rb_opportunities(group["board"], teams, players)
        held = board.filter(c("team_role") & (c("season") >= PAPER_YEAR))
        coverage.append(
            held.group_by("season", "opener")
            .agg(
                pl.len().alias("rows"),
                c("carry_history_known").sum().alias("with_player_history"),
                c("team_runs").count().alias("with_team_history"),
            )
            .with_columns(pl.lit(label).alias("board"))
        )
        blocks = (
            (("carry", CARRY),)
            if transactions
            else (
                ("baseline", ()),
                ("carry", CARRY),
                ("carry_goal", (*CARRY, *GOAL)),
            )
        )
        for block, features in blocks:
            pred = project(board, end_year, extra_features=features, positions=("RB",))
            if not pred.select(*ENTRY, "points").equals(held.select(*ENTRY, "points")):
                raise DataAcquisitionError(
                    "RB forecasts do not match locked role selections"
                )
            filename = f"{label}-{block}.parquet"
            pred.write_parquet(output / filename)
            outputs[filename] = {
                "rows": pred.height,
                "sha256": file_sha256(output / filename),
            }
            for name, frame in rb_tables(pred, universe).items():
                results.setdefault(name, []).append(
                    frame.with_columns(
                        pl.lit(label).alias("board"), pl.lit(block).alias("features")
                    )
                )
    tables["history-coverage"] = pl.concat(coverage).sort("board", "season", "opener")
    for name, frames in results.items():
        tables[name] = pl.concat(frames).sort(
            *[
                k
                for k in (
                    "board",
                    "features",
                    "season",
                    "opener",
                    "target",
                    "prediction",
                    "model",
                    "oracle",
                )
                if k in frames[0].columns
            ]
        )
    for name, frame in tables.items():
        filename = name + ".csv"
        frame.write_csv(output / filename, float_precision=8)
        outputs[filename] = {
            "rows": frame.height,
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "test_years": [PAPER_YEAR, end_year],
        **({"transaction_news_sha256": file_sha256(LEDGER)} if transactions else {}),
        "pbp_input_columns": PBP_COLUMNS,
        "pbp_sha256_by_season": pbp_hashes,
        "source_sha256": {
            **{k: v.sha256 for k, v in source.artifacts.items()},
            **{k: fingerprint_frame(k, v).sha256 for k, v in raw.items()},
        },
        "news_sha256": file_sha256(NEWS_FILE),
        "editions_index_sha256": file_sha256(SOURCES),
        "club_rosters": None
        if directory_only
        else {
            "index_sha256": file_sha256(REVISIONS),
            "facts_sha256": file_sha256(FACTS),
        },
        "outputs": outputs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {"directory": str(output), "test_years": [PAPER_YEAR, end_year]}
