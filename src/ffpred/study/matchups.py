"""Rebuild a point-in-time weekly league draft board and matchup forecasts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import nflreadpy as nfl
import polars as pl

from ffpred.datasets.io import file_sha256
from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.protocol import NflDataProvider
from ffpred.providers.provenance import ProvenanceProvider, fingerprint_frame
from ffpred.study.defense import defense_snapshots
from ffpred.study.matchup_eval import evaluate_matchups
from ffpred.study.matchup_models import project
from ffpred.study.matchup_outcomes import (
    matchup_evidence,
    strict_panel,
    superseded_at_game,
)
from ffpred.study.matchup_pool import (
    SEEDS,
    SNAP_START,
    WEEK,
    id_map,
    opener_seeds,
    played_labels,
    player_transfers,
    resolve_teams,
    trade_events,
    weekly_clocks,
)
from ffpred.study.opening import apply_editions, opening_evidence
from ffpred.study.panel import (
    CHANNELS,
    OTHER,
    PLAYER_WEEK,
    POSITIONS,
    SCORING,
    STATS,
    TEAM_CODES,
    WEIGHTS,
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
    GROUP,
    SLOTS,
    documented_absences,
    histories,
    role_outcomes,
    role_pool,
    select_roles,
)
from ffpred.study.run import HISTORY_START, LAST_COMPLETE, PAPER_YEAR
from ffpred.study.transactions import (
    apply_transactions,
    transaction_facts,
    transaction_seeds,
)

c = pl.col
SCORE_TOLERANCE = 1e-7
Clock = Literal["weekly", "kickoff"]
KICKOFF_NEWS = Path(__file__).with_name("kickoff_news.csv")


def _rank_openers(selected: pl.DataFrame, *, has_editions: bool) -> pl.DataFrame:
    selected = selected.with_columns(
        pl.when(c("opener") & c("depth_at").is_not_null())
        .then(c("depth_rank"))
        .otherwise(None)
        .alias("opening_depth"),
    )
    affiliations = ["club_confirmed", "edition_confirmed"] if has_editions else []
    ranks = (
        selected.filter(~c("ruled_out"))
        .sort(
            [
                "announced",
                "doubtful",
                "opening_depth",
                "current_qb1",
                *affiliations,
                "expected_usage",
                "depth_rank",
                "player_id",
            ],
            descending=[
                True,
                False,
                False,
                True,
                *[True for _ in affiliations],
                True,
                False,
                False,
            ],
            nulls_last=True,
        )
        .with_columns(c("player_id").cum_count().over(GROUP).alias("team_rank"))
        .select(*ENTRY, "team_rank")
    )
    return (
        selected.drop("team_rank", "team_role")
        .join(ranks, on=ENTRY, how="left")
        .with_columns(
            (c("team_rank") <= c("position").replace_strict(SLOTS))
            .fill_null(False)
            .alias("team_role")
        )
    )


def _opening_seeds(
    clocks: pl.DataFrame,
    seeds: pl.DataFrame,
    boxes: pl.DataFrame,
    editions: pl.DataFrame | None,
    *,
    kickoff: bool,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if editions is None:
        return seeds, None
    evidence = opening_evidence(clocks, seeds, boxes, editions, kickoff=kickoff)
    extra = (
        clocks.filter(c("opener"))
        .join(evidence.filter(c("listed")), on=["season", "team"])
        .with_columns(pl.lit("dated_affiliation").alias("roster_source"))
        .select(SEEDS)
    )
    return pl.concat([seeds, extra]).unique(
        ENTRY, keep="first", maintain_order=True
    ), evidence


def prepare_board(  # noqa: PLR0913 - keep the published board's source interface
    provider: ProvenanceProvider,
    end_year: int,
    trades: pl.DataFrame,
    draft: pl.DataFrame,
    snaps: pl.DataFrame,
    *,
    editions: pl.DataFrame | None = None,
    transactions: pl.DataFrame | None = None,
    strict_clock: Clock | None = None,
    pregame_filter: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> dict[str, pl.DataFrame]:
    years = tuple(range(HISTORY_START, end_year + 1))
    schedule = provider.load_schedules(years)
    sides = schedule_sides(schedule)
    kickoff = strict_clock == "kickoff"
    times = game_clocks(sides, schedule, include_openers=True)
    clocks = weekly_clocks(times, kickoff=kickoff)
    rosters = provider.load_rosters_weekly(years)
    boxes = box_scores(provider.load_player_stats(years))
    if sides.join(
        boxes.select("game_id", c("actual_team").alias("team")).unique(),
        on=SIDE,
        how="anti",
    ).height:
        raise DataAcquisitionError("Missing player boxes for a completed matchup")
    ids = id_map(provider.load_players(), rosters)
    injured = provider.load_injuries(years)
    absences = documented_absences(rosters, injured, boxes)
    reports = injury_evidence(clocks, injured)
    depth = depth_evidence(
        clocks,
        provider.load_depth_charts([y for y in years if y < DEPTH_CHANGE]),
        provider.load_depth_charts([DEPTH_CHANGE])
        if end_year >= DEPTH_CHANGE
        else None,
    )
    news_rows = (
        read_news(NEWS_FILE)
        if strict_clock is None
        else pl.concat([read_news(NEWS_FILE), read_news(KICKOFF_NEWS)])
    )
    news = official_news(clocks, news_rows)
    base = candidates(sides, rosters, boxes)
    transfers = player_transfers(trades, ids)
    moves = trade_events(transfers, rosters, clocks)
    seeds = pl.concat(
        [
            moves.with_columns(pl.lit("trade").alias("roster_source")).select(SEEDS),
            base.select(SEEDS),
            opener_seeds(clocks, rosters, draft, boxes),
        ]
    ).unique(ENTRY, keep="first", maintain_order=True)
    facts = (
        transaction_facts(transactions, clocks) if transactions is not None else None
    )
    if facts is not None:
        seeds = transaction_seeds(seeds, facts)
    seeds, evidence = _opening_seeds(clocks, seeds, boxes, editions, kickoff=kickoff)
    pool = role_pool(clocks, seeds, depth, news, reports)
    if facts is not None:
        pool = apply_transactions(pool, facts, boxes, clocks)
    if evidence is not None:
        pool = apply_editions(pool, evidence, transfers, reports)
    if strict_clock is not None:
        pool = pool.filter((c("kind") != "released").fill_null(True))
    pool, removed = resolve_teams(pool, reports, transfers, kickoff=kickoff)
    if pregame_filter is not None:
        if not kickoff:
            raise ConfigurationError(
                "Apply historical pregame availability only at kickoff"
            )
        pool = pregame_filter(pool)
    selected = _rank_openers(
        select_roles(pool, reports, histories(base, boxes, absences)),
        has_editions=editions is not None,
    )
    if strict_clock is None:
        panel = _weekly_panel(selected, boxes, snaps, ids)
    else:
        # Exact game-day announcements can confirm an inactive or a departure
        # after the control's freeze. Use them to label, never to back-edit, it.
        final = weekly_clocks(times, kickoff=True)
        game_news = official_news(final, news_rows)
        superseded = (
            superseded_at_game(
                selected, final, transfers, transaction_facts(transactions, final)
            )
            if not kickoff and transactions is not None
            else selected.select(ENTRY).clear()
        )
        evidence = matchup_evidence(
            selected,
            boxes,
            snaps,
            ids,
            rosters,
            sides=sides,
            game_news=game_news,
            superseded=superseded,
        )
        panel = strict_panel(selected, evidence, boxes)
    panel = (
        panel.join(
            defense_snapshots(sides, provider.load_team_stats(years)),
            on=["game_id", "opponent"],
            validate="m:1",
        )
        .join(
            absences.rename({"week": "roster_week"}).with_columns(
                pl.lit(True).alias("prior_absence")
            ),
            on=["season", "team", "roster_week", "player_id"],
            how="left",
        )
        .with_columns(
            c("prior_absence").fill_null(False),
            (c("designation") == "Questionable").fill_null(False).alias("questionable"),
            (c("practice_status") == "Limited Participation in Practice")
            .fill_null(False)
            .alias("limited"),
            (c("season") * 100 + c("week") - c("history_period")).alias("history_gap"),
            c("depth_at").is_not_null().alias("dated_depth"),
        )
    )
    if (
        panel.select(ENTRY if kickoff else PLAYER_WEEK).is_duplicated().any()
        or panel.filter(
            (
                c("points") - pl.sum_horizontal(c(k) * v for k, v in WEIGHTS.items())
            ).abs()
            > SCORE_TOLERANCE
        ).height
    ):
        raise DataAcquisitionError(
            "Ambiguous identity or unreconciled component ledger"
        )
    outputs = _role_results(
        selected, clocks, schedule, boxes, rosters, absences=absences
    )
    return {
        **outputs,
        "board": panel.sort(*WEEK, "team", "position", "player_id"),
        "removed": removed.sort(*PLAYER_WEEK, "team"),
        "sides": clocks,
    }


def _weekly_panel(
    selected: pl.DataFrame, boxes: pl.DataFrame, snaps: pl.DataFrame, ids: pl.DataFrame
) -> pl.DataFrame:
    panel = player_panel(selected, boxes)
    played = played_labels(panel, snaps, ids, boxes)
    stats = boxes.select(
        *PLAYER_WEEK,
        *STATS,
        *OTHER[:-1],
        (
            (c("special_teams_tds") + c("fumble_recovery_tds")) * SCORING.other_td
            + pl.sum_horizontal(c(f"{ch}_2pt_conversions") for ch in CHANNELS)
            * SCORING.conversion
            + c("receptions") * SCORING.reception
        )
        .round(2)
        .alias("other_points"),
    )
    extra = boxes.select(*PLAYER_WEEK, "points", *STATS[1::2])
    zeros = selected.select(PLAYER_WEEK).join(extra, on=PLAYER_WEEK, how="anti")
    history = (
        pl.concat(
            [
                extra,
                zeros.with_columns(
                    pl.lit(0.0).alias(v) for v in ["points", *STATS[1::2]]
                ).select(extra.columns),
            ],
            how="vertical_relaxed",
        )
        .sort("player_id", *WEEK)
        .with_columns(
            c("points")
            .shift()
            .rolling_std(10, min_samples=2)
            .over("player_id")
            .alias("volatility"),
            *(
                c(v)
                .shift()
                .rolling_mean(10, min_samples=1)
                .over("player_id")
                .alias("lag_" + v)
                for v in STATS[1::2]
            ),
        )
    )
    return (
        panel.join(played, on=PLAYER_WEEK, validate="1:1")
        .join(stats, on=PLAYER_WEEK, how="left")
        .join(
            boxes.select(*PLAYER_WEEK, *CHANNELS.values()), on=PLAYER_WEEK, how="left"
        )
        .with_columns(c([*WEIGHTS, *CHANNELS.values()]).fill_null(0))
        .join(
            history.select(
                *PLAYER_WEEK, "volatility", *(f"lag_{v}" for v in STATS[1::2])
            ),
            on=PLAYER_WEEK,
            validate="1:1",
        )
    )


def _role_results(  # noqa: PLR0913 - separate sources for the evaluation oracle
    selected: pl.DataFrame,
    clocks: pl.DataFrame,
    schedule: pl.DataFrame,
    boxes: pl.DataFrame,
    rosters: pl.DataFrame,
    *,
    absences: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    # Forecast eligible positions take precedence; the independent rest of league
    # uses target game roster position solely to label the evaluation oracle.
    pos = (
        rosters.filter(c("game_type") == "REG")
        .select(
            *WEEK,
            c("team").replace(TEAM_CODES).alias("actual_team"),
            c("gsis_id").alias("player_id"),
            c("position").replace({"FB": "RB", "HB": "RB"}).alias("game_pos"),
        )
        .unique()
    )
    pos = pos.filter(pl.len().over(*PLAYER_WEEK, "actual_team") == 1)
    truth = (
        boxes.join(pos, on=[*PLAYER_WEEK, "actual_team"], how="left")
        .select(
            *PLAYER_WEEK,
            pl.coalesce(
                "game_pos", c("box_position").replace({"FB": "RB", "HB": "RB"})
            ).alias("position"),
            "points",
        )
        .filter(c("position").is_in(POSITIONS))
    )
    realized = role_outcomes(clocks, schedule, boxes, rosters, absences)
    judged = selected.filter(c("team_role") & (c("season") >= PAPER_YEAR)).join(
        realized.select(*ENTRY, "position", pl.lit(True).alias("correct")),
        on=[*ENTRY, "position"],
        how="left",
        validate="1:1",
    )
    selection = (
        judged.group_by("position", "opener")
        .agg(pl.len().alias("selected"), c("correct").count().alias("correct"))
        .with_columns((c("correct") / c("selected")).alias("precision"))
    )
    return {
        "universe": truth,
        "selection": selection.sort("position", "opener"),
        "judged": judged.select(
            *ENTRY, "season", "position", "opener", c("correct").fill_null(False)
        ),
    }


def matchup_sources(end_year: int) -> dict[str, pl.DataFrame]:
    return {
        "trades": nfl.load_trades().filter(c("trade_date").dt.year() <= end_year),
        "draft": nfl.load_draft_picks(list(range(HISTORY_START, end_year + 1))),
        "snaps": nfl.load_snap_counts(list(range(SNAP_START, end_year + 1))),
    }


def run_matchups(
    output: Path, end_year: int, provider: NflDataProvider
) -> dict[str, object]:
    if not PAPER_YEAR <= end_year <= LAST_COMPLETE:
        raise ConfigurationError("Matchup holdouts must end in 2014..2025")
    source = ProvenanceProvider(provider)
    raw = matchup_sources(end_year)
    data = prepare_board(source, end_year, raw["trades"], raw["draft"], raw["snaps"])
    forecasts = project(data["board"], end_year)
    results = evaluate_matchups(forecasts, data)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "forecast.parquet": forecasts,
        "board.parquet": data["board"],
        "universe.parquet": data["universe"],
        "affiliations.csv": data["removed"].filter(c("season") >= PAPER_YEAR),
        "selection.csv": data["selection"],
        **results,
    }
    for name, frame in files.items():
        if name.endswith(".csv"):
            frame.write_csv(output / name)
        else:
            frame.write_parquet(output / name)
    manifest = {
        "test_years": [PAPER_YEAR, end_year],
        "lock": "60 minutes before first weekly kickoff in the completed-game archive",
        "sources": {
            **{k: asdict(v) for k, v in source.artifacts.items()},
            **{k: asdict(fingerprint_frame(k, v)) for k, v in raw.items()},
        },
        "news_sha256": file_sha256(NEWS_FILE),
        "outputs": {
            k: {"rows": v.height, "sha256": file_sha256(output / k)}
            for k, v in files.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {"directory": str(output), "forecasts": forecasts.height}
