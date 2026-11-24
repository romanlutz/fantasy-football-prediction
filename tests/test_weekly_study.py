import gzip
import io
import json
import tarfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.fakes import FakeProvider
from ffpred.study import club_rosters, matchups, opening, opportunity_study
from ffpred.study.benchmark import BLOCKS, backtest
from ffpred.study.bridge import read_gamecenter
from ffpred.study.defense import ADJUSTED, defense_snapshots
from ffpred.study.matchup_eval import K, starter_scores
from ffpred.study.matchup_models import NAMES, project
from ffpred.study.matchup_outcomes import (
    matchup_evidence,
    strict_panel,
    superseded_at_game,
)
from ffpred.study.matchup_pool import player_transfers, resolve_teams, weekly_clocks
from ffpred.study.methods import CHALLENGERS, KEY, REFERENCE, compare_methods
from ffpred.study.opening import apply_editions, opening_evidence
from ffpred.study.opportunities import (
    CARRY,
    GOAL,
    count_opportunities,
    include_snap_zeros,
    rb_opportunities,
)
from ffpred.study.panel import (
    PLAYER_FEATURES,
    PLAYER_WEEK,
    POSITIONS,
    SCORING,
    WEIGHTS,
    box_scores,
    candidates,
    player_panel,
    schedule_sides,
)
from ffpred.study.role_evidence import (
    ENTRY,
    depth_evidence,
    game_clocks,
    official_news,
    read_news,
)
from ffpred.study.roles import run_roles
from ffpred.study.run import run_study
from ffpred.study.transactions import (
    apply_transactions,
    available_at,
    transaction_facts,
    transaction_seeds,
)


def provider() -> FakeProvider:
    games, players, rosters, teams = [], [], [], []
    for year in (2010, 2014):
        for week in (1, 2, 4):
            game = f"{year}_{week}_ARI_LV"
            games.append(
                {
                    "season": year,
                    "week": week,
                    "game_type": "REG",
                    "game_id": game,
                    "home_team": "OAK",
                    "away_team": "ARI",
                    "home_score": 17,
                    "away_score": 10,
                }
            )
            for side in ("ARZ", "OAK"):
                rosters.extend(
                    {
                        "season": year,
                        "week": week,
                        "team": side,
                        "game_type": "REG",
                        "gsis_id": side + pos,
                        "status_description_abbr": "A01",
                        "full_name": side + pos,
                        "position": pos,
                    }
                    for pos in POSITIONS
                )
                team = "ARI" if side == "ARZ" else "LV"
                teams.append(
                    {
                        "game_id": game,
                        "team": team,
                        "season_type": "REG",
                        "attempts": 20,
                        "sacks_suffered": 2,
                        "carries": 15,
                        "passing_yards": 175,
                        "rushing_yards": 70,
                        "rushing_10": 2,
                        "passing_20": 1,
                        "passing_epa": 5,
                        "rushing_epa": -2,
                        "passing_interceptions": 1,
                        "fumbles_lost_total": 0,
                    }
                )
                for pos in POSITIONS:
                    if pos == "TE" and week != 1:
                        continue
                    parts = dict.fromkeys(SCORING.points().meta.root_names(), 0)
                    parts.update(
                        passing_yards=100,
                        receiving_yards=18,
                        receiving_tds=1,
                        fumbles_lost_total=1,
                    )
                    players.append(
                        dict(
                            season=year,
                            week=week,
                            game_id=game,
                            season_type="REG",
                            team=team,
                            player_id=side + pos,
                            player_display_name=side + pos,
                            position="TE",
                            attempts=10,
                            targets=1,
                            carries=0,
                            fumbles_total=2,
                            **parts,
                        )
                    )
    return FakeProvider(
        schedules=pl.DataFrame(games),
        team_stats=pl.DataFrame(teams),
        rosters_weekly=pl.DataFrame(rosters),
        player_stats=pl.DataFrame(players),
    )


def test_pool_role_scoring_and_strictly_prior_histories() -> None:
    p = provider()
    sides = schedule_sides(p.schedules)
    scores = box_scores(p.player_stats)
    pool = candidates(sides, p.rosters_weekly, scores)
    panel = player_panel(pool, scores)
    row = panel.filter((pl.col("player_id") == "ARZQB") & (pl.col("week") == 4)).row(
        0, named=True
    )
    assert row["position"] == "QB"  # Historical roster, not the box's TE label
    assert row["roster_week"] == row["history_week"] == 2  # Week 3 was a bye
    assert row["points"] == pytest.approx(9.8)
    assert row["team"] == "ARI"
    te = panel.filter(pl.col("player_id") == "ARZTE")
    assert not te["has_box"].any()
    assert te["points"].sum() == 0
    assert te.filter(pl.col("week") == 4)["last1_points"].to_list() == [0, 0]
    assert set(panel["week"]) == {2, 4}

    changed = p.player_stats.with_columns(
        pl.when(pl.col("week") == 4)
        .then(9000)
        .otherwise(pl.col("passing_yards"))
        .alias("passing_yards")
    )
    after = player_panel(pool, box_scores(changed))
    before_test = (pl.col("season") == 2010) & (pl.col("week") == 4)
    assert_frame_equal(
        panel.filter(before_test).select(PLAYER_FEATURES),
        after.filter(before_test).select(PLAYER_FEATURES),
    )
    roster_change = p.rosters_weekly.with_columns(
        pl.when(pl.col("week") == 4)
        .then(pl.lit("R01"))
        .otherwise(pl.col("status_description_abbr"))
        .alias("status_description_abbr")
    )
    assert_frame_equal(
        pool.sort(PLAYER_WEEK),
        candidates(sides, roster_change, scores).sort(PLAYER_WEEK),
    )
    # Mixed coverage in the same team-game: known codes take priority, and a
    # missing code requires prior box evidence, not target-game participation.
    missing_codes = p.rosters_weekly.with_columns(
        pl.when((pl.col("week") == 2) & (pl.col("position") == "WR"))
        .then(pl.lit("R01"))
        .when((pl.col("week") == 2) & (pl.col("position") != "RB"))
        .then(None)
        .otherwise(pl.col("status_description_abbr"))
        .alias("status_description_abbr")
    )
    fallback = candidates(sides, missing_codes, scores).filter(pl.col("week") == 4)
    assert set(fallback.filter(pl.col("roster_source") == "prior_box")["position"]) == {
        "QB"
    }
    assert set(fallback["position"]) == {"QB", "RB"}


def test_defense_updates_only_after_entire_week_and_adjusts_for_offense() -> None:
    p = provider()
    sides = schedule_sides(p.schedules)
    base = defense_snapshots(sides, p.team_stats)
    # Change one offense against the same defense; only subsequent weeks may change.
    changed = p.team_stats.with_columns(
        pl.when((pl.col("game_id") == "2010_2_ARI_LV") & (pl.col("team") == "LV"))
        .then(400)
        .otherwise(pl.col("rushing_yards"))
        .alias("rushing_yards")
    )
    new = defense_snapshots(sides, changed)
    current = pl.col("game_id") == "2010_2_ARI_LV"
    assert_frame_equal(base.filter(current), new.filter(current))
    following = (pl.col("game_id") == "2010_4_ARI_LV") & (pl.col("opponent") == "ARI")
    assert new.filter(following)["opp_last1_rushing_yards"][0] == 400
    assert (
        new.filter(following)[ADJUSTED[0]][0] > base.filter(following)[ADJUSTED[0]][0]
    )
    with pytest.raises(DataAcquisitionError, match="Incomplete team box"):
        defense_snapshots(sides, p.team_stats.slice(1))


def test_incomplete_feeds_do_not_become_zero_outcomes(tmp_path: Path) -> None:
    p = provider()
    with pytest.raises(DataAcquisitionError, match="Missing prior team roster"):
        candidates(
            schedule_sides(p.schedules),
            p.rosters_weekly.filter(pl.col("team") == "OAK"),
            box_scores(p.player_stats),
        )
    with pytest.raises(DataAcquisitionError, match="Non-unique"):
        box_scores(pl.concat([p.player_stats, p.player_stats.head(1)]))
    p.player_stats = p.player_stats.filter(
        ~((pl.col("game_id") == "2014_2_ARI_LV") & (pl.col("team") == "LV"))
    )
    with pytest.raises(DataAcquisitionError, match="Missing player box"):
        run_study(tmp_path, 2014, p, paper_source=legacy_fixture)


def legacy_fixture() -> pl.DataFrame:
    buf = io.BytesIO()
    passing = {
        "name": "Old QB",
        "att": 10,
        "yds": 100,
        "tds": 0,
        "ints": 0,
        "twoptm": 0,
    }
    side = {"abbr": "ARI", "stats": {"passing": {"ARZQB": passing}}}
    other = {"abbr": "OAK", "stats": {"passing": {"OAKQB": {**passing, "att": 4}}}}
    files = {
        "schedule.json": json.dumps(
            {"games": [["g", {"year": 2014, "week": 2, "season_type": "REG"}]]}
        ).encode(),
        "gamecenter-json/g.json.gz": gzip.compress(
            json.dumps({"g": {"home": side, "away": other}}).encode()
        ),
    }
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for path, content in files.items():
            info = tarfile.TarInfo("nflgame-1.2.15/nflgame/" + path)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return read_gamecenter(buf.getvalue())


def test_study_replay_audit_and_paired_predictions(tmp_path: Path) -> None:
    p = provider()
    result = run_study(tmp_path, 2014, p, paper_source=legacy_fixture)
    assert result["legacy_2014_rows"] == 1
    audit = pl.read_csv(tmp_path / "reconciliation-2014.csv")
    row = audit.filter(pl.col("membership") == "both").row(0, named=True)
    assert row["box_position"] == "TE" and row["position"] == "QB"
    assert row["paper_points"] == 4
    assert row["points"] - row["paper_points"] == pytest.approx(
        row["source_revision_delta"]
        + row["fumble_rule_delta"]
        + row["other_scoring_delta"]
    )
    annual = pl.read_csv(tmp_path / "scores-by-season.csv")
    assert len(annual) == 4 * 7
    assert annual["player_games"].unique().to_list() == [4]
    panel = pl.read_parquet(tmp_path / "player-games.parquet")
    pred = pl.read_parquet(tmp_path / "predictions.parquet")
    # Swapping test targets without altering the inputs cannot change predictions.
    modified = panel.with_columns(
        pl.when(pl.col("season") == 2014)
        .then(800)
        .otherwise(pl.col("points"))
        .alias("points")
    )
    value_cols = ["trailing10", *(f"{m}_{b}" for m in ("svr", "boost") for b in BLOCKS)]
    assert_frame_equal(
        pred.select(value_cols),
        backtest(modified, range(2014, 2015)).select(value_cols),
    )
    with pytest.raises(ConfigurationError):
        run_study(tmp_path, 2026, p)
    assert backtest(panel, range(2014, 2015), exact_svr=True).height == pred.height


def role_provider() -> FakeProvider:
    p = provider()
    c = pl.col
    p.schedules = p.schedules.with_columns(
        pl.concat_str(
            c("season"),
            pl.lit("-09-"),
            (c("week") * 7).cast(pl.String).str.pad_start(2, "0"),
        ).alias("gameday"),
        pl.lit("13:00").alias("gametime"),
        pl.lit("OAKQB").alias("home_qb_id"),
        pl.when((c("season") == 2014) & (c("week") == 4))
        .then(pl.lit("NEW"))
        .otherwise(pl.lit("ARZQB"))
        .alias("away_qb_id"),
    )
    p.rosters_weekly = p.rosters_weekly.with_columns(
        pl.when((c("gsis_id") == "ARZRB") & (c("week") == 2))
        .then(pl.lit("R01"))
        .otherwise(c("status_description_abbr"))
        .alias("status_description_abbr")
    )
    p.player_stats = p.player_stats.filter(
        ~((c("player_id") == "ARZRB") & (c("week") == 2))
    )
    p.depth_charts = p.rosters_weekly.select(
        "season",
        "week",
        "game_type",
        c("team").alias("club_code"),
        "gsis_id",
        "full_name",
        "position",
        pl.lit("1").alias("depth_team"),
    )
    p.injuries = pl.DataFrame(
        {
            "season": [2014] * 3,
            "week": [4] * 3,
            "team": ["ARI", "ARI", "LV"],
            "game_type": ["REG"] * 3,
            "gsis_id": ["ARZRB", "ARZQB", "OAKWR"],
            "full_name": ["ARZRB", "ARZQB", "OAKWR"],
            "position": ["RB", "QB", "WR"],
            "report_status": ["Questionable", "Out", "Out"],
            "practice_status": ["Full Participation in Practice", None, None],
            "report_primary_injury": ["knee"] * 3,
            "practice_primary_injury": [None] * 3,
            # 13:00 Eastern in September: the cutoff is 16:00 UTC.
            "date_modified": [
                datetime(2014, 9, 28, 15, 0, tzinfo=UTC),
                datetime(2014, 9, 28, 16, 0, tzinfo=UTC),
                datetime(2014, 9, 28, 16, 1, tzinfo=UTC),
            ],
        }
    )
    return p


def test_roles_freeze_news_keep_returns_and_audit_misses(tmp_path: Path) -> None:
    p = role_provider()
    csv = tmp_path / "news.csv"
    csv.write_text(
        "game_id,team,player_id,name,position,kind,published_at,source,max_opportunities\n"
        "2014_4_ARI_LV,ARI,NEW,New,QB,starter,2014-09-28T15:55:00Z,https://example.org/1,\n"
        "2014_4_ARI_LV,ARI,NEW,New,QB,inactive,2014-09-28T16:01:00Z,https://example.org/2,\n"
        "2014_4_ARI_LV,LV,OAKRB,OAKRB,RB,limited,2014-09-27T12:00:00Z,https://example.org/3,0\n",
        encoding="utf-8",
    )
    run_roles(tmp_path, 2014, p, news_file=csv, compare_models=True)
    c = pl.col
    rows = pl.read_parquet(tmp_path / "candidates.parquet").filter(c("week") == 4)
    selected = rows.filter(c("team_role"))
    assert selected.filter(c("player_id") == "NEW")["announced"].item()
    assert not rows.filter(c("player_id") == "ARZQB")["team_role"].item()
    assert selected.filter(c("player_id") == "OAKWR")["report_at"].item() is None
    rb = selected.filter(c("player_id") == "ARZRB").row(0, named=True)
    assert (
        rb["roster_source"] is None
        and rb["practice_status"] == "Full Participation in Practice"
    )
    assert rb["prior3_targets"] == 1  # Previous injury absence was not a zero.
    assert selected.filter(c("player_id") == "OAKRB")["expected_usage"].item() == 0
    truth = pl.read_parquet(tmp_path / "realized-roles.parquet")
    assert truth.filter((c("player_id") == "ARZRB") & (c("week") == 4))[
        "injury_return"
    ].item()
    preds = pl.read_parquet(tmp_path / "predictions.parquet")
    # A rostered game with no box is still scored, not excluded in hindsight.
    assert preds.filter(c("player_id") == "NEW")["points"].item() == 0
    methods = pl.read_parquet(tmp_path / "method-predictions.parquet")
    assert_frame_equal(
        methods.select(*KEY, "points", *REFERENCE).sort(KEY),
        preds.filter(c("team_role")).select(*KEY, "points", *REFERENCE).sort(KEY),
    )
    audit = pl.read_csv(tmp_path / "methods-role-audit.csv").filter(
        (c("position") == "RB") & (c("model") == "ridge")
    )
    assert (
        audit.filter(c("cohort") == "missed_previous_game_injured")["slots"].item() == 1
    )
    assert audit.filter(c("cohort") == "realized_return_role")["slots"].item() == 1
    selection = pl.read_csv(tmp_path / "selection-pooled.csv")
    assert (selection["selected"] <= selection["planned"]).all()
    assert selection.filter(c("position") == "RB")["correct_returns"].item() == 1
    assert pl.read_csv(tmp_path / "scores-pooled.csv").height == 56
    invalid = read_news(csv).with_columns(pl.lit("2014-09-28").alias("published_at"))
    clocks = game_clocks(schedule_sides(p.schedules), p.schedules)
    with pytest.raises(DataAcquisitionError, match="Invalid timestamped"):
        official_news(clocks, invalid)


def test_compared_methods_never_fit_the_holdout() -> None:
    c = pl.col
    p = provider()
    sides = schedule_sides(p.schedules)
    boxes = box_scores(p.player_stats)
    panel = (
        player_panel(candidates(sides, p.rosters_weekly, boxes), boxes)
        .join(defense_snapshots(sides, p.team_stats), on=["game_id", "opponent"])
        .filter(c("position") == "QB")
        .with_columns(pl.lit(True).alias("team_role"))
    )
    test = panel.filter(c("season") == 2014)
    ref = test.select(*KEY, "points", "team_role").with_columns(
        pl.lit(0.0).alias(name) for name in REFERENCE
    )
    with pytest.raises(DataAcquisitionError, match="identical team-role"):
        compare_methods(panel, test, ref.slice(1))
    with pytest.raises(DataAcquisitionError, match="identical team-role"):
        compare_methods(panel, test, ref.with_columns(c("points") + 10))
    baseline = compare_methods(panel, test, ref)
    changed = compare_methods(
        panel.with_columns(
            pl.when(c("season") == 2014)
            .then(500)
            .otherwise(c("points"))
            .alias("points")
        ),
        test.with_columns(c("points") + 10),
        ref.with_columns(c("points") + 10),
    )
    assert_frame_equal(
        baseline["method-predictions.parquet"].select(CHALLENGERS),
        changed["method-predictions.parquet"].select(CHALLENGERS),
    )
    assert baseline["methods-exact-rbf-2014.csv"]["model"].to_list() == [
        "rbf_approx",
        "rbf_exact",
    ]
    paired = baseline["methods-paired-mae.csv"].filter(c("model") == "ridge")
    assert paired["low95"].item() == pytest.approx(paired["mae_delta"].item())


def test_daily_depth_is_a_whole_snapshot_before_the_cutoff() -> None:
    p = role_provider()
    clocks = (
        game_clocks(schedule_sides(p.schedules), p.schedules)
        .filter(
            (pl.col("season") == 2014)
            & (pl.col("week") == 4)
            & (pl.col("team") == "ARI")
        )
        .with_columns(pl.lit(2025).alias("season"))
    )
    depth = pl.DataFrame(
        {
            "dt": [
                "2014-09-27T17:00:00Z",
                "2014-09-28T15:30:00Z",
                "2014-09-28T16:01:00Z",
            ],
            "team": ["ARI"] * 3,
            "gsis_id": ["departed", "known", "future"],
            "player_name": ["A", "B", "C"],
            "pos_abb": ["QB"] * 3,
            "pos_rank": [1] * 3,
        }
    )
    assert depth_evidence(clocks, p.depth_charts, depth)["player_id"].to_list() == [
        "known"
    ]


def test_only_known_player_moves_change_a_weekly_assignment() -> None:
    c = pl.col
    raw = pl.DataFrame(
        {
            "pfr_id": ["A", "A"],
            "pfr_name": ["a", "a"],
            "received": ["SEA", "SF"],
            "trade_date": [date(2022, 4, 20), date(2022, 10, 20)],
            "pick_season": [2022, None],
            "pick_round": [2, None],
        }
    )
    ids = pl.DataFrame({"pfr_id": ["A"], "player_id": ["1"]})
    moves = player_transfers(raw, ids)
    assert moves.select("team", "move_at").rows() == [
        ("SF", datetime(2022, 10, 21, 4, tzinfo=UTC))
    ]
    t = datetime(2022, 10, 20, 23, tzinfo=UTC)
    old = pl.DataFrame(
        {
            "season": [2022],
            "week": [7],
            "game_id": ["g"],
            "team": ["CAR"],
            "player_id": ["1"],
            "prior_at": [datetime(2022, 10, 16, 19, tzinfo=UTC)],
            "cutoff": [t],
        }
    ).with_columns(
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(k)
        for k in ("depth_at", "identity_at")
    )
    reports = old.select(
        "game_id", "team", "player_id", c("depth_at").alias("report_at")
    ).clear()
    assert resolve_teams(old, reports, moves)[0]["team"].to_list() == ["CAR"]
    both = pl.concat(
        [
            old,
            old.with_columns(pl.lit("SF").alias("team"), pl.lit("g2").alias("game_id")),
        ]
    ).with_columns(pl.lit(datetime(2022, 10, 21, 5, tzinfo=UTC)).alias("cutoff"))
    fixed, removed = resolve_teams(both, reports, moves)
    assert fixed["team"].to_list() == ["SF"] and removed["stale_trade"].item()


def test_weekly_oracle_credits_boundary_ties_not_every_tied_player() -> None:
    k = K["QB"]
    pts = [20.0] * (k - 1) + [10.0, 10.0, 10.0]
    p = pl.DataFrame(
        {
            "season": [2014] * len(pts),
            "week": [2] * len(pts),
            "position": ["QB"] * len(pts),
            "player_id": [str(i) for i in range(len(pts))],
            "points": pts,
        }
    ).with_columns(
        pl.Series(name, [0.0, *range(1, len(pts))], dtype=pl.Float64) for name in NAMES
    )
    u = (
        p.select("season", "week", "player_id", "position", "points")
        .head(1)
        .with_columns(
            pl.lit("unprojected").alias("player_id"), pl.lit(30.0).alias("points")
        )
    )
    weekly, _ = starter_scores(p, u)
    roles = weekly.filter(pl.col("oracle") == "roles").head(1)
    league = weekly.filter(pl.col("oracle") == "league").head(1)
    assert roles["hits"].item() == 10 and roles["regret"].item() == 20
    assert league["hits"].item() == 9 and league["regret"].item() == 40


def test_an_earlier_prediction_survives_a_later_teams_deadline() -> None:
    early = datetime(2022, 10, 20, 23, tzinfo=UTC)
    later = early + timedelta(days=3)
    games = pl.DataFrame(
        {
            "season": [2022] * 2,
            "week": [7] * 2,
            "game_id": ["g1", "g2"],
            "team": ["CAR", "SF"],
            "cutoff": [early, later],
            "roster_week": [6, 6],
        }
    )
    assert weekly_clocks(games)["cutoff"].to_list() == [early, early]
    pool = weekly_clocks(games, kickoff=True).with_columns(
        pl.lit("runner").alias("player_id"),
        pl.lit(early - timedelta(days=4)).alias("prior_at"),
        *(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(k)
            for k in ("identity_at", "depth_at")
        ),
    )
    assert pool["cutoff"].to_list() == [
        t + timedelta(hours=1, microseconds=-1) for t in (early, later)
    ]
    reports = pool.select(*ENTRY, pl.col("depth_at").alias("report_at")).clear()
    moved = pl.DataFrame(
        {
            "player_id": ["runner"],
            "team": ["SF"],
            "move_at": [early + timedelta(days=1)],
        }
    )
    kept, _ = resolve_teams(pool, reports, moved, kickoff=True)
    assert set(kept["game_id"]) == {"g1", "g2"}
    # An exactly-at-kickoff announcement is already too late on the later side.
    news = pl.DataFrame(
        [
            ("g1", "CAR", "runner", early + timedelta(minutes=59)),
            ("g2", "SF", "runner", later + timedelta(hours=1)),
        ],
        schema=[*ENTRY, "stamp"],
        orient="row",
    ).with_columns(
        pl.col("stamp").dt.strftime("%Y-%m-%dT%H:%M:%SZ").alias("published_at"),
        pl.lit("Runner").alias("name"),
        pl.lit("RB").alias("position"),
        pl.lit("inactive").alias("kind"),
        pl.lit("https://example.org/").alias("source"),
        pl.lit(None, dtype=pl.Float64).alias("max_opportunities"),
    )
    assert official_news(pool.select("team", "game_id", "cutoff"), news)[
        "team"
    ].to_list() == ["CAR"]


def test_only_exact_game_and_team_targets_are_scored() -> None:
    p, _ = matchup_fixture()
    c = pl.col
    names = [
        "switch",
        "wrong_game",
        "snap",
        "roster",
        "missing",
        "other_roster",
        "cut",
        "traded",
    ]
    pool = pl.DataFrame(
        {
            "game_id": ["old"] * len(names) + ["new"],
            "team": ["ARI"] * len(names) + ["LV"],
            "player_id": [*names, "switch"],
        }
    ).with_columns(
        pl.lit(2014).alias("season"),
        pl.lit(3).alias("week"),
        pl.lit(201403).alias("period"),
    )
    template = box_scores(p.player_stats).head(1)
    boxes = pl.concat(
        [
            template.with_columns(
                pl.lit("switch").alias("player_id"),
                pl.lit("new").alias("game_id"),
                pl.lit("LV").alias("actual_team"),
            ),
            template.with_columns(
                pl.lit("wrong_game").alias("player_id"),
                pl.lit("other").alias("game_id"),
                pl.lit("ARI").alias("actual_team"),
            ),
        ]
    ).with_columns(pl.lit(2014).alias("season"), pl.lit(3).alias("week"))
    snaps = pl.DataFrame(
        {
            "game_id": ["old", "new"],
            "team": ["ARI", "LV"],
            "pfr_player_id": ["snap", "switch"],
        }
    ).with_columns(
        pl.lit("REG").alias("game_type"),
        pl.lit(2014).alias("season"),
        pl.lit(3).alias("week"),
        pl.lit(1).alias("offense_snaps"),
        pl.lit(0).alias("defense_snaps"),
        pl.lit(0).alias("st_snaps"),
    )
    ids = pool.select("player_id", c("player_id").alias("pfr_id")).unique()
    rosters = pl.DataFrame(
        {"gsis_id": ["roster", "other_roster", "traded"], "team": ["ARI", "LV", "ARI"]}
    ).with_columns(
        pl.lit("REG").alias("game_type"),
        pl.lit(2014).alias("season"),
        pl.lit(3).alias("week"),
    )
    sides = pool.select("season", "week", "game_id", "team").unique()
    news = pool.filter(c("player_id") == "cut").select(
        *ENTRY, pl.lit("released").alias("kind")
    )
    freeze = datetime(2014, 9, 17, 23, tzinfo=UTC)
    final = sides.with_columns(pl.lit(freeze + timedelta(days=4)).alias("cutoff"))
    moved = pl.DataFrame(
        {
            "player_id": ["traded"],
            "team": ["LV"],
            "move_at": [freeze + timedelta(days=1)],
        }
    )
    facts = news.clear().select(
        *ENTRY, c("kind").alias("state"), pl.lit(freeze).alias("move_at")
    )
    superseded = superseded_at_game(
        pool.with_columns(pl.lit(freeze).alias("cutoff")), final, moved, facts
    )
    evidence = matchup_evidence(
        pool,
        boxes,
        snaps,
        ids,
        rosters,
        sides=sides,
        game_news=news,
        superseded=superseded,
    )
    panel = strict_panel(pool, evidence, boxes)
    foreign = panel.filter((c("player_id") == "switch") & (c("team") == "ARI"))
    assert foreign.select("outcome_status", "points", "played", "carries").row(0) == (
        "other_box",
        None,
        None,
        None,
    )
    good = panel.filter((c("player_id") == "switch") & (c("team") == "LV"))
    assert good.select("points", "played").row(0) == (boxes["points"][0], True)
    zeros = panel.filter(c("player_id").is_in(["snap", "roster"])).sort("player_id")
    assert zeros.select("points", "played").rows() == [(0.0, False), (0.0, True)]
    excluded = panel.filter(c("points").is_null())
    assert set(excluded["outcome_status"]) == {
        "unverified",
        "other_box",
        "other_roster",
        "superseded_release",
        "superseded_move",
    }
    assert not excluded["played"].count()


def test_transactions_respect_the_common_lock_and_do_not_create_playing_status() -> (
    None
):
    c = pl.col
    t = datetime(2022, 10, 20, 23, 15, tzinfo=UTC)
    assert available_at("2022-10-20").astimezone(UTC) == datetime(
        2022, 10, 21, 4, tzinfo=UTC
    )
    with pytest.raises(DataAcquisitionError, match="offset"):
        available_at("2022-10-20T19:00:00")
    clocks = pl.DataFrame(
        [
            {
                "season": 2022,
                "week": w,
                "team": team,
                "game_id": f"g{w}",
                "cutoff": t + timedelta(days=7 * (w - 1)),
            }
            for w in (1, 2, 3)
            for team in ("CAR", "SF")
        ]
    )
    events = pl.DataFrame(
        [
            ("cut", "release", None, t, None),
            ("cut", "agreement", "SF", t + timedelta(seconds=1), None),
            ("ps", "practice", "SF", t, None),
            ("ban", "opening_suspension", None, t - timedelta(days=40), 2),
        ],
        schema=["player_id", "kind", "new_team", "move_at", "games"],
        orient="row",
    ).with_columns(
        pl.lit("RB").alias("position"),
        pl.lit("player").alias("name"),
        pl.lit(2022).alias("season"),
        pl.lit("CAR").alias("old_team"),
    )
    facts = transaction_facts(events, clocks)
    seeds = (
        facts.select("game_id", "team", "player_id", "position", "name")
        .clear()
        .with_columns(pl.lit(None, dtype=pl.String).alias("roster_source"))
    )
    assert transaction_seeds(seeds, facts).select(
        "game_id", "team", "player_id"
    ).rows() == [("g2", "SF", "cut")]
    pool = (
        clocks.join(events.select("player_id").unique(), how="cross")
        .with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(k)
            for k in ("identity_at", "depth_at", "news_at")
        )
        .with_columns(pl.lit(None, dtype=pl.String).alias("kind"))
    )
    # A target-week box cannot reverse a release; it can establish a return for
    # the following week's board. The control has no box at all.
    boxes = pool.filter(
        (c("week") == 2) & (c("team") == "CAR") & (c("player_id") == "cut")
    ).rename({"team": "actual_team"})
    fixed = apply_transactions(pool, facts, boxes, clocks)
    assert fixed.filter((c("player_id") == "cut") & (c("team") == "CAR"))[
        "week"
    ].to_list() == [3]
    assert (
        not apply_transactions(pool, facts, boxes.clear(), clocks)
        .filter((c("player_id") == "cut") & (c("team") == "CAR"))
        .height
    )
    assert not fixed.filter((c("player_id") == "ps") & (c("week") == 1)).height
    assert fixed.filter((c("team") == "CAR") & (c("player_id") == "ban"))[
        "week"
    ].to_list() == [3]
    arrival = fixed.filter(
        (c("team") == "SF") & (c("week") == 2) & (c("player_id") == "cut")
    )
    assert arrival["identity_at"].item() == t + timedelta(seconds=1)
    assert arrival["kind"].item() is None


def test_roster_corroboration_cannot_turn_into_evidence_of_play() -> None:
    observed = pl.DataFrame(
        {
            "season": [2014] * 5,
            "week": [1] * 5,
            "team": ["LV"] * 5,
            "player_id": ["a", "b", "c", "d", "e"],
            "evidence": [
                "unverified",
                "unverified",
                "unverified",
                "same_box",
                "unverified",
            ],
        }
    )
    rosters = pl.DataFrame(
        {
            "gsis_id": ["a", "b", "b", "c", "d", "e"],
            "team": ["OAK", "BUF", "NYJ", "BUF", "BUF", "OAK"],
            "status_description_abbr": [None, "A01", "I01", "P07", "A01", "A01"],
            "game_type": ["REG"] * 5 + ["PRE"],
        }
    ).with_columns(pl.lit(2014).alias("season"), pl.lit(1).alias("week"))
    result = opportunity_study.roster_audit(observed, rosters).sort("player_id")
    assert result["evidence"].to_list() == [
        "same_roster_only",
        "conflicting_rosters",
        "other_roster_only",
        "same_box",
        "no_record",
    ]
    assert result["roster_codes"][0] in ("", None)


def matchup_fixture() -> tuple[FakeProvider, dict[str, pl.DataFrame]]:
    p = role_provider()
    c = pl.col
    for field in (
        "rosters_weekly",
        "player_stats",
        "team_stats",
        "depth_charts",
        "schedules",
    ):
        f = getattr(p, field)
        early = f.filter(
            c("game_id").str.starts_with("2010")
            if field == "team_stats"
            else c("season") == 2010
        )
        cols = (
            [c("season").replace({2010: 2013}).alias("season")]
            if "season" in f.columns
            else []
        )
        cols.extend(
            c(k).str.replace("2010", "2013")
            for k in ("game_id", "gameday")
            if k in f.columns
        )
        setattr(p, field, pl.concat([f, early.with_columns(cols)]))
    p.rosters_weekly = p.rosters_weekly.with_columns(c("gsis_id").alias("pfr_id"))
    p.players = p.rosters_weekly.select("gsis_id", "pfr_id").unique()
    snaps = p.rosters_weekly.filter(
        (c("season") >= 2013) & (c("position") != "TE")
    ).select(
        "season",
        "week",
        "game_type",
        c("team").replace({"ARZ": "ARI", "OAK": "LV"}),
        c("gsis_id").alias("pfr_player_id"),
        pl.lit(1).alias("offense_snaps"),
        pl.lit(0).alias("defense_snaps"),
        pl.lit(0).alias("st_snaps"),
    )
    draft = (
        p.rosters_weekly.filter(c("season") == 2014)
        .select(
            "season",
            "team",
            "position",
            "gsis_id",
            c("full_name").alias("pfr_player_name"),
        )
        .unique()
    )
    raw = pl.DataFrame(
        schema={
            "pfr_id": pl.String,
            "trade_date": pl.Date,
            "pick_season": pl.Int32,
            "pick_round": pl.Int32,
            "received": pl.String,
            "pfr_name": pl.String,
        }
    )
    return p, {"trades": raw, "draft": draft, "snaps": snaps}


def test_matchup_replay_scoring_intervals_and_holdout_invariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, raw = matchup_fixture()
    c = pl.col
    monkeypatch.setattr(matchups, "matchup_sources", lambda end_year: raw)
    for pos in POSITIONS:
        monkeypatch.setitem(K, pos, 1)
    matchups.run_matchups(tmp_path, 2014, p)
    pred = pl.read_parquet(tmp_path / "forecast.parquet")
    assert (
        pred.filter(c("week") == 1).height
        and not pred.select(PLAYER_WEEK).is_duplicated().any()
    )
    assert pred.filter(
        (c("q10") > c("q25")) | (c("q25") > c("q75")) | (c("q75") > c("q90"))
    ).is_empty()
    ledger = pl.sum_horizontal(c("forest_" + k) * v for k, v in WEIGHTS.items())
    assert pred.select((ledger - c("factor_forest")).abs().max()).item() < 1e-7
    board = pl.read_parquet(tmp_path / "board.parquet")
    changed = project(
        board.with_columns(
            pl.when(c("season") == 2014)
            .then(500.0)
            .otherwise(c("points"))
            .alias("points")
        ),
        2014,
    )
    assert_frame_equal(
        changed.select(*NAMES, "q10", "q90", "play_probability"),
        pred.select(*NAMES, "q10", "q90", "play_probability"),
    )


def test_opener_evidence_is_dated_and_absence_is_not_a_release() -> None:
    c = pl.col
    lock = datetime(2024, 9, 5, 23, tzinfo=UTC)
    clocks = pl.DataFrame(
        {
            "game_id": ["g", "g"],
            "team": ["SEA", "SF"],
            "season": [2024, 2024],
            "week": [1, 1],
            "cutoff": [lock, lock],
            "opener": [True, True],
        }
    )
    # The old team has all six players. A dated directory explicitly assigns
    # one to SF, and marks two unavailable. A later return clears the latter.
    seeds = (
        clocks.head(1)
        .join(pl.DataFrame({"player_id": list("abcdef")}), how="cross")
        .select("game_id", "team", "player_id")
    )
    vintage = pl.DataFrame(
        {
            "player_id": list("abcefxyn"),
            "team": ["SF", "FA", "SEA", "SEA", "SEA", "SF", "SF", "SF"],
            "listed": [True, False, False, True, True, True, True, True],
            "edition_at": [
                lock - timedelta(days=n) for n in (5, 5, 5, -1, 31, 5, 5, 5)
            ],
            "season": [2024] * 8,
            "rookie": [False] * 6 + [True, False],
            "has_active_status": [False] * 8,
        }
    )
    boxes = pl.DataFrame({"season": [2024], "player_id": ["n"]})
    ev = opening_evidence(clocks, seeds, boxes, vintage)
    assert set(ev["player_id"]) == set("abcy")
    directory = vintage.with_columns(
        pl.lit("RB").alias("position"),
        c("player_id").alias("name"),
        pl.lit(False).alias("club_roster"),
    )
    revision = directory.filter(c("player_id") == "a").with_columns(
        pl.lit(True).alias("club_roster"),
        pl.lit("SEA").alias("team"),
        pl.lit(lock - timedelta(hours=2)).alias("edition_at"),
    )
    for other, time, expected in [
        ("SF", lock + timedelta(seconds=1), "SEA"),
        ("SEA", lock, "SEA"),
        ("SF", lock, "SF"),
    ]:
        conflicting = revision.with_columns(
            pl.lit(time).alias("edition_at"), pl.lit(other).alias("team")
        )
        chosen = opening_evidence(
            clocks, seeds, boxes, pl.concat([directory, revision, conflicting])
        ).filter(c("player_id") == "a")
        assert chosen["team"].to_list() == [expected]
        assert chosen["club_roster"].item() == (expected == "SEA")
    pool = (
        pl.concat(
            [
                seeds,
                clocks.join(ev.filter(c("listed")), on=["team", "season"]).select(
                    seeds.columns
                ),
            ]
        )
        .join(clocks, on=["game_id", "team"])
        .with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(k)
            for k in ("depth_at", "identity_at", "prior_at")
        )
    )
    reports = seeds.filter(c("player_id") == "c").with_columns(
        pl.lit(lock - timedelta(days=1)).alias("report_at")
    )
    moves = pool.select("player_id", c("identity_at").alias("move_at"), "team").clear()
    fixed = apply_editions(pool, ev, moves, reports)
    assert sorted(fixed.select("player_id", "team").rows()) == [
        ("a", "SF"),
        ("c", "SEA"),
        ("d", "SEA"),
        ("e", "SEA"),
        ("f", "SEA"),
        ("y", "SF"),
    ]
    assert set(fixed.filter(c("edition_confirmed"))["player_id"]) == {"a", "y"}
    # A subsequent documented trade back takes precedence over the directory.
    trade = pl.DataFrame(
        {"player_id": ["a"], "team": ["SEA"], "move_at": [lock - timedelta(days=1)]}
    )
    with_prior = pool.with_columns(pl.lit(lock - timedelta(days=150)).alias("prior_at"))
    reconciled = resolve_teams(
        apply_editions(with_prior, ev, trade, reports), reports, trade
    )[0]
    assert reconciled.filter(c("player_id") == "a")["team"].to_list() == ["SEA"]


def test_opportunity_study_replays_pinned_editions_and_paired_rb_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = pl.col
    p, raw = matchup_fixture()
    monkeypatch.setattr(opportunity_study, "matchup_sources", lambda end_year: raw)
    monkeypatch.setitem(K, "RB", 1)
    index = tmp_path / "index.csv"
    pl.DataFrame(
        {
            "season": [2013, 2014],
            "repository": ["test/source"] * 2,
            "commit": ["abc", "def"],
            "stamp": ["2013-08-30T12:00:00Z", "2014-08-30T12:00:00Z"],
            "path": ["players.json", "database.csv"],
        }
    ).write_csv(index)
    monkeypatch.setattr(opening, "SOURCES", index)
    monkeypatch.setattr(opportunity_study, "SOURCES", index)
    archive = {
        "00-0000001": {
            "position": "HB",
            "team": "KCC",
            "status": "ACT",
            "full_name": "Backup",
        },
        "00-0000002": {
            "position": "QB",
            "team": "SFO",
            "status": "RES",
            "full_name": "Reserve",
        },
        "00-0000003": {"position": "TE", "team": "SEA", "full_name": "Old"},
    }

    class Response:
        def __init__(self, url: str) -> None:
            self.content = (
                json.dumps(archive)
                if url.endswith("json")
                else "gsis_id,name,pos,team,draft_year\n00-0000001,Backup,RB,FA*,2012\n"
                "00-0000002,Duplicate,QB,SFO,2010\n00-0000002,Duplicate,QB,TBB,2010\n"
            ).encode()

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(opening.requests, "get", lambda url, timeout: Response(url))
    editions, _ = opening.opening_editions(2014)
    assert editions.select("season", "team", "position", "listed").rows() == [
        (2013, "KC", "RB", True),
        (2013, "SF", "QB", False),
        (2014, "FA*", "RB", False),
    ]
    sides = schedule_sides(p.schedules)
    plays = sides.select(
        "game_id",
        "season",
        "week",
        c("team").alias("posteam"),
        pl.lit("REG").alias("season_type"),
        pl.lit("run").alias("play_type"),
        pl.lit(1).alias("rush_attempt"),
        pl.lit(0).alias("qb_kneel"),
        c("team").alias("rusher_player_id"),
        pl.lit(None, dtype=pl.String).alias("receiver_player_id"),
        pl.lit(5).alias("yardline_100"),
        pl.lit(0).alias("play_deleted"),
        pl.lit(0).alias("two_point_attempt"),
    )
    for year in range(2009, 2015):
        p.pbp_by_season[year] = plays.filter(c("season") == year)
    out = tmp_path / "opportunity"
    roster_index, facts = tmp_path / "revisions.csv", tmp_path / "listings.csv"
    pl.DataFrame(
        {
            "season": [2014],
            "team": ["ARI"],
            "stamp": ["2014-08-31T00:00:00Z"],
            "revision": [1],
        }
    ).write_csv(roster_index)
    name = p.rosters_weekly.filter(c("position") == "RB")["full_name"][0]
    pl.DataFrame(
        {
            "revision": [1, 1],
            "name": [name, "No matching identity"],
            "section": ["active", "reserve"],
            "position": ["RB", "TE"],
        }
    ).write_csv(facts)
    for module in (club_rosters, opportunity_study):
        monkeypatch.setattr(module, "REVISIONS", roster_index)
        monkeypatch.setattr(module, "FACTS", facts)
    p.players = p.players.join(
        p.rosters_weekly.select("gsis_id", c("full_name").alias("display_name")).unique(
            "gsis_id"
        ),
        on="gsis_id",
    ).with_columns(
        pl.lit(None, dtype=pl.String).alias(k)
        for k in (
            "first_name",
            "last_name",
            "football_name",
            "common_first_name",
            "suffix",
        )
    )
    opportunity_study.run_opportunity(out, 2014, p)
    assert pl.read_csv(out / "club-revisions.csv").select("listings", "linked").row(
        0
    ) == (2, 1)
    baseline = pl.read_parquet(out / "original-baseline.parquet")
    added = pl.read_parquet(out / "original-carry_goal.parquet")
    assert_frame_equal(
        baseline.select(*PLAYER_WEEK, "team", "points"),
        added.select(*PLAYER_WEEK, "team", "points"),
    )
    assert set(pl.read_csv(out / "scores.csv")["target"]) == {
        "points",
        "carries",
        "rushing_tds",
        "rushing_yards",
    }
    compared = tmp_path / "transaction-replay"
    opportunity_study.run_opportunity(compared, 2014, p, transactions=True)
    assert_frame_equal(
        pl.read_parquet(compared / "club_only-carry.parquet"),
        pl.read_parquet(out / "dated-carry.parquet"),
    )
    assert set(pl.read_csv(compared / "affiliation-tiers.csv")["board"]) == {
        "club_only",
        "transactions",
    }


def test_club_namesakes_cannot_be_resolved_from_the_target_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = pl.col
    revisions, facts = tmp_path / "revisions.csv", tmp_path / "facts.csv"
    pl.DataFrame(
        {
            "revision": [10],
            "stamp": ["2018-09-05T00:00:00Z"],
            "season": [2018],
            "team": ["SEA"],
        }
    ).write_csv(revisions)
    pl.DataFrame(
        {
            "revision": [10],
            "position": ["RB"],
            "name": ["Same Name"],
            "section": ["active"],
        }
    ).write_csv(facts)
    monkeypatch.setattr(club_rosters, "REVISIONS", revisions)
    monkeypatch.setattr(club_rosters, "FACTS", facts)
    players = pl.DataFrame(
        {"gsis_id": ["a", "b"], "display_name": ["Same Name"] * 2}
    ).with_columns(
        pl.lit(None, dtype=pl.String).alias(k)
        for k in (
            "first_name",
            "common_first_name",
            "football_name",
            "suffix",
            "last_name",
        )
    )
    current = pl.DataFrame(
        {"gsis_id": ["a"], "season": [2018], "position": ["RB"], "game_type": ["REG"]}
    )
    draft = current.drop("game_type").clear()
    unresolved, meta = club_rosters.club_editions(2018, players, current, draft)
    assert unresolved.is_empty() and meta["linked"].item() == 0
    prior = current.with_columns(
        pl.lit("b").alias("gsis_id"), (c("season") - 1).alias("season")
    )
    evidence, _ = club_rosters.club_editions(
        2018, players, pl.concat([current, prior]), draft
    )
    assert evidence["player_id"].to_list() == ["b"]


def test_rb_opportunities_use_prior_full_offense_and_real_zeros() -> None:
    c = pl.col
    plays = pl.DataFrame(
        {
            "season_type": ["REG"] * 10 + ["POST"],
            "game_id": ["old"] * 11,
            "season": [2023] * 11,
            "week": [18] * 11,
            "posteam": ["SEA"] * 11,
            "play_type": [
                "run",
                "run",
                "run",
                "run",
                "no_play",
                "run",
                "run",
                "pass",
                "pass",
                "pass",
                "run",
            ],
            "rush_attempt": [1] * 7 + [0, 0, 0, 1],
            "qb_kneel": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "two_point_attempt": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "play_deleted": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "rusher_player_id": [
                "back",
                "qb",
                None,
                "qb",
                "back",
                "back",
                "back",
                None,
                None,
                None,
                "back",
            ],
            "receiver_player_id": [None] * 7 + ["wr", "back", None, None],
            "yardline_100": [5, 6, 4, 1, 3, 4, 1, 1, 6, 5, 3],
        }
    )
    other = plays.head(1).with_columns(
        pl.lit("SF").alias("posteam"),
        pl.lit("sf_old").alias("game_id"),
        pl.lit(20).alias("yardline_100"),
        pl.lit("other").alias("rusher_player_id"),
    )
    future = other.with_columns(
        pl.lit("new").alias("game_id"),
        pl.lit(2024).alias("season"),
        pl.lit(1).alias("week"),
    )
    team, player = count_opportunities(
        pl.concat([plays, other, future], how="vertical_relaxed")
    )
    assert team.filter(c("team") == "SEA").select(
        "runs", "inside5_runs", "inside5_targets"
    ).rows() == [(3, 2, 1)]
    assert team.filter(c("team") == "SF").select(
        "inside5_runs", "inside5_targets"
    ).rows() == [(0, 0), (0, 0)]
    snaps = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [18, 18],
            "team": ["SEA", "SEA"],
            "game_type": ["REG", "REG"],
            "pfr_player_id": ["zero", "dnp"],
            "offense_snaps": [0, 0],
            "defense_snaps": [0, 0],
            "st_snaps": [1, 0],
        }
    )
    ids = (
        snaps.select("pfr_player_id")
        .rename({"pfr_player_id": "pfr_id"})
        .with_columns(c("pfr_id").alias("player_id"))
    )
    obs = include_snap_zeros(player, team, snaps, ids)
    board = pl.DataFrame(
        {
            "game_id": ["new"] * 4,
            "season": [2024] * 4,
            "week": [1] * 4,
            "position": ["RB"] * 4,
            "team": ["SF"] * 4,
            "player_id": ["back", "zero", "dnp", "other"],
            "ruled_out": [False, False, False, True],
        }
    )
    result = rb_opportunities(board, team, obs)
    back = result.filter(c("player_id") == "back")
    assert back.select(
        "share_runs", "team_runs", "expected_runs", "backfield_competition"
    ).row(0) == pytest.approx((1 / 3, 1, 1 / 3, 0))
    assert back.select("share_inside5_runs", "expected_inside5_runs").row(
        0
    ) == pytest.approx((2 / 5, 0))
    assert result.filter(c("player_id") == "zero")["share_runs"].item() == 0
    assert result.filter(c("player_id") == "dnp")["share_runs"].item() is None
    # Neither today's player allocation nor today's team totals can change X.
    perturb = pl.when(c("season") == 2024).then(123).otherwise(c("runs")).alias("runs")
    assert_frame_equal(
        result.select(*CARRY, *GOAL),
        rb_opportunities(
            board, team.with_columns(perturb), obs.with_columns(perturb)
        ).select(*CARRY, *GOAL),
    )
