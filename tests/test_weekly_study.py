import gzip
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.fakes import FakeProvider
from ffpred.study.benchmark import BLOCKS, backtest
from ffpred.study.bridge import read_gamecenter
from ffpred.study.defense import ADJUSTED, defense_snapshots
from ffpred.study.methods import CHALLENGERS, KEY, REFERENCE, compare_methods
from ffpred.study.panel import (
    PLAYER_FEATURES,
    PLAYER_WEEK,
    POSITIONS,
    SCORING,
    box_scores,
    candidates,
    player_panel,
    schedule_sides,
)
from ffpred.study.role_evidence import (
    depth_evidence,
    game_clocks,
    official_news,
    read_news,
)
from ffpred.study.roles import run_roles
from ffpred.study.run import run_study


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
