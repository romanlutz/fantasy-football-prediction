import gzip
import io
import json
import tarfile
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ffpred.errors import ConfigurationError, DataAcquisitionError
from ffpred.providers.fakes import FakeProvider
from ffpred.study.benchmark import BLOCKS, backtest
from ffpred.study.bridge import read_gamecenter
from ffpred.study.defense import ADJUSTED, defense_snapshots
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
