"""Matched completion/catch-stage experiment on the strict kickoff board."""

import polars as pl

from ffpred.errors import DataAcquisitionError
from ffpred.study.matchup_models import CATCH_CHANNELS, NAMES
from ffpred.study.methods import method_scores, paired_mae
from ffpred.study.panel import CHANNELS, TEAM_CODES
from ffpred.study.role_evidence import ENTRY

c = pl.col
PAIRS = {
    "catch_linear": "factor_linear",
    "catch_forest": "factor_forest",
    "catch_ensemble": "ensemble",
}


def catch_labels(board: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    labels = (
        stats.filter(c("season_type") == "REG")
        .select("game_id", c("team").replace(TEAM_CODES), "player_id", "completions")
        .drop_nulls(ENTRY)
    )
    rows = board.join(labels, on=ENTRY, how="left", validate="1:1").with_columns(
        pl.when(c("outcome_status") == "same_box")
        .then(c("completions"))
        .when(c("points").is_not_null())
        .then(0)
        .otherwise(None)
        .alias("completions")
    )
    for ch, made in CATCH_CHANNELS.items():
        if rows.filter(
            c("points").is_not_null()
            & ~c(made).is_between(0, c(CHANNELS[ch])).fill_null(False)
        ).height:
            raise DataAcquisitionError(f"Invalid {made} label in a verified matchup")
    return rows


def catch_tables(scored: pl.DataFrame) -> dict[str, pl.DataFrame]:
    annual, pooled = method_scores(scored, [*NAMES, *PAIRS])
    paired = pl.concat(
        paired_mae(annual.filter(c("model").is_in([new, old])), reference=old).filter(
            c("model") == new
        )
        for new, old in PAIRS.items()
    )
    rows = []
    for method in ("linear", "forest"):
        prefix = f"catch_{method}_"
        for ch, made in CATCH_CHANNELS.items():
            unit = CHANNELS[ch]
            n, m = c(unit), c(made)
            r = c(prefix + made + "_rate")
            for stat in (f"{ch}_yards", f"{ch}_tds"):
                y = c(stat)
                per = c(f"{prefix}{stat}_per_catch")
                error = c(prefix + stat) - y
                volume = (c(f"{method}_{unit}") - n) * r * per
                conversion = (n * r - m) * per
                # Signed contributions add to raw stat MAE; the last term
                # includes rare receiving production without a credited catch.
                rows.append(
                    scored.group_by("position").agg(
                        pl.lit(method).alias("method"),
                        pl.lit(ch).alias("channel"),
                        pl.lit(stat).alias("stat"),
                        pl.len().alias("games"),
                        (c(f"{method}_{stat}") - y).abs().mean().alias("old_stat_mae"),
                        error.abs().mean().alias("stat_mae"),
                        (volume * error.sign()).mean().alias("from_opportunities"),
                        (conversion * error.sign()).mean().alias("from_catch_rate"),
                        ((error - volume - conversion) * error.sign())
                        .mean()
                        .alias("from_yield"),
                        (c(prefix + made) - m).abs().mean().alias("catch_count_mae"),
                        (per - y / m)
                        .abs()
                        .filter(m > 0)
                        .mean()
                        .alias("per_catch_mae_when_used"),
                        (m > 0).sum().alias("games_with_catch"),
                        ((m == 0) & (y != 0))
                        .sum()
                        .alias("zero_catch_production_games"),
                    )
                )
    return {
        "scores.csv": pooled,
        "annual.csv": annual,
        "paired.csv": paired.sort("position", "model"),
        "stages.csv": pl.concat(rows).sort("position", "method", "stat"),
    }
