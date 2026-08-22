"""Streamlit application for exploring fantasy-football predictions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Never, TypeVar

import altair as alt
import polars as pl
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.web import cli as streamlit_cli

from ffpred.dashboard.data import (
    ACTUAL_COLUMN,
    CONSENSUS_MODEL,
    DashboardDataError,
    draft_board,
    load_prediction_files,
    model_choices,
    model_scorecard,
    player_history,
    select_model,
    weekly_board,
)

CANVAS = "#0d1517"
SURFACE = "#162326"
TEXT = "#dce8e4"
GREEN = "#77d584"
CYAN = "#62b9c8"
AMBER = "#d4af63"
CORAL = "#d98578"
VIOLET = "#9b91c9"
GRID = "#294044"
PROJECTED_BLUE = "#56b4e9"
ACTUAL_NEUTRAL = "#e6efec"
PROJECTED_PACE_PURPLE = "#cc79a7"
ACTUAL_PACE_YELLOW = "#f0e442"

SEASON_FILTER_KEY = "forecast-target-season"
POSITION_FILTER_KEY = "forecast-positions"
MODEL_FILTER_KEY = "forecast-model"
FILTER_STATE_KEY = "_forecast-filter-state"
FILTER_QUERY_KEYS = ("season", "position", "model")

DRAFT_BAR_FIELDS = {
    "Projected": "projected_points",
    "Actual": "actual_points",
    "Adjusted at projected PPG": "projected_pace_adjusted_actual",
    "Adjusted at actual PPG": "actual_pace_adjusted_actual",
}
DRAFT_BAR_COLORS = {
    "Projected": PROJECTED_BLUE,
    "Actual": ACTUAL_NEUTRAL,
    "Adjusted at projected PPG": PROJECTED_PACE_PURPLE,
    "Adjusted at actual PPG": ACTUAL_PACE_YELLOW,
}

T = TypeVar("T")


def _inject_styles() -> None:
    st.html(
        """
        <!--
        THESIS: A night-game forecast command center that makes model provenance
        as visible as player rank, rejecting both spreadsheet beige and neon sci-fi.
        OWN-WORLD: Soft charcoal field layers, field-green signal, cyan comparison
        data, amber actuals, low-contrast grid lines, and broadcast-grade typography.
        STORY: Confirm the forecast state, select a decision horizon, filter the
        field, and compare projected performance without losing model context.
        FIRST VIEWPORT: A compact forecast header and live telemetry rail lead
        directly into three persistent workspaces and the active ranking surface.
        FORM: Stadium operations display; dense horizontal telemetry staging;
        restrained color strategy optimized for long evening sessions.
        -->
        <style>
        :root {
            --canvas: #0d1517;
            --panel: #121e21;
            --surface: #162326;
            --surface-high: #1b2b2f;
            --line: #294044;
            --line-strong: #3c5b60;
            --text: #dce8e4;
            --muted: #94aaa5;
            --green: #77d584;
            --green-deep: #214c34;
            --cyan: #62b9c8;
            --amber: #d4af63;
            --coral: #d98578;
        }
        .stApp {
            background:
                linear-gradient(rgba(119, 213, 132, .022) 1px, transparent 1px),
                linear-gradient(90deg, rgba(119, 213, 132, .016) 1px, transparent 1px),
                var(--canvas);
            background-size: 40px 40px;
            color: var(--text);
            overflow-x: hidden;
        }
        [data-testid="stHeader"] {
            background: rgba(13, 21, 23, .92);
            border-bottom: 1px solid rgba(60, 91, 96, .45);
        }
        .block-container {
            box-sizing: border-box;
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4.5rem;
            width: 100%;
        }
        h1, h2, h3 {
            color: var(--text) !important;
            font-family: "Bahnschrift", "Arial Narrow", "Segoe UI",
                sans-serif !important;
        }
        h1 {
            font-size: clamp(2.4rem, 4vw, 4.1rem) !important;
            font-weight: 650 !important;
            letter-spacing: -.035em !important;
            line-height: .94;
        }
        h2 {
            font-size: clamp(1.65rem, 2.6vw, 2.3rem) !important;
            font-weight: 620 !important;
            letter-spacing: -.025em !important;
            margin: 2.6rem 0 .55rem !important;
        }
        h3 { font-weight: 600 !important; }
        p, label, button, input, textarea, [data-testid="stMarkdownContainer"] {
            font-family: "Segoe UI", Arial, sans-serif;
        }
        p, [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
            line-height: 1.58;
        }
        [data-testid="stMain"] label { color: var(--text) !important; }
        .command-header {
            align-items: end;
            background: linear-gradient(112deg, var(--surface-high), var(--panel));
            border: 1px solid var(--line);
            border-radius: 10px 10px 0 0;
            display: grid;
            gap: 2rem;
            grid-template-columns: minmax(0, 1fr) auto;
            overflow: hidden;
            padding: clamp(1.4rem, 3vw, 2.5rem);
            position: relative;
        }
        .command-header::before {
            background: var(--green);
            bottom: 0;
            content: "";
            height: 3px;
            left: 0;
            position: absolute;
            width: 18%;
            animation: field-lock 700ms cubic-bezier(.2, .8, .2, 1) both;
        }
        @keyframes field-lock {
            from { width: 4%; filter: brightness(.7); }
            to { width: 18%; filter: brightness(1); }
        }
        @media (prefers-reduced-motion: reduce) {
            .command-header::before { animation: none; }
        }
        .command-header h1 { margin: 0; }
        .command-header > div { min-width: 0; }
        .command-header p {
            color: var(--muted);
            font-size: 1rem;
            margin: .7rem 0 0;
            max-width: 66ch;
        }
        .forecast-state {
            align-self: center;
            background: rgba(119, 213, 132, .075);
            border: 1px solid rgba(119, 213, 132, .38);
            border-radius: 8px;
            display: grid;
            grid-template-columns: auto 1fr;
            min-width: 12.5rem;
            padding: .75rem .9rem;
        }
        .status-dot {
            background: var(--green);
            border-radius: 50%;
            grid-row: 1 / 3;
            height: .55rem;
            margin: .45rem .7rem 0 0;
            width: .55rem;
        }
        .forecast-state span {
            color: var(--muted);
            font-size: .68rem;
            font-weight: 650;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .forecast-state strong {
            color: var(--green);
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: .93rem;
            font-weight: 650;
            margin-top: .12rem;
        }
        .telemetry-rail {
            background: var(--surface);
            border: 1px solid var(--line);
            border-top: 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
            margin-bottom: 1.35rem;
        }
        .st-key-mission-controls {
            background: var(--panel);
            border-left: 1px solid var(--line);
            border-right: 1px solid var(--line);
            padding: .78rem 1rem .9rem;
        }
        .st-key-mission-controls label {
            color: var(--muted) !important;
            font-size: .68rem !important;
            font-weight: 650 !important;
            letter-spacing: .095em !important;
            text-transform: uppercase;
        }
        .st-key-mission-controls [data-baseweb="select"] > div {
            background: var(--surface-high);
        }
        .telemetry-item {
            border-right: 1px solid var(--line);
            min-width: 0;
            padding: .8rem 1rem .9rem;
        }
        .telemetry-item:last-child { border-right: 0; }
        .telemetry-item span {
            color: var(--muted);
            display: block;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: .65rem;
            font-weight: 650;
            letter-spacing: .095em;
            margin-bottom: .28rem;
            text-transform: uppercase;
        }
        .telemetry-item strong {
            color: var(--text);
            display: block;
            font-family: "Bahnschrift", "Segoe UI", sans-serif;
            font-size: .93rem;
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        [data-testid="stToolbar"] .rc-overflow:has(
            [data-testid="stTopNavLink"]
        ) {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 9px;
            box-sizing: border-box;
            gap: .35rem;
            max-width: 36rem;
            padding: .3rem;
            width: min(36rem, calc(100vw - 4rem));
        }
        [data-testid="stToolbar"] .rc-overflow-item:has(
            [data-testid="stTopNavLink"]
        ) {
            flex: 1 1 0;
            min-width: 0;
        }
        [data-testid="stTopNavLinkContainer"] {
            min-width: 0;
            width: 100%;
        }
        [data-testid="stTopNavLink"] {
            border: 1px solid transparent;
            border-radius: 6px;
            color: var(--muted) !important;
            font-family: "Segoe UI", Arial, sans-serif !important;
            font-size: .86rem;
            font-weight: 600;
            justify-content: center;
            min-width: 0;
            padding: .2rem .7rem !important;
            width: 100%;
        }
        [data-testid="stTopNavLink"] p {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        [data-testid="stTopNavLink"][aria-current="page"] {
            background: var(--green-deep) !important;
            border-color: rgba(119, 213, 132, .28);
            color: #e5f3e7 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, .18);
            overflow: hidden;
        }
        .stButton > button, .stDownloadButton > button {
            border: 1px solid rgba(119, 213, 132, .48);
            border-radius: 7px;
            background: var(--green-deep);
            color: #e5f3e7;
            box-shadow: 0 8px 20px rgba(0, 0, 0, .16);
            font-weight: 650;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #2a6140;
            border-color: var(--green);
            color: #eef8ef;
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, .22);
        }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
            outline: 2px solid var(--cyan);
            outline-offset: 2px;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--line-strong);
            background: var(--surface);
        }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div {
            background: var(--surface);
            border-color: var(--line-strong);
            border-radius: 7px;
            color: var(--text);
        }
        [data-testid="stExpander"] {
            background: rgba(22, 35, 38, .68);
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: var(--green);
            border-color: var(--green);
        }
        [data-testid="stToggle"] [data-checked="true"] {
            background: var(--green-deep);
        }
        hr { border-color: var(--line) !important; }
        code {
            background: var(--surface-high) !important;
            color: var(--cyan) !important;
        }
        @media (max-width: 760px) {
            .block-container { padding: 3.35rem .8rem 3rem; }
            .command-header {
                align-items: start;
                grid-template-columns: 1fr;
                gap: 1.2rem;
                padding: 1.35rem;
            }
            .command-header h1 {
                font-size: 2.15rem !important;
                max-width: 100%;
                overflow-wrap: normal;
                word-break: normal;
            }
            .command-header p { overflow-wrap: anywhere; }
            .forecast-state { min-width: 0; width: fit-content; }
            .st-key-mission-controls { padding: .7rem .8rem .8rem; }
            .st-key-mission-controls [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            .st-key-mission-controls [data-testid="stColumn"] {
                flex: 1 1 13rem !important;
                min-width: min(13rem, 100%);
            }
            .telemetry-rail {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .telemetry-item {
                padding: .72rem .8rem;
            }
            [data-testid="stTopNavLink"] {
                font-size: .74rem;
                padding-inline: .25rem !important;
            }
        }
        </style>
        """
    )


def _chart_theme(
    chart: alt.Chart | alt.LayerChart,
) -> alt.Chart | alt.LayerChart:
    themed = (
        chart.configure(background=SURFACE)
        .configure_view(fill=SURFACE, stroke=None)
        .configure_axis(
            domainColor=GRID,
            gridColor=GRID,
            labelColor=TEXT,
            labelFont="Segoe UI",
            tickColor=GRID,
            titleColor=TEXT,
            titleFont="Segoe UI",
        )
        .configure_legend(
            labelColor=TEXT,
            labelFont="Segoe UI",
            titleColor=TEXT,
            titleFont="Segoe UI",
        )
    )
    assert isinstance(themed, (alt.Chart, alt.LayerChart))
    return themed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--predictions", action="append", type=Path, default=[])
    arguments, _ = parser.parse_known_args()
    return arguments


def _discover_prediction_files() -> list[Path]:
    forecast_candidates = list(Path.cwd().glob("artifacts/*/*-predictions.parquet"))
    candidates = forecast_candidates or [
        *Path.cwd().glob("*-predictions.parquet"),
        *Path.cwd().glob("artifacts/*-predictions.parquet"),
    ]
    return sorted({path.resolve() for path in candidates})


@st.cache_data(show_spinner=False)
def _load_paths(paths: tuple[str, ...]) -> pl.DataFrame:
    return load_prediction_files([Path(path) for path in paths])


def _load_data() -> pl.DataFrame:
    arguments = _arguments()
    paths = arguments.predictions or _discover_prediction_files()

    if paths:
        return _load_paths(tuple(str(path) for path in paths))

    st.warning(
        "No prediction sheets were found. Build the forecast archive or start "
        "the dashboard with `--predictions PATH`."
    )
    st.code(
        "uv run ffpred-dashboard --predictions artifacts/svr-predictions.parquet",
        language="powershell",
    )
    st.stop()


def _matching_option(
    value: str | None,
    options: Sequence[T],
    fallback: T,
) -> T:
    if value is None:
        return fallback
    normalized = value.casefold()
    return next(
        (option for option in options if str(option).casefold() == normalized),
        fallback,
    )


def _query_positions(positions: Sequence[str]) -> list[str]:
    raw = st.query_params.get_all("position")
    requested = {
        position.strip().casefold()
        for value in raw
        for position in value.split(",")
        if position.strip()
    }
    return [position for position in positions if position.casefold() in requested]


def _write_filter_query() -> None:
    season = st.session_state[SEASON_FILTER_KEY]
    chosen_positions = st.session_state[POSITION_FILTER_KEY]
    model = st.session_state[MODEL_FILTER_KEY]
    st.session_state[FILTER_STATE_KEY] = {
        "season": season,
        "positions": chosen_positions,
        "model": model,
    }
    st.query_params["season"] = str(season)
    st.query_params["position"] = chosen_positions or [""]
    st.query_params["model"] = model


def _hydrate_filter_state(
    seasons: Sequence[int],
    positions: Sequence[str],
    models: Sequence[str],
) -> None:
    query_present = any(key in st.query_params for key in FILTER_QUERY_KEYS)
    if query_present:
        season = _matching_option(st.query_params.get("season"), seasons, seasons[0])
        chosen_positions = (
            _query_positions(positions)
            if "position" in st.query_params
            else list(positions)
        )
        model = _matching_option(st.query_params.get("model"), models, models[0])
    else:
        saved = st.session_state.get(FILTER_STATE_KEY, {})
        season = saved.get("season", seasons[0])
        if season not in seasons:
            season = seasons[0]
        saved_positions = saved.get("positions", positions)
        chosen_positions = [
            position for position in positions if position in saved_positions
        ]
        model = saved.get("model", models[0])
        if model not in models:
            model = models[0]

    st.session_state[SEASON_FILTER_KEY] = season
    st.session_state[POSITION_FILTER_KEY] = chosen_positions
    st.session_state[MODEL_FILTER_KEY] = model
    st.session_state[FILTER_STATE_KEY] = {
        "season": season,
        "positions": chosen_positions,
        "model": model,
    }


def _global_filters(
    frame: pl.DataFrame,
    *,
    header: st.delta_generator.DeltaGenerator,
) -> tuple[pl.DataFrame, int, list[str], str]:
    seasons = sorted(frame["target_season"].unique().to_list(), reverse=True)
    positions = sorted(frame["position"].unique().to_list())
    models = model_choices(frame)
    _hydrate_filter_state(seasons, positions, models)

    with st.container(key="mission-controls"):
        season_control, position_control, model_control = st.columns([1, 2, 1])
        season = int(
            season_control.selectbox(
                "Target season",
                seasons,
                key=SEASON_FILTER_KEY,
                on_change=_write_filter_query,
                help="Choose an upcoming forecast or a frozen historical replay.",
            )
        )
        chosen_positions = [
            str(position)
            for position in position_control.multiselect(
                "Position",
                positions,
                key=POSITION_FILTER_KEY,
                on_change=_write_filter_query,
                help="Filter every workspace to one or more fantasy positions.",
            )
        ]
        model = model_control.selectbox(
            "Model view",
            models,
            key=MODEL_FILTER_KEY,
            on_change=_write_filter_query,
            help="Consensus averages matching SVR and MLP predictions.",
        )

    _write_filter_query()
    selected = select_model(
        frame.filter(pl.col("target_season") == season),
        model,
    )
    _command_header(header, selected)
    return selected, season, chosen_positions, model


def _command_header(
    header: DeltaGenerator,
    frame: pl.DataFrame,
) -> None:
    actual_rows = frame[ACTUAL_COLUMN].count()
    is_frozen_forecast = "history_through_season" in frame.columns
    if is_frozen_forecast and actual_rows:
        artifact_mode = "Point-in-time replay"
    elif is_frozen_forecast:
        artifact_mode = "Upcoming forecast"
    else:
        artifact_mode = "Rolling backtest"
    header.html(
        f"""
        <section class="command-header">
          <div>
            <h1>Fantasy Forecast Center</h1>
            <p>Season-long draft value and weekly matchup signals, kept in
            separate decision lanes and tied to the same model evidence.</p>
          </div>
          <div class="forecast-state">
            <span class="status-dot" aria-hidden="true"></span>
            <span>Forecast state</span>
            <strong>{artifact_mode}</strong>
          </div>
        </section>
        """
    )


def _masthead(frame: pl.DataFrame, model: str) -> None:
    actual_rows = frame[ACTUAL_COLUMN].count()
    is_frozen_forecast = "history_through_season" in frame.columns
    provenance_items = ""
    if is_frozen_forecast:
        history_through = int(frame["history_through_season"][0])
        forecast_as_of = escape(str(frame["forecast_as_of"][0]))
        provenance_items = (
            "<div class='telemetry-item'>"
            f"<span>History through</span><strong>{history_through}</strong></div>"
            "<div class='telemetry-item'>"
            f"<span>Forecast lock</span><strong>{forecast_as_of}</strong></div>"
        )
    st.html(
        f"""
        <div class="telemetry-rail">
          <div class="telemetry-item">
            <span>Matchup rows</span><strong>{frame.height:,}</strong>
          </div>
          {provenance_items}
        </div>
        """
    )
    if is_frozen_forecast and actual_rows:
        st.caption(
            "This historical forecast is frozen before the target season. "
            "Actual results are shown only for comparison."
        )
    elif is_frozen_forecast:
        st.caption(
            "This upcoming forecast uses only completed seasons through the "
            "history cutoff shown above."
        )
    elif actual_rows:
        st.caption(
            "This rolling backtest updates player history during the selected "
            "season; it is not a preseason projection."
        )
    if model == CONSENSUS_MODEL:
        st.caption(
            "Consensus is the per-game average of the loaded SVR and MLP "
            "predictions. Choose either model in the controls above to view it alone."
        )


def _draft_view(
    frame: pl.DataFrame,
    *,
    season: int,
    positions: list[str],
    model: str,
) -> None:
    st.header("Draft board")
    st.write(
        "Projected and actual totals sit together. Two availability-adjusted "
        "scores fill each source-backed injury absence at either the preseason "
        "projected pace or the player's actual scoring pace. Opportunity estimates "
        "use the current depth chart, recent player shares, and the offense's prior "
        "season volume."
    )
    if not positions:
        st.warning("Choose at least one position to build the draft board.")
        return

    board = draft_board(
        frame,
        season=season,
        positions=positions,
    )
    has_actuals = board["actual_points"].count() > 0
    controls = st.columns([2, 1, 1.6] if has_actuals else [2, 1])
    search = controls[0].text_input("Find player", placeholder="Search the board")
    top_n = controls[1].selectbox("Board depth", [12, 24, 50, 100], index=1)
    sort_label = (
        controls[2].selectbox("Sort players by", list(DRAFT_BAR_FIELDS))
        if has_actuals
        else "Projected"
    )
    if search:
        board = board.filter(
            pl.col("player_name").str.contains(search, literal=True, strict=False)
        )
    display = board.sort(
        DRAFT_BAR_FIELDS[sort_label],
        descending=True,
        nulls_last=True,
    ).head(top_n)

    if display.is_empty():
        st.warning("No players match this search.")
        return

    chart_rows = display.head(18)
    player_order = chart_rows["player_name"].to_list()
    identity = [
        "player_name",
        "injury_games",
        "projected_target_share",
        "projected_carry_share",
        "team_previous_season_offensive_plays",
        "team_previous_season_pass_rate",
        "opportunity_risk",
    ]
    regular_bars = pl.concat(
        [
            chart_rows.select(
                *identity,
                pl.lit(label).alias("measure"),
                pl.col(field).alias("base_points"),
                pl.col(field).alias("total_points"),
                pl.lit(0.0).alias("addition_points"),
            )
            for label, field in list(DRAFT_BAR_FIELDS.items())[:2]
        ],
        how="vertical",
    )
    adjusted_bars = pl.concat(
        [
            chart_rows.select(
                *identity,
                pl.lit(label).alias("measure"),
                pl.col("actual_points").alias("base_points"),
                pl.col(total_field).alias("total_points"),
                pl.col(addition_field).alias("addition_points"),
            )
            for label, total_field, addition_field in (
                (
                    "Adjusted at projected PPG",
                    "projected_pace_adjusted_actual",
                    "projected_pace_injury_points",
                ),
                (
                    "Adjusted at actual PPG",
                    "actual_pace_adjusted_actual",
                    "actual_pace_injury_points",
                ),
            )
        ],
        how="vertical",
    ).filter(pl.col("injury_games") > 0)
    chart_data = pl.concat([regular_bars, adjusted_bars], how="vertical").drop_nulls(
        ["base_points", "total_points"]
    )
    chart_frame = chart_data.to_pandas()
    available_measures = set(chart_data["measure"].to_list())
    bar_order = [label for label in DRAFT_BAR_FIELDS if label in available_measures]
    color = alt.Color(
        "measure:N",
        scale=alt.Scale(
            domain=bar_order,
            range=[DRAFT_BAR_COLORS[label] for label in bar_order],
        ),
        sort=bar_order,
        legend=alt.Legend(
            columns=2,
            direction="horizontal",
            labelLimit=240,
            orient="bottom",
            symbolType="square",
            title=None,
        ),
    )
    y = alt.Y(
        "player_name:N",
        sort=player_order,
        title=None,
        scale=alt.Scale(paddingInner=0.34, paddingOuter=0.16),
    )
    y_offset = alt.YOffset(
        "measure:N",
        sort=bar_order,
        scale=alt.Scale(paddingInner=0.12),
        title=None,
    )
    tooltip = [
        alt.Tooltip("player_name:N", title="Player"),
        alt.Tooltip("measure:N", title="Measure"),
        alt.Tooltip("total_points:Q", title="Total", format=".1f"),
        alt.Tooltip("addition_points:Q", title="Injury addition", format=".1f"),
        alt.Tooltip("injury_games:Q", title="Injury misses"),
        alt.Tooltip(
            "projected_target_share:Q",
            title="Target share",
            format=".1%",
        ),
        alt.Tooltip(
            "projected_carry_share:Q",
            title="Carry share",
            format=".1%",
        ),
        alt.Tooltip(
            "team_previous_season_offensive_plays:Q",
            title="Offensive plays/game",
            format=".1f",
        ),
        alt.Tooltip(
            "team_previous_season_pass_rate:Q",
            title="Pass rate",
            format=".1%",
        ),
        alt.Tooltip("opportunity_risk:N", title="Usage watch"),
    ]
    base_bars = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("base_points:Q", title="Season points"),
            y=y,
            yOffset=y_offset,
            color=color,
            tooltip=tooltip,
        )
    )
    additions = (
        alt.Chart(chart_frame)
        .transform_filter(alt.datum.addition_points > 0)
        .mark_bar(
            cornerRadiusEnd=3,
            fillOpacity=0.2,
            strokeWidth=1.5,
            strokeDash=[6, 3],
        )
        .encode(
            x=alt.X("base_points:Q"),
            x2=alt.X2("total_points:Q"),
            y=y,
            yOffset=y_offset,
            color=color,
            stroke=color,
            tooltip=tooltip,
        )
    )
    st.altair_chart(
        _chart_theme(
            (base_bars + additions).properties(height=max(420, len(player_order) * 58))
        ),
        width="stretch",
    )
    st.caption(
        "Solid bars show recorded or modeled points. On adjusted bars, the "
        "translucent dashed extension estimates points for injury misses. "
        "Adjusted bars appear only for players with source-backed Out or "
        "reserve-list absences."
    )

    table = display.select(
        pl.col("position_rank").alias("Rank"),
        pl.col("position").alias("Pos"),
        pl.col("player_name").alias("Player"),
        pl.col("team").alias("Team"),
        pl.col("projected_points").alias("Projected"),
        (pl.col("projected_target_share") * 100.0).alias("Target share %"),
        (pl.col("projected_carry_share") * 100.0).alias("Carry share %"),
        pl.col("team_previous_season_offensive_plays").alias("Off. plays/game"),
        (pl.col("team_previous_season_pass_rate") * 100.0).alias("Pass rate %"),
        pl.col("opportunity_risk").alias("Usage watch"),
        pl.col("opportunity_basis").alias("Usage basis"),
        pl.col("actual_points").alias("Actual"),
        pl.col("injury_games").alias("Inj. missed"),
        pl.col("projected_pace_adjusted_actual").alias("Adj. at proj. PPG"),
        pl.col("projected_pace_adjusted_delta_percent").alias("Proj. pace gap %"),
        pl.col("actual_pace_adjusted_actual").alias("Adj. at actual PPG"),
        pl.col("actual_pace_adjusted_delta_percent").alias("Actual pace gap %"),
        pl.col("points_per_game").alias("Projected PPG"),
        pl.col("actual_points_per_game").alias("Actual PPG"),
        pl.col("actual_games").alias("Played"),
        pl.col("projected_games").alias("Scheduled"),
        pl.col("volatility").alias("Weekly swing"),
        pl.col("model_spread").alias("Model spread"),
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Projected": st.column_config.NumberColumn(format="%.1f"),
            "Target share %": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Estimated share of team targets. Current depth-chart roles "
                "reallocate opportunity when teammates arrive or leave.",
            ),
            "Carry share %": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Estimated share of team rushing attempts. RB values below 45% "
                "are marked as committee risk.",
            ),
            "Off. plays/game": st.column_config.NumberColumn(
                format="%.1f",
                help="Prior-season team pass attempts plus carries per game. Sacks "
                "and nullified plays are excluded.",
            ),
            "Pass rate %": st.column_config.NumberColumn(
                format="%.1f%%",
                help=(
                    "Prior-season pass attempts divided by pass attempts plus carries."
                ),
            ),
            "Usage watch": st.column_config.TextColumn(
                help="Flags RB committee risk, sub-15% WR/TE target share, and "
                "offenses below 58 plays per game.",
            ),
            "Usage basis": st.column_config.TextColumn(
                help="Depth-chart estimates are normalized across the current roster; "
                "older artifacts use trailing history.",
            ),
            "Actual": st.column_config.NumberColumn(format="%.1f"),
            "Inj. missed": st.column_config.NumberColumn(
                format="%d",
                help="Scheduled games missed with an Out injury report or "
                "reserve-list roster status.",
            ),
            "Adj. at proj. PPG": st.column_config.NumberColumn(
                format="%.1f",
                help="Actual points plus injury misses multiplied by preseason "
                "projected points per game.",
            ),
            "Proj. pace gap %": st.column_config.NumberColumn(
                format="%+.1f%%",
                help="Projected-pace adjusted total versus the original projection.",
            ),
            "Adj. at actual PPG": st.column_config.NumberColumn(
                format="%.1f",
                help="Actual points plus injury misses multiplied by actual points "
                "per game in games played.",
            ),
            "Actual pace gap %": st.column_config.NumberColumn(
                format="%+.1f%%",
                help="Actual-pace adjusted total versus the original projection.",
            ),
            "Projected PPG": st.column_config.NumberColumn(format="%.1f"),
            "Actual PPG": st.column_config.NumberColumn(format="%.1f"),
            "Weekly swing": st.column_config.NumberColumn(format="%.1f"),
            "Model spread": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.download_button(
        "Export draft sheet",
        table.write_csv(),
        file_name=f"{season}-{model.lower()}-draft-board.csv",
        mime="text/csv",
    )


def _weekly_view(
    frame: pl.DataFrame,
    *,
    season: int,
    positions: list[str],
) -> None:
    st.header("Weekly decisions")
    st.write(
        "Choose the week first, then compare a short list. Rank is local to the "
        "selected position; model agreement reports spread, not certainty."
    )
    if not positions:
        st.warning("Choose at least one position to open the matchup desk.")
        return

    weeks = sorted(
        frame.filter(pl.col("target_season") == season)["target_week"]
        .unique()
        .to_list()
    )
    if not weeks:
        st.warning("No weekly predictions are available for this season.")
        return
    week = st.select_slider("Week", options=weeks, value=weeks[-1])
    board = weekly_board(
        frame,
        season=season,
        week=week,
        positions=positions,
    )
    if board.is_empty():
        st.warning("No players match this weekly desk.")
        return

    names = board["player_name"].to_list()
    default_names = names[: min(3, len(names))]
    compared = st.multiselect(
        "Comparison slip",
        names,
        default=default_names,
        max_selections=5,
    )
    comparison = board.filter(pl.col("player_name").is_in(compared))
    if not comparison.is_empty():
        comparison_chart = (
            alt.Chart(comparison.to_pandas())
            .mark_bar(color=CYAN, cornerRadiusEnd=3)
            .encode(
                x=alt.X("prediction:Q", title=f"Week {week} projected points"),
                y=alt.Y("player_name:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("player_name:N", title="Player"),
                    alt.Tooltip("prediction:Q", title="Projected", format=".1f"),
                    alt.Tooltip(
                        "model_spread:Q",
                        title="Model spread",
                        format=".1f",
                    ),
                    alt.Tooltip("model_agreement:N", title="Agreement"),
                ],
            )
            .properties(height=max(160, len(compared) * 52))
        )
        st.altair_chart(
            _chart_theme(comparison_chart),
            width="stretch",
        )

        history = player_history(
            frame,
            player_ids=comparison["player_id"].to_list(),
            season=season,
        )
        history_long = history.select(
            "player_name",
            "target_week",
            "prediction",
            ACTUAL_COLUMN,
        ).unpivot(
            index=["player_name", "target_week"],
            on=["prediction", ACTUAL_COLUMN],
            variable_name="series",
            value_name="points",
        )
        trend = (
            alt.Chart(history_long.drop_nulls("points").to_pandas())
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("target_week:O", title="Week"),
                y=alt.Y("points:Q", title="Fantasy points"),
                color=alt.Color(
                    "player_name:N",
                    title="Player",
                    scale=alt.Scale(
                        range=[GREEN, CYAN, AMBER, CORAL, VIOLET],
                    ),
                ),
                strokeDash=alt.StrokeDash(
                    "series:N",
                    title="Line",
                    scale=alt.Scale(
                        domain=["prediction", ACTUAL_COLUMN],
                        range=[[1, 0], [6, 3]],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("player_name:N", title="Player"),
                    alt.Tooltip("target_week:O", title="Week"),
                    alt.Tooltip("series:N", title="Line"),
                    alt.Tooltip("points:Q", title="Points", format=".1f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(
            _chart_theme(trend),
            width="stretch",
        )

    weekly_table = board.select(
        pl.col("position_rank").alias("Rank"),
        pl.col("position").alias("Pos"),
        pl.col("player_name").alias("Player"),
        pl.col("team").alias("Team"),
        pl.col("opponent").alias("Opponent"),
        pl.col("prediction").alias("Projected"),
        pl.col("model_spread").alias("Model spread"),
        pl.col("model_agreement").alias("Agreement"),
        pl.col(ACTUAL_COLUMN).alias("Actual"),
    )
    st.dataframe(
        weekly_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Projected": st.column_config.NumberColumn(format="%.1f"),
            "Model spread": st.column_config.NumberColumn(format="%.1f"),
            "Actual": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def _model_room(frame: pl.DataFrame) -> None:
    st.header("Model room")
    st.write(
        "Audit the sheet before trusting it. Lower error is better; bias above "
        "zero means the model tends to project too high."
    )
    scorecard = model_scorecard(frame)
    if scorecard.is_empty():
        st.warning("Actual results are required to calculate model accuracy.")
        return

    display = scorecard.select(
        pl.col("model").alias("Model"),
        pl.col("samples").alias("Rows"),
        pl.col("mae").alias("MAE"),
        pl.col("rmse").alias("RMSE"),
        pl.col("bias").alias("Bias"),
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "Bias": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

    chart_frame = scorecard.to_pandas()
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(color=CORAL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("model:N", title=None, sort="y"),
            y=alt.Y("mae:Q", title="Mean absolute error"),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("mae:Q", title="MAE", format=".2f"),
                alt.Tooltip("rmse:Q", title="RMSE", format=".2f"),
                alt.Tooltip("bias:Q", title="Bias", format="+.2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(
        _chart_theme(chart),
        width="stretch",
    )


def _workspace_page(source: pl.DataFrame, workspace: str) -> None:
    header = st.empty()
    selected, season, positions, model = _global_filters(source, header=header)
    selected_season = selected.filter(pl.col("target_season") == season)
    _masthead(selected_season, model)
    if workspace == "draft":
        _draft_view(
            selected,
            season=season,
            positions=positions,
            model=model,
        )
    elif workspace == "weekly":
        _weekly_view(selected, season=season, positions=positions)
    else:
        _model_room(source.filter(pl.col("target_season") == season))


def render() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Fantasy Forecast Center",
        page_icon="F",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    try:
        source = _load_data()
    except (DashboardDataError, OSError, pl.exceptions.PolarsError) as error:
        st.error(f"Could not open the prediction sheet: {error}")
        st.stop()

    draft_page = st.Page(
        lambda: _workspace_page(source, "draft"),
        title="Draft Board",
        url_path="draft",
    )
    weekly_page = st.Page(
        lambda: _workspace_page(source, "weekly"),
        title="Weekly Decisions",
        url_path="weekly",
    )
    model_page = st.Page(
        lambda: _workspace_page(source, "model"),
        title="Model Room",
        url_path="model",
    )

    def open_draft_board() -> None:
        st.switch_page(draft_page)

    landing_page = st.Page(
        open_draft_board,
        title="Fantasy Forecast Center",
        default=True,
        visibility="hidden",
    )
    page = st.navigation(
        [landing_page, draft_page, weekly_page, model_page],
        position="top",
    )
    page.run()


def run() -> Never:
    """Launch the dashboard through the console script."""
    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path), "--", *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    render()
