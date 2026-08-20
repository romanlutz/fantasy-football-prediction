"""Streamlit application for exploring fantasy-football predictions."""

from __future__ import annotations

import argparse
import sys
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Never

import altair as alt
import polars as pl
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
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
    prepare_predictions,
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
        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            border-bottom: 1px solid var(--line);
            color: var(--text) !important;
            font-size: .82rem !important;
            letter-spacing: .11em !important;
            margin: 1.45rem 0 .85rem !important;
            padding-bottom: .65rem;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: var(--surface-high);
            border-color: var(--line-strong);
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
        [data-testid="stRadio"] > div {
            box-sizing: border-box;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .35rem;
            max-width: 36rem;
            padding: .3rem;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 9px;
            width: 100%;
        }
        [data-testid="stRadio"] label > div:first-child { display: none; }
        [data-testid="stRadio"] label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            justify-content: center;
            padding: .58rem 1rem;
            font-weight: 600;
            width: 100%;
        }
        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--green-deep);
            border-color: rgba(119, 213, 132, .28);
            color: #e5f3e7;
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
        [data-testid="stFileUploader"] button {
            background: var(--surface-high) !important;
            border: 1px solid var(--line-strong) !important;
            border-radius: 7px !important;
            color: var(--text) !important;
            font-weight: 600;
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
            .telemetry-rail { grid-template-columns: 1fr 1fr; }
            .telemetry-item:nth-child(even) { border-right: 0; }
            .telemetry-item {
                border-bottom: 1px solid var(--line);
                padding: .72rem .8rem;
            }
            .telemetry-item:nth-last-child(-n + 2) { border-bottom: 0; }
            [data-testid="stRadio"] > div { max-width: none; }
            [data-testid="stRadio"] label {
                font-size: .74rem;
                justify-content: center;
                padding: .55rem .2rem;
                text-align: center;
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


def _load_uploaded_files(files: list[UploadedFile]) -> pl.DataFrame:
    frames = [
        prepare_predictions(
            pl.read_parquet(BytesIO(upload.getvalue())),
            model_name=Path(upload.name).stem.removesuffix("-predictions").upper(),
        )
        for upload in files
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def _load_data() -> pl.DataFrame:
    arguments = _arguments()
    paths = arguments.predictions or _discover_prediction_files()

    with st.sidebar:
        st.markdown("## Data feeds")
        uploads = st.file_uploader(
            "Add prediction Parquet files",
            type=["parquet"],
            accept_multiple_files=True,
            help="Each file needs player, season, week, and prediction columns.",
        )
        if paths:
            st.caption("Active prediction feeds")
            for path in paths:
                st.code(path.name, language=None)

    if uploads:
        return _load_uploaded_files(uploads)
    if paths:
        return _load_paths(tuple(str(path) for path in paths))

    st.warning(
        "No prediction sheets found. Upload a Parquet file in the sidebar, "
        "or start with `--predictions PATH`."
    )
    st.code(
        "uv run ffpred-dashboard --predictions artifacts/svr-predictions.parquet",
        language="powershell",
    )
    st.stop()


def _global_filters(frame: pl.DataFrame) -> tuple[pl.DataFrame, int, list[str], str]:
    seasons = sorted(frame["target_season"].unique().to_list(), reverse=True)
    positions = sorted(frame["position"].unique().to_list())
    models = model_choices(frame)

    with st.sidebar:
        st.markdown("## Mission controls")
        season = int(st.selectbox("Season", seasons))
        chosen_positions = [
            str(position)
            for position in st.multiselect(
                "Position",
                positions,
                default=positions,
                help="Filter the board to one or more standard fantasy positions.",
            )
        ]
        model = st.selectbox(
            "Model view",
            models,
            index=0,
            help="Consensus averages predictions when multiple sheets are loaded.",
        )
        st.divider()
        st.caption(
            "Standard non-PPR scoring. Predictions support decisions; they do not "
            "guarantee player outcomes."
        )

    selected = select_model(
        frame.filter(pl.col("target_season") == season),
        model,
    )
    return selected, season, chosen_positions, model


def _masthead(frame: pl.DataFrame, model: str) -> None:
    positions = escape(", ".join(sorted(frame["position"].unique().to_list())))
    target_season = int(frame["target_season"][0])
    actual_rows = frame[ACTUAL_COLUMN].count()
    is_frozen_forecast = "history_through_season" in frame.columns
    if is_frozen_forecast and actual_rows:
        artifact_mode = "Point-in-time replay"
    elif is_frozen_forecast:
        artifact_mode = "Upcoming forecast"
    else:
        artifact_mode = "Rolling backtest"
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
        <div class="telemetry-rail">
          <div class="telemetry-item">
            <span>Target season</span><strong>{target_season}</strong>
          </div>
          <div class="telemetry-item">
            <span>Model view</span><strong>{escape(model)}</strong>
          </div>
          <div class="telemetry-item">
            <span>Matchup rows</span><strong>{frame.height:,}</strong>
          </div>
          <div class="telemetry-item">
            <span>Positions</span><strong>{positions}</strong>
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
            "predictions. Choose either model sheet in the sidebar to view it alone."
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
        "projected pace or the player's actual scoring pace."
    )
    if not positions:
        st.warning("Choose at least one position to build the draft board.")
        return

    controls = st.columns([1.25, 1, 1, 1])
    search = controls[0].text_input("Find player", placeholder="Search the board")
    minimum_games = controls[1].number_input(
        "Minimum projected games",
        min_value=1,
        max_value=18,
        value=8,
    )
    top_n = controls[2].selectbox("Board depth", [12, 24, 50, 100], index=1)
    hide_drafted = controls[3].toggle("Hide drafted", value=True)

    board = draft_board(
        frame,
        season=season,
        positions=positions,
        minimum_games=minimum_games,
    )
    if search:
        board = board.filter(
            pl.col("player_name").str.contains(search, literal=True, strict=False)
        )
    drafted = st.multiselect(
        "Drafted players",
        board["player_name"].to_list(),
        help="Mark players as the room takes them. Hidden players remain selected.",
    )
    if hide_drafted and drafted:
        board = board.filter(~pl.col("player_name").is_in(drafted))
    display = board.head(top_n)

    if display.is_empty():
        st.warning("No players match these draft-board controls.")
        return

    chart_frame = display.head(18).sort("projected_points").to_pandas()
    comparison = (
        chart_frame[["player_name", "projected_points", "actual_points"]]
        .melt(
            id_vars="player_name",
            value_vars=["projected_points", "actual_points"],
            var_name="measure",
            value_name="points",
        )
        .dropna(subset=["points"])
    )
    comparison["measure"] = comparison["measure"].replace(
        {
            "projected_points": "Projected",
            "actual_points": "Actual",
        }
    )
    bars = (
        alt.Chart(comparison)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("points:Q", title="Season points"),
            y=alt.Y("player_name:N", sort=None, title=None),
            yOffset=alt.YOffset(
                "measure:N",
                sort=["Projected", "Actual"],
                title=None,
            ),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(
                    domain=["Projected", "Actual"],
                    range=[GREEN, AMBER],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("player_name:N", title="Player"),
                alt.Tooltip("measure:N", title="Measure"),
                alt.Tooltip("points:Q", title="Points", format=".1f"),
            ],
        )
    )
    projected_pace_adjusted = (
        alt.Chart(chart_frame.dropna(subset=["projected_pace_adjusted_actual"]))
        .mark_point(color=CYAN, filled=True, shape="diamond", size=80)
        .encode(
            x=alt.X("projected_pace_adjusted_actual:Q"),
            y=alt.Y("player_name:N", sort=None),
            tooltip=[
                alt.Tooltip("player_name:N", title="Player"),
                alt.Tooltip(
                    "projected_pace_adjusted_actual:Q",
                    title="Adjusted at projected PPG",
                    format=".1f",
                ),
                alt.Tooltip("injury_games:Q", title="Injury misses"),
                alt.Tooltip(
                    "projected_pace_injury_points:Q",
                    title="Projected-pace addition",
                    format=".1f",
                ),
            ],
        )
    )
    actual_pace_adjusted = (
        alt.Chart(chart_frame.dropna(subset=["actual_pace_adjusted_actual"]))
        .mark_point(color=CORAL, filled=True, shape="circle", size=70)
        .encode(
            x=alt.X("actual_pace_adjusted_actual:Q"),
            y=alt.Y("player_name:N", sort=None),
            tooltip=[
                alt.Tooltip("player_name:N", title="Player"),
                alt.Tooltip(
                    "actual_pace_adjusted_actual:Q",
                    title="Adjusted at actual PPG",
                    format=".1f",
                ),
                alt.Tooltip("injury_games:Q", title="Injury misses"),
                alt.Tooltip(
                    "actual_pace_injury_points:Q",
                    title="Actual-pace addition",
                    format=".1f",
                ),
            ],
        )
    )
    st.altair_chart(
        _chart_theme(
            (bars + projected_pace_adjusted + actual_pace_adjusted).properties(
                height=max(360, len(chart_frame) * 32)
            )
        ),
        width="stretch",
    )
    st.caption(
        "Green: projected. Amber: actual. Cyan diamond: injury games filled at "
        "projected PPG. Coral circle: injury games filled at actual PPG. Only "
        "Out and reserve-list absences count."
    )

    table = display.select(
        pl.col("position_rank").alias("Rank"),
        pl.col("position").alias("Pos"),
        pl.col("player_name").alias("Player"),
        pl.col("projected_points").alias("Projected"),
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


def render() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Fantasy Forecast Center",
        page_icon="F",
        layout="wide",
        initial_sidebar_state="auto",
    )
    _inject_styles()
    try:
        source = _load_data()
    except (DashboardDataError, OSError, pl.exceptions.PolarsError) as error:
        st.error(f"Could not open the prediction sheet: {error}")
        st.stop()

    selected, season, positions, model = _global_filters(source)
    selected_season = selected.filter(pl.col("target_season") == season)
    _masthead(selected_season, model)
    workspace = st.radio(
        "Workspace",
        ["Draft Board", "Weekly Decisions", "Model Room"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if workspace == "Draft Board":
        _draft_view(
            selected,
            season=season,
            positions=positions,
            model=model,
        )
    elif workspace == "Weekly Decisions":
        _weekly_view(selected, season=season, positions=positions)
    else:
        _model_room(source.filter(pl.col("target_season") == season))


def run() -> Never:
    """Launch the dashboard through the console script."""
    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path), "--", *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    render()
