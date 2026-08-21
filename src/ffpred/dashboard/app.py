"""Streamlit application for exploring fantasy-football predictions."""

from __future__ import annotations

import argparse
import sys
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

PAPER = "#f3eedf"
INK = "#17243b"
RED = "#b7352d"
GOLD = "#d8a928"
BLUE = "#2f6680"
MUTED = "#5d6670"


def _inject_styles() -> None:
    st.html(
        """
        <!--
        THESIS: A working fantasy war room, not a generic analytics dashboard.
        OWN-WORLD: Ink-blue auction sheets, red stamps, gold highlights, hard
        rules, clipped corners, and handwritten value marks on warm paper.
        STORY: Load model sheets, rank a season for draft day, then move to a
        separate matchup desk for weekly choices and inspect model evidence.
        FIRST VIEWPORT: A ledger masthead, artifact status, and three explicit
        workspaces above a full-width ranking sheet.
        FORM: Dense war-room ledger, fourth grounded direction; fixed-sheet
        staging selected for scanability; seed 40420016.
        -->
        <style>
        :root {
            --paper: #f3eedf;
            --paper-deep: #e8dec7;
            --ink: #17243b;
            --red: #b7352d;
            --gold: #d8a928;
            --blue: #2f6680;
            --muted: #5d6670;
            --rule: #9d927b;
        }
        .stApp {
            background:
                linear-gradient(rgba(23, 36, 59, .035) 1px, transparent 1px),
                var(--paper);
            background-size: 100% 28px;
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: var(--ink);
            border-right: 1px solid #78849a;
        }
        [data-testid="stSidebar"] * { color: #f8f1df; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background: #233552;
            border-color: #78849a;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 2.25rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 {
            color: var(--ink) !important;
            font-family: "Bahnschrift Condensed", "Franklin Gothic Medium",
                sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -.035em !important;
        }
        h1 { font-size: clamp(2.4rem, 5vw, 5.3rem) !important; line-height: .9; }
        h2 {
            border-bottom: 3px solid var(--ink);
            padding-bottom: .35rem;
            margin-top: 2.7rem !important;
        }
        p, label, button, input, [data-testid="stMarkdownContainer"] {
            font-family: "Aptos", Tahoma, sans-serif;
        }
        [data-testid="stMain"] label { color: var(--ink) !important; }
        .war-masthead {
            position: relative;
            border-top: 10px solid var(--ink);
            border-bottom: 3px solid var(--ink);
            padding: 1rem 0 1.15rem;
            margin-bottom: 1rem;
        }
        .war-masthead h1 { margin: 0; text-transform: uppercase; }
        .war-masthead p {
            color: var(--muted);
            font-size: 1.05rem;
            margin: .6rem 0 0;
            max-width: 70ch;
        }
        .stamp {
            position: absolute;
            right: 0;
            top: 1.25rem;
            border: 3px solid var(--red);
            color: var(--red);
            font-family: "Bahnschrift Condensed", Tahoma, sans-serif;
            font-size: .8rem;
            font-weight: 900;
            letter-spacing: .12em;
            padding: .4rem .65rem;
            text-transform: uppercase;
            transform: rotate(-2deg);
        }
        .evidence-strip {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem 1.6rem;
            background: var(--paper-deep);
            border-bottom: 1px solid var(--rule);
            padding: .65rem .8rem .8rem;
            margin-bottom: 1rem;
            font-family: "Bahnschrift Condensed", Tahoma, sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
            clip-path: polygon(
                0 0, calc(100% - 12px) 0, 100% 12px,
                100% 100%, 12px 100%, 0 calc(100% - 12px)
            );
        }
        .evidence-strip strong {
            color: var(--red);
            display: inline-block;
            font-family: "Segoe Print", "Bradley Hand", cursive;
            letter-spacing: -.04em;
            transform: rotate(-1deg);
        }
        [data-testid="stRadio"] > div {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0;
            border-bottom: 4px solid var(--ink);
            max-width: 31rem;
        }
        [data-testid="stRadio"] label {
            background: var(--paper-deep);
            border: 1px solid var(--ink);
            border-bottom: 0;
            padding: .7rem 1.2rem;
            margin-right: -1px;
            font-weight: 800;
            text-transform: uppercase;
            width: 100%;
        }
        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--ink);
            color: #fff7e7;
        }
        div[data-testid="stDataFrame"] {
            border: 2px solid var(--ink);
            box-shadow: 6px 6px 0 rgba(23, 36, 59, .16);
        }
        .stButton > button, .stDownloadButton > button {
            border: 2px solid var(--ink);
            border-radius: 0;
            background: var(--gold);
            color: var(--ink);
            box-shadow: 3px 3px 0 var(--ink);
            font-weight: 800;
            text-transform: uppercase;
        }
        [data-testid="stFileUploader"] button {
            background: var(--gold) !important;
            border: 2px solid #fff7e7 !important;
            border-radius: 0 !important;
            color: var(--ink) !important;
            clip-path: polygon(
                0 0, calc(100% - 8px) 0, 100% 8px,
                100% 100%, 8px 100%, 0 calc(100% - 8px)
            );
            font-weight: 800;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--red);
            color: var(--red);
            transform: translate(1px, 1px);
            box-shadow: 2px 2px 0 var(--red);
        }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
            outline: 3px solid var(--blue);
            outline-offset: 3px;
        }
        [data-testid="stAlert"] {
            border-radius: 0;
            border: 2px solid var(--ink);
            background: #faf4e5;
        }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div {
            border-radius: 0;
        }
        @media (max-width: 760px) {
            .block-container { padding: 1rem .75rem 3rem; }
            .stamp { position: static; display: inline-block; margin-top: 1rem; }
            [data-testid="stRadio"] > div { max-width: none; }
            [data-testid="stRadio"] label {
                font-size: .7rem;
                justify-content: center;
                padding: .55rem .2rem;
                text-align: center;
            }
            .evidence-strip { display: grid; grid-template-columns: 1fr 1fr; }
        }
        </style>
        """
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--predictions", action="append", type=Path, default=[])
    arguments, _ = parser.parse_known_args()
    return arguments


def _discover_prediction_files() -> list[Path]:
    candidates = [
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
        st.markdown("## Model sheets")
        uploads = st.file_uploader(
            "Add prediction Parquet files",
            type=["parquet"],
            accept_multiple_files=True,
            help="Each file needs player, season, week, and prediction columns.",
        )
        if paths:
            st.caption("Loaded from disk")
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
        st.markdown("## Board controls")
        season = int(st.selectbox("Season", seasons))
        chosen_positions = [
            str(position)
            for position in st.multiselect(
                "Position",
                positions,
                default=positions,
                help="The current training pipeline produces quarterback rows.",
            )
        ]
        model = st.selectbox(
            "Model sheet",
            models,
            index=0,
            help="Consensus averages predictions when multiple sheets are loaded.",
        )
        st.divider()
        st.caption(
            "Predictions support decisions; they do not guarantee player outcomes."
        )

    selected = select_model(frame, model)
    return selected, season, chosen_positions, model


def _masthead(frame: pl.DataFrame, model: str) -> None:
    positions = ", ".join(sorted(frame["position"].unique().to_list()))
    actual_rows = frame[ACTUAL_COLUMN].count()
    artifact_mode = "Historical backtest" if actual_rows else "Forward projection"
    st.html(
        f"""
        <section class="war-masthead">
          <h1>Fantasy War Room</h1>
          <p>Season value belongs on the draft board. Weekly matchups belong on
          their own desk. Use the same evidence without mixing the decisions.</p>
          <span class="stamp">{artifact_mode}</span>
        </section>
        <div class="evidence-strip">
          <span>Sheet <strong>{model}</strong></span>
          <span>Rows <strong>{frame.height:,}</strong></span>
          <span>Positions <strong>{positions}</strong></span>
          <span>Mode <strong>{artifact_mode}</strong></span>
        </div>
        """
    )
    if positions == "QB":
        with st.expander("Why does this artifact only show QB?"):
            st.write(
                "The current model pipeline only trains quarterback predictions. "
                "The position control expands automatically when future artifacts "
                "include other positions."
            )
    if actual_rows:
        st.caption(
            "This sheet includes completed-game outcomes, so ranks shown here are "
            "a model backtest, not a live upcoming-season forecast."
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
        "Rank total season value here. Weekly volatility and model disagreement "
        "stay visible, but they do not replace projected points. Opportunity "
        "share and team volume expose fragile production behind recent scores."
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
    tooltips = [
        alt.Tooltip("player_name:N", title="Player"),
        alt.Tooltip("projected_points:Q", title="Projected", format=".1f"),
        alt.Tooltip("points_per_game:Q", title="Per game", format=".1f"),
        alt.Tooltip("volatility:Q", title="Weekly swing", format=".1f"),
    ]
    if "team" in display.columns:
        tooltips.append(alt.Tooltip("team:N", title="Team"))
    if "target_share" in display.columns:
        tooltips.append(
            alt.Tooltip("target_share:Q", title="Target share", format=".1%")
        )
    if "carry_share" in display.columns:
        tooltips.append(alt.Tooltip("carry_share:Q", title="Carry share", format=".1%"))
    if "team_offensive_plays" in display.columns:
        tooltips.append(
            alt.Tooltip(
                "team_offensive_plays:Q",
                title="Play volume",
                format=".1f",
            )
        )
    bars = (
        alt.Chart(chart_frame)
        .mark_bar(color=INK, cornerRadiusEnd=0)
        .encode(
            x=alt.X("projected_points:Q", title="Projected season points"),
            y=alt.Y("player_name:N", sort=None, title=None),
            tooltip=tooltips,
        )
    )
    labels = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color=INK,
        font="Segoe Print",
        fontWeight="bold",
    ).encode(text=alt.Text("projected_points:Q", format=".0f"))
    actual = (
        alt.Chart(chart_frame.dropna(subset=["actual_points"]))
        .mark_tick(color=RED, thickness=3, size=18)
        .encode(x="actual_points:Q", y=alt.Y("player_name:N", sort=None))
    )
    st.altair_chart(
        (bars + labels + actual)
        .properties(height=max(360, len(chart_frame) * 28))
        .configure(background=PAPER)
        .configure_view(fill=PAPER, stroke=None)
        .configure_axis(gridColor="#cfc5ae", labelColor=INK, titleColor=INK),
        width="stretch",
    )
    st.caption("Ink bar: projection. Red mark: actual total when available.")

    table_columns: list[pl.Expr] = [
        pl.col("position_rank").alias("Rank"),
        pl.col("position").alias("Pos"),
        pl.col("player_name").alias("Player"),
        pl.col("projected_points").alias("Projected"),
        pl.col("points_per_game").alias("Per game"),
        pl.col("projected_games").alias("Games"),
        pl.col("volatility").alias("Weekly swing"),
        pl.col("model_spread").alias("Model spread"),
        pl.col("actual_points").alias("Actual"),
    ]
    if "team" in display.columns:
        table_columns.insert(3, pl.col("team").alias("Team"))
    opportunity_display_columns = {
        "target_share": (pl.col("target_share") * 100).alias("Target share"),
        "carry_share": (pl.col("carry_share") * 100).alias("Carry share"),
        "team_offensive_plays": pl.col("team_offensive_plays").alias("Play volume"),
        "team_pass_rate": (pl.col("team_pass_rate") * 100).alias("Pass rate"),
        "usage_warning": pl.col("usage_warning").alias("Volume flag"),
        "opportunity_basis": pl.col("opportunity_basis").alias("Usage basis"),
    }
    table_columns.extend(
        expression
        for column, expression in opportunity_display_columns.items()
        if column in display.columns
    )
    table = display.select(table_columns)
    column_config = {
        "Projected": st.column_config.NumberColumn(format="%.1f"),
        "Per game": st.column_config.NumberColumn(format="%.1f"),
        "Weekly swing": st.column_config.NumberColumn(format="%.1f"),
        "Model spread": st.column_config.NumberColumn(format="%.1f"),
        "Actual": st.column_config.NumberColumn(format="%.1f"),
        "Target share": st.column_config.NumberColumn(format="%.1f%%"),
        "Carry share": st.column_config.NumberColumn(format="%.1f%%"),
        "Play volume": st.column_config.NumberColumn(format="%.1f"),
        "Pass rate": st.column_config.NumberColumn(format="%.1f%%"),
    }
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )
    if "opportunity_basis" in display.columns:
        st.caption(
            "Shares and rates use depth-chart projections when projected_* "
            "columns are supplied; otherwise they are leakage-safe trailing "
            "10-game values. Play volume is pass attempts plus carries. "
            "Draft-season roster changes still require a depth-chart "
            "projection source."
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
            .mark_bar(color=BLUE)
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
            comparison_chart.configure(background=PAPER)
            .configure_view(fill=PAPER, stroke=None)
            .configure_axis(gridColor="#cfc5ae", labelColor=INK, titleColor=INK),
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
                        range=[INK, RED, BLUE, GOLD, "#6d5b87"],
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
            trend.configure(background=PAPER)
            .configure_view(fill=PAPER, stroke=None)
            .configure_axis(gridColor="#cfc5ae", labelColor=INK, titleColor=INK),
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
        .mark_bar(color=RED)
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
        chart.configure(background=PAPER)
        .configure_view(fill=PAPER, stroke=None)
        .configure_axis(gridColor="#cfc5ae", labelColor=INK, titleColor=INK),
        width="stretch",
    )


def render() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Fantasy War Room",
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
    _masthead(selected, model)
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
        _model_room(source)


def run() -> Never:
    """Launch the dashboard through the console script."""
    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path), "--", *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    render()
