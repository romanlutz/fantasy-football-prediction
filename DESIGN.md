---
name: Fantasy Forecast Center
description: A night-game forecast command center where model provenance stays as visible as player rank.
colors:
  charcoal-field: "#0d1517"
  panel-deep: "#121e21"
  field-surface: "#162326"
  surface-high: "#1b2b2f"
  grid-line: "#294044"
  grid-line-strong: "#3c5b60"
  broadcast-text: "#dce8e4"
  telemetry-muted: "#94aaa5"
  field-green: "#77d584"
  green-deep: "#214c34"
  comparison-cyan: "#62b9c8"
  actual-amber: "#d4af63"
  evidence-coral: "#d98578"
  evidence-violet: "#9b91c9"
typography:
  display:
    fontFamily: "Bahnschrift, Arial Narrow, Segoe UI, sans-serif"
    fontSize: "clamp(2.4rem, 4vw, 4.1rem)"
    fontWeight: 650
    lineHeight: 0.94
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Bahnschrift, Arial Narrow, Segoe UI, sans-serif"
    fontSize: "clamp(1.65rem, 2.6vw, 2.3rem)"
    fontWeight: 620
    letterSpacing: "-0.025em"
  body:
    fontFamily: "Segoe UI, Arial, sans-serif"
    fontWeight: 400
    lineHeight: 1.58
  label:
    fontFamily: "Segoe UI, Arial, sans-serif"
    fontSize: "0.68rem"
    fontWeight: 650
    letterSpacing: "0.1em"
rounded:
  xs: "6px"
  sm: "7px"
  md: "8px"
  lg: "9px"
  xl: "10px"
  full: "50%"
components:
  button-primary:
    backgroundColor: "{colors.green-deep}"
    textColor: "#e5f3e7"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
  button-primary-hover:
    backgroundColor: "#2a6140"
    textColor: "#eef8ef"
  button-upload:
    backgroundColor: "{colors.surface-high}"
    textColor: "{colors.broadcast-text}"
    rounded: "{rounded.sm}"
  workspace-tab:
    backgroundColor: "transparent"
    textColor: "{colors.broadcast-text}"
    typography: "{typography.body}"
    rounded: "{rounded.xs}"
  workspace-tab-active:
    backgroundColor: "{colors.green-deep}"
    textColor: "#e5f3e7"
  command-header:
    backgroundColor: "linear-gradient(112deg, {colors.surface-high}, {colors.panel-deep})"
    rounded: "{rounded.xl}"
  forecast-state:
    backgroundColor: "rgba(119, 213, 132, .075)"
    textColor: "{colors.field-green}"
    rounded: "{rounded.md}"
  dataframe-container:
    rounded: "{rounded.md}"
  sidebar-panel:
    backgroundColor: "{colors.panel-deep}"
    textColor: "{colors.broadcast-text}"
  input-field:
    backgroundColor: "{colors.field-surface}"
    textColor: "{colors.broadcast-text}"
    rounded: "{rounded.sm}"
  alert:
    backgroundColor: "{colors.field-surface}"
    rounded: "{rounded.md}"
---

# Design System: Fantasy Forecast Center

## Overview

**Creative North Star: "The Night-Game Command Deck"**

This is a stadium operations display for player evidence, not a generic analytics dashboard: a charcoal field, gridded like turf under stadium lights, carries a single restrained accent strategy so model provenance never gets lost behind ranking noise. Field Green is the only signal for state and action; Comparison Cyan and Actual Amber carry the two evidence roles a forecast needs — what the models project against each other, and what already happened. There is no paper, no ink stamp, no cursive handwriting anywhere in this system; every trace of the prior ledger metaphor has been retired.

The build stays disciplined about role separation: green signals state and action, cyan compares model output, amber marks confirmed fact, and a small five-color evidence palette (green, cyan, amber, coral, violet) extends only as far as the weekly trend chart needs it. A compact command header and a live telemetry rail lead directly into three persistent workspaces (Draft Board, Weekly Decisions, Model Room), rendered as one equal-width row at every viewport, because moving between season-long and single-week decisions without losing model context is the product's central promise.

**Key Characteristics:**
- Charcoal-field canvas with a faint green-tinted 40px grid, read as a night-lit stadium turf
- Small radii everywhere (6–10px) plus one fully circular status dot; no clip-path notches, no square corners
- Soft, blurred, black-based drop shadows for elevation — never a hard flat offset
- A compact command header with an animated field-lock underline and a live telemetry rail as the signature devices
- Three equal, never-stacking workspace tabs above the active ranking surface

## Colors

A charcoal, night-game palette where nearly all of the interface is dark neutral field, and each accent color carries exactly one evidentiary role.

### Primary
- **Field Green** (`#77d584`): The action/state signal color — the status dot, the forecast-state badge text, the active workspace pill, the primary/download buttons (via its deep variant), the slider thumb, the active toggle track, and the draft-board projection bars. It is the "the model is live and this is its call" color.
- **Green Deep** (`#214c34`): Field Green's recessed low-light variant — the fill for primary/download buttons and the active workspace pill, so controls read as illuminated turf rather than a flat color chip.

### Secondary
- **Comparison Cyan** (`#62b9c8`): The weekly-comparison bar chart, `code` text, and the button `focus-visible` outline — a color-independent focus cue kept deliberately distinct from the green hover state.

### Tertiary
- **Actual Amber** (`#d4af63`): The confirmed/actual-value marker — the draft-board's actual-points tick marks and the "actual" line in the weekly trend chart. Never used for a projection.
- **Evidence Coral** (`#d98578`): The model-room mean-absolute-error bar chart, and the fourth line in the multi-player trend comparison.
- **Evidence Violet** (`#9b91c9`): Reserved for the fifth compared player's trend line only — the rarest color in the system.

### Neutral
- **Charcoal Field** (`#0d1517`): The base canvas, overlaid with a faint two-axis 40px grid of green-tinted hairlines (`rgba(119, 213, 132, .022)` / `rgba(119, 213, 132, .016)`) so the page itself reads as a night-lit field.
- **Panel Deep** (`#121e21`): The sidebar and the workspace-selector tray fill, one step lighter than the canvas.
- **Field Surface** (`#162326`): Dataframes, inputs, alerts, and every Altair chart's background.
- **Surface High** (`#1b2b2f`): The command header's gradient highlight and the sidebar's form-field fill — the lightest charcoal step.
- **Grid Line** (`#294044`): Borders throughout, and every Altair chart's axis, gridline, and tick color.
- **Grid Line Strong** (`#3c5b60`): Stronger borders on inputs, alerts, and sidebar form fields.
- **Broadcast Text** (`#dce8e4`): The primary text color throughout, headings and body alike.
- **Telemetry Muted** (`#94aaa5`): Captions, paragraph copy, and telemetry-item labels.

### Named Rules
**The Field Signal Rule.** Field Green (and its Green Deep variant) is reserved for state and action chrome — the status dot, the forecast-state badge, the active workspace pill, buttons, the slider thumb, and the active toggle. It never appears as a comparison or actual-value color inside a chart.

**The Confirmed-Fact Amber Rule.** Actual Amber marks only confirmed, already-happened values — the draft-board's actual-points tick and the trend chart's actual line. A projection is never drawn in amber.

**The Evidence Palette Order Rule.** When the weekly trend chart compares more than one player, added lines take colors from a fixed five-step sequence — Field Green, Comparison Cyan, Actual Amber, Evidence Coral, Evidence Violet — always in that order, never reshuffled or reassigned per player.

## Typography

**Display Font:** Bahnschrift (with Arial Narrow, Segoe UI, sans-serif fallback)
**Body Font:** Segoe UI (with Arial, sans-serif fallback)

**Character:** One condensed, broadcast-grade display face carries every heading; the same plain, highly legible Segoe UI face runs everything else — paragraphs, labels, buttons, inputs, telemetry values, and dataframe cells — so the system reads like a stadium scoreboard typeface, not a mixed type family. There is no separate label or mono face; label styling comes from weight, tracking, and case on the same body stack.

### Hierarchy
- **Display** (`h1`, 650, `clamp(2.4rem, 4vw, 4.1rem)`, line-height .94, `-0.035em` letter-spacing): The "Fantasy Forecast Center" masthead title.
- **Headline** (`h2`, 620, `clamp(1.65rem, 2.6vw, 2.3rem)`, `-0.025em` letter-spacing, 2.6rem top margin): Workspace section headers — "Draft board," "Weekly decisions," "Model room."
- **Title** (`h3`, 600): Sub-headers within a workspace; inherits the `h1`/`h2` Bahnschrift stack.
- **Body** (400, Segoe UI, 1.58 line-height for paragraphs): Paragraphs, form labels, buttons, inputs, and dataframe cell text.
- **Label** (650, `.65–.68rem`, `.095–.11em` letter-spacing, uppercase, Segoe UI): The telemetry-rail item captions and the forecast-state "Forecast state" caption — small and wide-tracked, but on the body face, not a distinct one.

### Named Rules
**The One Display Face Rule.** Bahnschrift is the only face used for `h1`–`h3`. Every other role — including uppercase labels — stays on the Segoe UI body stack, distinguished by weight and tracking, never by swapping fonts.

## Layout

The container is capped at 1440px with 2rem of top padding and 4.5rem of bottom padding, sitting on the charcoal canvas's faint two-axis 40px green grid. The first viewport stacks, in order: a command header (title and description beside a bordered forecast-state badge, with an animated Field Green underline growing along its bottom edge), a telemetry rail of stat cells directly beneath it, a three-cell equal-width workspace selector, then the active workspace's full-width content and ranking table.

A permanent Panel Deep sidebar (bordered `1px solid` Grid Line) holds two labeled sections — "Data feeds" (the Parquet uploader and the active feed list) and "Mission controls" (season, position, and model filters) — one step darker than the main canvas, though only by a single tonal step rather than a hard color break.

At the 760px breakpoint, container padding becomes `3.35rem .8rem 3rem`, the command header collapses to a single column (the forecast-state badge drops below the title and description instead of sitting beside them), the telemetry rail reflows from `auto-fit, minmax(125px, 1fr)` into a fixed 2-column grid with alternating border removal, and workspace-pill labels shrink to `.74rem` — but the selector's underlying CSS grid stays `repeat(3, minmax(0, 1fr))` at every width, so the three destinations never stack or collapse into a dropdown.

### Named Rules
**The Equal Three Rule.** The workspace selector is a real CSS grid of `repeat(3, minmax(0, 1fr))` — Draft Board, Weekly Decisions, Model Room always render as one equal-width row, at every viewport width, with only label size and padding shrinking below 760px.

## Elevation & Depth

This system uses soft, blurred, black-based drop shadows exclusively — the opposite of a hard offset. Depth reads as a gentle lift off the dark charcoal canvas: dataframes and buttons each carry a diffuse black shadow whose opacity and blur radius scale with how "raised" the element should feel, never a flat colored offset.

### Shadow Vocabulary
- **Ranking sheet lift** (`box-shadow: 0 14px 34px rgba(0, 0, 0, .18)`): Applied to every `st.dataframe` table.
- **Button rest** (`box-shadow: 0 8px 20px rgba(0, 0, 0, .16)`): Resting state for primary and download buttons.
- **Button hover** (`box-shadow: 0 10px 24px rgba(0, 0, 0, .22)`, paired with `transform: translateY(-1px)`): The button hover state — the shadow deepens and the button lifts, rather than pressing down.

### Named Rules
**The Soft Lift Rule.** Every shadow in this system is a soft, blurred, black-based drop shadow (`0 Npx Npx rgba(0,0,0,.16–.22)`). A hard, non-blurred, colored offset shadow never appears here.

## Shapes

Every interactive surface carries a small radius instead of a square or clipped corner: 6px on the workspace pill, 7px on buttons/inputs/selects, 8px on dataframes/alerts/the expander/the forecast-state badge, 9px on the workspace-selector tray, and 10px on the command header's top corners only (its bottom stays square, so the header reads as a mounted panel). The status dot is the one fully circular (50%) shape in the system. No `clip-path` notch appears anywhere. The command header's signature move is its `::before` underline: a 3px Field Green bar that animates from 4% to 18% width on load (`field-lock`, 700ms `cubic-bezier(.2, .8, .2, 1)`, disabled under `prefers-reduced-motion`).

### Named Rules
**The Small-Radius Rule.** Corner treatment is a small radius (6–10px) scaled to the surface's size, never zero and never a clip-path notch. The single exception is the fully circular (50%) status dot.

## Components

### Buttons
- **Shape:** 7px radius, `1px solid rgba(119, 213, 132, .48)` border.
- **Primary:** Green Deep (`#214c34`) background, `#e5f3e7` text, weight 650, with the Button Rest shadow at rest.
- **Hover / Focus:** Hover lightens the background to `#2a6140`, solidifies the border to Field Green, brightens the text to `#eef8ef`, and lifts the button (`translateY(-1px)`) into the Button Hover shadow. `focus-visible` gets a distinct `2px solid` Comparison Cyan outline with 2px offset — deliberately not green, so focus is never confused with hover.
- **Secondary (file uploader):** Surface High (`#1b2b2f`) background, Grid Line Strong border, Broadcast Text color, the same 7px radius — no `clip-path`, no shadow; visually quieter than the primary action.

### Cards / Containers
- **Corner Style:** 8px radius on dataframes, alerts, and the expander.
- **Background:** Dataframes, inputs, and alerts sit on Field Surface (`#162326`); the expander sits on a translucent near-match (`rgba(22, 35, 38, .68)`).
- **Shadow Strategy:** Dataframes use the Ranking Sheet Lift shadow (see Elevation).
- **Border:** `1px solid` Grid Line on dataframes and the expander; `1px solid` Grid Line Strong on alerts.
- **Internal Padding:** Telemetry items use `.8rem 1rem .9rem`; the forecast-state badge uses `.75rem .9rem`.

### Inputs / Fields
- **Style:** 7px radius, Field Surface background, Grid Line Strong border. Inside the sidebar, the same controls fill with Surface High instead, so they read one step lighter than the panel behind them.
- **Focus:** Buttons carry an explicit Comparison Cyan `focus-visible` outline; other controls use BaseWeb/Streamlit's default focus treatment.

### Navigation (Workspace Selector)
- **Style:** A `repeat(3, minmax(0, 1fr))` grid on a Panel Deep tray (9px radius, `1px solid` Grid Line border); each pill hides its native radio dot, centers its label, and takes a 6px radius. The checked pill fills Green Deep with a `rgba(119, 213, 132, .28)` border and `#e5f3e7` text; unchecked pills stay transparent.
- **Mobile:** The same three-cell grid persists; only label size (`.74rem`) and padding shrink.

### Command Header + Forecast State (signature component)
A `112deg` gradient panel (Surface High → Panel Deep, 10px top radius) holding the masthead title and description on the left, and a bordered Forecast State badge — status dot, "Forecast state" label, and the live artifact mode ("Point-in-time replay" / "Upcoming forecast" / "Rolling backtest") — on the right. Its bottom edge carries the `field-lock` animated Field Green underline, growing from 4% to 18% width on load.

### Telemetry Rail (signature component)
A dense horizontal strip of stat cells directly under the command header (`auto-fit, minmax(125px, 1fr)`), each pairing an uppercase muted-text label with a bold value: target season, model view, matchup rows, positions, and — for frozen forecasts — history-through-season and forecast-lock date. It is the live telemetry the system promises: provenance facts sit at the same visual weight as the ranking data below them.

### Sidebar (signature component)
A permanent Panel Deep filing panel split into two labeled sections — "Data feeds" (the Parquet uploader plus the active feed list, shown as `st.code` chips) and "Mission controls" (season, position, and model filters) — one step darker than the main canvas, bordered by a single `1px solid` Grid Line rule rather than a hard color break.

## Do's and Don'ts

### Do:
- **Do** reserve Field Green (and Green Deep) for state and action chrome only — the status dot, forecast-state badge, active workspace pill, buttons, slider thumb, and toggle.
- **Do** keep the workspace selector a true `repeat(3, minmax(0, 1fr))` grid at every viewport; shrink only label size and padding below 760px.
- **Do** use soft, blurred, black-based shadows (`0 Npx Npx rgba(0,0,0,.16-.22)`) for elevation; never a hard offset.
- **Do** follow the fixed five-color evidence-palette order (green, cyan, amber, coral, violet) when the trend chart adds compared players.
- **Do** keep Bahnschrift exclusive to `h1`–`h3`; everything else, including uppercase labels, stays on Segoe UI.
- **Do** respect `prefers-reduced-motion` by disabling the command header's `field-lock` animation.

### Don't:
- **Don't** use Actual Amber for a projected or comparison value — it marks confirmed actuals only.
- **Don't** add a `clip-path` notch or a square corner anywhere; every surface carries a small radius (6–10px) except the fully circular status dot.
- **Don't** stack, wrap, or collapse the workspace selector into a dropdown on narrow viewports.
- **Don't** substitute the Comparison Cyan `focus-visible` outline with the Field Green hover treatment; the two must stay visually distinct.
