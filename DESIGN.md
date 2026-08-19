---
name: Fantasy War Room
description: A dense war-room ledger for ranking a season and running weekly matchups against model evidence.
colors:
  ink-navy: "#17243b"
  warm-paper: "#f3eedf"
  paper-deep: "#e8dec7"
  red-stamp: "#b7352d"
  gold-highlight: "#d8a928"
  blue-info: "#2f6680"
  muted-ink: "#5d6670"
  rule-line: "#9d927b"
  sidebar-cream: "#f8f1df"
  sidebar-field: "#233552"
  chart-grid: "#cfc5ae"
  chart-line-violet: "#6d5b87"
  alert-paper: "#faf4e5"
  radio-active-text: "#fff7e7"
typography:
  display:
    fontFamily: "Bahnschrift Condensed, Franklin Gothic Medium, sans-serif"
    fontSize: "clamp(2.4rem, 5vw, 5.3rem)"
    fontWeight: 800
    lineHeight: 0.9
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Bahnschrift Condensed, Franklin Gothic Medium, sans-serif"
    fontWeight: 800
    letterSpacing: "-0.035em"
  body:
    fontFamily: "Aptos, Tahoma, sans-serif"
    fontWeight: 400
  label:
    fontFamily: "Bahnschrift Condensed, Tahoma, sans-serif"
    fontWeight: 700
    letterSpacing: "0.04em"
  mark:
    fontFamily: "Segoe Print, Bradley Hand, cursive"
    fontWeight: 700
rounded:
  flat: "0px"
components:
  button-primary:
    backgroundColor: "{colors.gold-highlight}"
    textColor: "{colors.ink-navy}"
    typography: "{typography.body}"
    rounded: "{rounded.flat}"
  button-primary-hover:
    textColor: "{colors.red-stamp}"
  workspace-tab:
    backgroundColor: "{colors.paper-deep}"
    textColor: "{colors.ink-navy}"
    typography: "{typography.label}"
    rounded: "{rounded.flat}"
    padding: "0.7rem 1.2rem"
  workspace-tab-active:
    backgroundColor: "{colors.ink-navy}"
    textColor: "{colors.radio-active-text}"
  sidebar-panel:
    backgroundColor: "{colors.ink-navy}"
    textColor: "{colors.sidebar-cream}"
  stamp:
    textColor: "{colors.red-stamp}"
    typography: "{typography.label}"
    padding: "0.4rem 0.65rem"
  evidence-strip:
    backgroundColor: "{colors.paper-deep}"
    padding: "0.65rem 0.8rem 0.8rem"
  dataframe-container:
    rounded: "{rounded.flat}"
---

# Design System: Fantasy War Room

## Overview

**Creative North Star: "The Fantasy War Room"**

This is a working war room for player evidence, not a generic analytics dashboard: warm ruled paper stands in for a legal pad, ink-navy carries every hard rule and border, a rotated red stamp reports artifact status like a rubber stamp on a filed sheet, and gold marks the one action that matters on any given screen. The system never softens itself with rounded corners or blurred shadows — depth comes from flat, offset "stamp" shadows, and corners get personality only from clipped-path notches, never radius.

The build stays disciplined about role separation: ink structures, red marks status and hard-won facts (actuals, hover/error), gold acts, blue informs, and a cursive "handwritten" face is reserved exclusively for data values scrawled onto the ledger — never for body copy or structural labels. Three equal workspaces (Draft Board, Weekly Decisions, Model Room) sit as one unbroken row under the masthead at every viewport width, because moving between season-long and single-week decisions is the product's central promise.

**Key Characteristics:**
- Warm ruled-paper canvas with a permanent dark ink-navy sidebar as the control desk
- Zero border-radius everywhere; clipped-corner notches are the only corner treatment
- Hard, non-blurred offset shadows ("ledger stack" / "stamped button"), never soft glows
- A rotated red stamp and tilted cursive "handwritten" marks as the signature evidence devices
- Three equal, never-stacking workspace tabs above a full-width ranking sheet

## Colors

A warm, ink-on-paper palette where color carries meaning sparingly: most of the interface is ink on paper, and every accent color has exactly one job.

### Primary
- **Ink Navy** (`#17243b`): The dominant structural color — all headings, hard rules, borders, chart marks, and the sidebar's fill. It is the "pen" the whole ledger is written in.

### Secondary
- **Gold Highlight** (`#d8a928`): The single actionable color. Reserved for primary/download buttons and the file-uploader control — nothing else uses it.

### Tertiary
- **Red Stamp** (`#b7352d`): Status and hard-fact color. Used for the artifact-mode stamp, the "actual points" chart marks, the handwritten values in the evidence strip, and the hover/error state on buttons.
- **Blue Info** (`#2f6680`): Informational accent. Used for the weekly-comparison chart bars and as the button focus-visible outline — a color-independent focus cue distinct from the red hover state.

### Neutral
- **Warm Paper** (`#f3eedf`): The base canvas background, overlaid with a faint 28px-tall ruled-line pattern (`rgba(23, 36, 59, .035)`) to read as ledger paper.
- **Paper Deep** (`#e8dec7`): Raised-surface paper — the evidence strip and unselected workspace tabs sit here, one shade darker than the canvas.
- **Muted Ink** (`#5d6670`): Secondary/caption text, e.g. the masthead subhead.
- **Rule Line** (`#9d927b`): Hairline dividers, e.g. under the evidence strip.
- **Sidebar Cream** (`#f8f1df`): Text color inside the ink-navy sidebar.
- **Sidebar Field** (`#233552`): Input/select fill inside the sidebar, one step lighter than the sidebar background.
- **Chart Grid** (`#cfc5ae`): Gridlines in every Altair chart, always drawn on a Warm Paper chart background.
- **Alert Paper** (`#faf4e5`): Background for `st.warning`/`st.error` alerts, lighter than Paper Deep so alerts read as freshly attached notices.

### Named Rules
**The One Action Rule.** Gold appears only on primary buttons and the uploader control — the single actionable color on any screen. It is never used decoratively or for status.

**The Stamp Rule.** Red never carries body text or long-form emphasis. It is reserved for the status stamp, handwritten actual-value marks, and the hover/error interaction states — always a fact or an action, never prose.

## Typography

**Display Font:** Bahnschrift Condensed (with Franklin Gothic Medium, sans-serif fallback)
**Body Font:** Aptos (with Tahoma, sans-serif fallback)
**Label/Mono Font:** Segoe Print (with Bradley Hand, cursive fallback) — the handwritten value-mark face

**Character:** A heavy, condensed grotesque for structure paired with a plain humanist body face, then broken deliberately by a cursive "handwritten" font used only where a real analyst would scrawl a number onto a printed sheet.

### Hierarchy
- **Display** (800, `clamp(2.4rem, 5vw, 5.3rem)`, line-height 0.9): The masthead title "Fantasy War Room," uppercase, negative letter-spacing (`-0.035em`).
- **Headline** (800, browser default `h2` size, `-0.035em` letter-spacing): Section headers — "Draft board," "Weekly decisions," "Model room" — each underlined with a 3px ink rule and pushed down by a 2.7rem top margin.
- **Body** (400, Aptos): Paragraphs, form labels, buttons, inputs, and dataframe cell text.
- **Label** (700–900, uppercase, `0.04–0.12em` letter-spacing, Bahnschrift Condensed): The evidence-strip field labels, the artifact-mode stamp text, and the workspace-tab captions.
- **Handwritten mark** (700, Segoe Print, tilted -1deg): The evidence-strip values (sheet name, row count, positions, mode) and the ink-colored data labels drawn directly onto the draft-board bar chart.

### Named Rules
**The Handwritten Mark Rule.** Segoe Print (or its Bradley Hand fallback) appears only on data values and call-outs — the evidence strip's figures and chart data labels — never on structural labels, headings, or body copy.

## Layout

The container is capped at 1480px with 2.25rem of top padding and 4rem of bottom padding, sitting on a warm-paper canvas ruled with a repeating 28px horizontal line pattern to read as ledger paper. The first viewport stacks, in order: a ledger masthead (10px top / 3px bottom ink border, uppercase display title, rotated stamp), a clipped-corner evidence strip reporting sheet/row/position/mode, a three-cell equal-width workspace selector, then the active workspace's full-width ranking sheet.

A permanent ink-navy sidebar (bordered `1px solid #78849a`) holds the "Model sheets" loader and "Board controls" filters, kept structurally and chromatically separate from the warm-paper workspace canvas.

At the 760px breakpoint, the container padding tightens to `1rem .75rem 3rem`, the stamp drops from an absolutely-positioned corner badge to an inline block below the masthead copy, the workspace tabs shrink their type/padding and center their labels, and the evidence strip reflows from a wrapping flex row into a fixed 2-column grid — but the workspace selector stays exactly three equal cells in one row at every width. The QB-only limitation is disclosed through a persistent `st.expander`, tucked out of the way but never hidden, at any viewport.

### Named Rules
**The Equal Three Rule.** The three workspace tabs (Draft Board / Weekly Decisions / Model Room) always render as one equal-width row. They never stack, wrap, or collapse into a dropdown, even at mobile widths — only their type size and padding shrink.

## Elevation & Depth

This system uses no blurred shadows anywhere — every `box-shadow` value has a zero blur radius. Depth reads instead as a stack of flat, solid-color offsets, like a rubber stamp pressed onto paper or a card physically laid on top of another, plus flat color-blocking (the ink-navy sidebar against the warm-paper canvas) for structural separation.

### Shadow Vocabulary
- **Ledger stack** (`box-shadow: 6px 6px 0 rgba(23, 36, 59, .16)` with a `2px solid` ink border): Applied to every `st.dataframe` table, giving ranking sheets a stacked-paper feel.
- **Stamped button** (`box-shadow: 3px 3px 0 var(--ink)` with a `2px solid` ink border): Resting state for primary and download buttons.
- **Pressed stamp** (`box-shadow: 2px 2px 0 var(--red)` with the border/text swapped to red and `transform: translate(1px, 1px)`): The button hover state — the shadow shrinks and the button visually presses down, like a stamp making contact.

### Named Rules
**The Flat Offset Rule.** Every shadow is a hard, non-blurred offset in a single flat color (`Npx Npx 0 color`). A soft or blurred `box-shadow` never appears in this system.

## Shapes

Border-radius is zero on every control — buttons, selects, text inputs, alerts, and the file uploader all have square corners; roundedness never softens a surface here. Corner character instead comes from `clip-path` notches: the evidence strip and the file-uploader button each clip two opposite corners (12px and 8px respectively) into a filed-ledger-tab silhouette. Hard rules — 2–4px solid ink borders — replace soft dividers throughout: the masthead's top/bottom borders, the underline beneath every `h2`, and the 4px baseline under the workspace tabs. The status stamp is a rotated (-2deg) bordered rectangle; handwritten marks tilt the opposite way (-1deg) for a hand-applied feel.

### Named Rules
**The Hard Corner Rule.** No `border-radius` is ever non-zero. When a corner needs character, it is clipped with `clip-path`, not rounded — and clipping is reserved for ledger/tab-like surfaces (the evidence strip, the uploader button), not applied everywhere.

## Components

### Buttons
- **Shape:** Square corners (`border-radius: 0`), `2px solid` ink border.
- **Primary:** Gold Highlight (`#d8a928`) background, ink text, uppercase, weight 800, with the Stamped Button shadow at rest.
- **Hover / Focus:** Hover swaps the border and text to Red Stamp and shrinks to the Pressed Stamp shadow with a 1px translate. `focus-visible` gets a distinct `3px solid` Blue Info outline with 3px offset — a color-independent cue separate from the red hover state.
- **Secondary (file uploader):** Same gold fill, but with clipped corners (8px notch) and a cream (`#fff7e7`) border instead of ink, marking it as an upload action rather than a form-submit action.

### Cards / Containers
- **Corner Style:** Square by default; the evidence strip is the one container clipped into a notched ledger-tab shape.
- **Background:** Dataframes and the evidence strip sit on Paper Deep (`#e8dec7`); alerts use the lighter Alert Paper (`#faf4e5`).
- **Shadow Strategy:** Dataframes use the Ledger Stack shadow (see Elevation).
- **Border:** `2px solid` ink on dataframes and alerts; a `1px solid` Rule Line hairline under the evidence strip.
- **Internal Padding:** Evidence strip uses `.65rem .8rem .8rem`.

### Inputs / Fields
- **Style:** Square corners on all selects and text inputs. Inside the sidebar, fields fill with Sidebar Field (`#233552`) and a `#78849a` border to read against the ink-navy panel.
- **Focus:** Buttons carry an explicit blue `focus-visible` outline; other controls use BaseWeb/Streamlit's default focus treatment.

### Navigation (Workspace Tabs)
- **Style:** Three equal joined tabs on a `4px solid` ink baseline, uppercase Bahnschrift Condensed labels. The unselected state sits on Paper Deep with an ink border; the selected tab inverts to a solid ink fill with cream (`#fff7e7`) text.
- **Mobile:** The exact same three-cell row is kept — only label size and padding shrink, and labels center.

### Sidebar (signature component)
A permanent dark ink-navy filing panel (background Ink Navy, all text Sidebar Cream) holding the "Model sheets" loader and "Board controls" filters — chromatically inverted from the warm-paper workspace so the control desk always reads as distinct from the ledger canvas.

### Masthead + Stamp (signature component)
A ledger masthead (10px top / 3px bottom ink rule) carries the uppercase display title and, in the top-right corner, a rotated (-2deg) `3px solid` red-bordered stamp reporting the artifact mode ("Historical backtest" / "Forward projection") — the literal stamp that anchors the war-room metaphor.

### Evidence Strip (signature component)
A clipped-corner metadata ledger row beneath the masthead, pairing uppercase Bahnschrift Condensed labels with tilted (-1deg) Segoe Print handwritten values (sheet name, row count, positions, mode). The same handwritten-mark device reappears in the Altair charts as the ink-colored projected-point labels on the draft board's bars.

## Do's and Don'ts

### Do:
- **Do** reserve Segoe Print (or its cursive fallback) strictly for data values and call-outs — the evidence strip's figures and chart data labels — never for structural labels or body copy.
- **Do** keep `border-radius: 0` on every control; use `clip-path` notches, not rounding, when a corner needs character.
- **Do** use hard, non-blurred offset shadows (`Npx Npx 0 color`) for elevation; never a blurred `box-shadow`.
- **Do** keep the three workspace tabs in one equal-width row at every viewport width.
- **Do** reserve Gold Highlight exclusively for primary actionable buttons and the uploader control.

### Don't:
- **Don't** use Red Stamp for body text or long-form content; it is reserved for the stamp, handwritten actual-value marks, and hover/error states.
- **Don't** add soft or blurred shadows/glows anywhere; depth comes from flat, hard-edged offsets only.
- **Don't** stack, wrap, or collapse the workspace tabs into a dropdown on mobile; shrink type and padding instead.
- **Don't** introduce a second display face — Bahnschrift Condensed is the only heading/label face across headings, the stamp, and the evidence-strip labels.
