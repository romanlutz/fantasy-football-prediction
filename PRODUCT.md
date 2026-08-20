# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Fantasy-football players preparing for a draft and making weekly roster decisions.
They need player projections that are useful at two distinct time horizons: total
season value for drafting and game-level outlooks for weekly decisions.

## Product Purpose

Turn the project's pulled NFL data and model predictions into an explorable
decision-support interface. Success means users can compare players for a draft,
then switch to a separate weekly workflow without confusing season-long and
single-week signals.

## Positioning

The product exposes reproducible, leakage-safe model outputs alongside actual
results and evaluation context, rather than presenting unexplained rankings.

## Operating Context

Users review a season-long draft board before and during a fantasy draft. During
the season, they inspect a specific week, compare players, and use filters to
narrow the field for roster decisions.

## Capabilities and Constraints

- Treat season projection and weekly prediction as separate workflows.
- Support filtering by position and other useful player, team, season, week, and
  performance attributes when those fields are present in an artifact.
- Work with the project's Parquet datasets and prediction artifacts.
- Preserve point-in-time provenance: a target season's features may use only
  completed game statistics through the recorded prior-season cutoff.
- Use preseason schedule and QB1 depth-chart assignments for upcoming seasons,
  and keep historical outcomes separate from the features used to predict them.
- The current prediction pipeline is quarterback-focused. The interface must
  communicate unavailable positions honestly while remaining compatible with
  future multi-position artifacts.
- Model output is decision support, not a guarantee of fantasy performance.

## Evidence on Hand

- Reproducible train and test Parquet datasets with provenance manifests.
- SVR and MLP per-game prediction artifacts.
- Actual fantasy-point targets and shared regression metrics.
- No testimonials, commercial claims, or guaranteed-performance evidence.

## Product Principles

- Separate draft strategy from weekly lineup decisions.
- Show the evidence behind every ranking.
- Make comparison faster than spreadsheet work.
- Preserve uncertainty and model limitations instead of hiding them.
- Keep artifact loading local and reproducible.

## Accessibility & Inclusion

Use semantic controls, keyboard-accessible interactions, visible focus states,
and color-independent status cues.
