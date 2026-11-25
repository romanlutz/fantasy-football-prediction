# Does forecasting team pass volume and player shares improve fantasy points?

**Yes for WRs and RBs, but by hundredths of a fantasy point per player-game.**
On the same 44,081 verified non-PPR matchups from 2014 through 2025, the new
ensemble improved WR and RB mean absolute error (MAE) in 11 of 12 seasons
each. The QB and TE point estimates also improved, but their paired intervals
include no change. This is a comparison of opportunity forecasts, not a change
to who's selected, how points are scored, or what a catch is worth.

| Position | Scored | Original ensemble MAE | Team + share ensemble MAE | Change in MAE (95% interval) | Original / new RMSE |
|---|---:|---:|---:|---:|---:|
| QB | 6,301 | 6.6171 | 6.6049 | -0.0122 (-0.0259, +0.0031) | 8.2207 / 8.2116 |
| RB | 12,588 | 4.8733 | 4.8670 | -0.0063 (-0.0086, -0.0040) | 6.4428 / 6.4408 |
| WR | 18,894 | 4.3083 | 4.2876 | -0.0207 (-0.0305, -0.0121) | 5.6203 / 5.6101 |
| TE | 6,298 | 3.6742 | 3.6688 | -0.0054 (-0.0144, +0.0052) | 4.7827 / 4.7844 |

All 44,109 forecasts were issued before the player's kickoff. The 28
unverified outcomes remain withheld; the 5,422 verified zeros and 402
negative scores remain in the evaluation. The two direct points models are
unchanged. Each ensemble averages those two models with the corresponding
linear and forest component predictions. The reported intervals resample the
12 paired test seasons 4,000 times with player-game weighting. They describe
uncertainty in the **difference between models**, not the range of scores for
an individual player. The original seven models reproduce the published annual
and pooled scores. [All models](scores.csv) · [By season](annual.csv) ·
[Paired differences](paired.csv)

## Did the underlying workload get easier to predict?

Across 6,302 team-games, the team model missed attempts by **6.34** on
average, versus **6.57** for the offense's previous-eight-game mean.
For targets the comparison was **6.17 vs. 6.38**. Not every attempted pass has a
receiver target, so each has its own official team total. [Team results](team-volume.csv)

| Player workload | Old player linear MAE | Old player forest MAE | Team + share MAE | MAE if team total known | MAE if player's share known |
|---|---:|---:|---:|---:|---:|
| QB pass attempts | 8.75 | 8.72 | 8.66 | 5.15 | 5.53 |
| RB targets | 1.63 | 1.64 | 1.61 | 1.52 | 0.48 |
| WR targets | 2.48 | 2.50 | 2.46 | 2.24 | 1.01 |
| TE targets | 2.25 | 2.25 | 2.25 | 2.04 | 0.86 |

The last two columns are **hindsight diagnostics, not forecasts**. Substitute
the actual team total but keep the predicted individual share, or substitute
the player's actual share (including zero) but keep the predicted team total.
The errors do not add and these are not causal effects. For receivers, even
perfect knowledge of the team's passing volume would leave most of the miss;
knowing who gets the targets is much more informative. For QBs, both the
team's volume and the individual QB's share matter. Compared to forecasting
player counts directly, the workload improvement was distinguishable from zero
for QB attempts and RB/WR targets against both control models, not TE targets.
[Count and share errors](player-volume.csv) · [Paired count errors](volume-paired.csv) ·
[Hindsight diagnostics](volume-diagnostics.csv)

## How the comparison works

1. **Forecast the team's attempts and targets.** Train a standardized ridge
   model on earlier regular seasons only (history starts in 2009). Use the
   offense's previous three and eight games and the opponent defense's
   previous three and eight games of attempts, targets, carries and sacks.
   History carries across season openers; past-season training medians handle
   missing history. Predicted targets cannot exceed predicted attempts.
2. **Estimate individual shares if playing.** Train a separate ridge for each
   position and count on prior verified, selected players who actually played,
   weighting their share by that game's team total. Add the player's share
   over the last five observed appearances, appearance count and the sum of
   the other selected teammates' recent shares to the existing pregame role
   features (including the existing RB carry features). A bye or absence does
   not create an appearance; a positive mapped snap can establish a genuine
   zero-usage appearance. Historical shares can follow a traded player, but
   future workload comes from the destination team. A player without a history
   still gets a forecast from role features and training-only imputation.
3. **Allocate only the workload the selected players are expected to receive.**
   Multiply each conditional share by the existing probability of playing.
   Scale down proportionally only if the selected shares sum to more than
   100%. On average, 11.4% of attempts and 20.4% of targets remained for
   players outside the selected roles. The cap applied to 119 team-games for
   attempts and seven for targets. Multiply the allocated shares by the
   respective team forecasts, not by the realized totals.
4. **Change counts, not efficiency.** Replace the passing-attempt and target
   counts in the [original per-opportunity component models](../README.md).
   Keep their yards and TD rates, zero-opportunity production allowance,
   rushing, interceptions, fumbles, other scoring and ensemble weights fixed.
   The forest and linear point models use the **same** new workload forecasts
   but retain different efficiency estimates. In `player-volume.csv`, the
   control models' implied shares divide their predicted counts by the common
   forecast team total; all models are compared to actual shares in percentage
   points. The [catch-stage experiment](../catch-stages/README.md) evaluated
   changing efficiency stages separately.

**Implication for the paper.** Keep the original ensemble as a control and
report the workload variant for WR and RB. The gains are too small to claim
better lineup decisions without a playable start/sit replay. The best next
workload test is to measure who inherits targets when a regular receiver is
out or limited, using only reports available at the game's kickoff. This
experiment did not add route participation, forecast weather, or game-script
inputs, or test a change to individual prediction intervals.

```console
uv run --locked ffpred weekly-kickoff --arm pass-workload --end-year 2025
```

The [manifest](manifest.json) records data and output hashes. The command
writes the full player and team forecasts to
`artifacts/weekly-kickoff/pass-workload/{forecast,teams}.parquet`. Selecting
the original 80 player-forecast columns reproduces the
[audited control Parquet SHA-256](../kickoff-audited/manifest.json) exactly.
The raw `players` source-frame fingerprint differs from that manifest; the
play-by-play fingerprints and all matched control forecasts agree.
