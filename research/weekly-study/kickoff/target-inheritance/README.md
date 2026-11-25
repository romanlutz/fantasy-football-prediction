# Who gets the targets when a regular pass catcher is out or limited?

**Keep the [team-volume and player-share forecast](../pass-workload/README.md).
Explicitly reallocating targets using teammate reports made target MAE worse
at RB, WR and TE, including in games where a regular teammate was ruled out.**
The experiment covers the identical 44,109 issued forecasts from the 2014-2025
regular seasons. It scores the same 44,081 verified player-games, including
5,422 zeros and 402 negatives, and withholds the same 28 unknown outcomes.
QBs, team passing volume, efficiency, scoring, and direct models did not change.

| Position | Scored | All: target MAE, control / new | Out peer: target MAE, control / new | Limited-only peers: target MAE, control / new | All: non-PPR ensemble MAE, control / new |
|---|---:|---:|---:|---:|---:|
| RB | 12,588 | 1.614 / 1.622 | 1.640 / 1.677 | 1.571 / 1.589 | 4.8670 / 4.8685 |
| WR | 18,894 | 2.464 / 2.473 | 2.434 / 2.495 | 2.435 / 2.449 | 4.2876 / 4.2921 |
| TE | 6,298 | 2.248 / 2.253 | 2.205 / 2.210 | 2.255 / 2.283 | 3.6688 / 3.6695 |

For WRs with an out peer, target MAE increased by **0.061** (95% paired
season-bootstrap interval +0.016 to +0.136); for RBs it increased by **0.037**
(+0.010 to +0.077). For TEs the interval included no change. Overall point
MAE worsened by less than 0.005 at every position, while overall point RMSE
fell slightly at each receiving position. The two metrics answer different
questions: there is no broad improvement in typical absolute error. Intervals
use 4,000 paired season resamples, not independent player-games. The overall
comparison includes all 12 seasons; affected-player comparisons include 11.
[Scores and RMSE](scores.csv) · [Annual scores](annual.csv) · [Paired differences](paired.csv)

## Who actually got more than the control expected?

When a **regular WR was ruled out**, the clearest pattern was at the
second-ranked WR. The baseline forecast **4.73** targets, the adjustment
forecast **5.18**, and the observed mean was **5.08**, across 469 player-games.
The same adjustment gave too much to the third-ranked WR and first-ranked RB:

| Pregame recipient role | Player-games | Control targets | Adjusted targets | Actual targets |
|---|---:|---:|---:|---:|
| WR1 | 469 | 7.19 | 7.40 | 7.29 |
| WR2 | 469 | 4.73 | 5.18 | 5.08 |
| WR3 | 467 | 3.13 | 3.62 | 3.25 |
| RB1 | 469 | 3.27 | 3.50 | 3.28 |
| TE1 | 469 | 4.95 | 4.91 | 4.97 |

These are ranks **after** pregame reports, not a claim that the preseason WR2
received the missing player's targets. The comparison is against a control
that already knows which roles were selected, not against the same matchup
with the missing player healthy. The group means do not identify a causal
transfer or imply more accurate individual predictions. [Every recipient
rank by reported teammate position and status](recipients.csv); rows can
appear in more than one group if multiple types of teammates were flagged.

**Could we pick the teammate with the most targets?** Among selected RB/WR/TEs
not themselves flagged regulars, with at least two choices, all choices
verified and at least one actual target, the baseline got **382 of 711** out
team-games right; the adjustment got **377**. For limited-only team-games the
counts were **713 vs. 721 of 1,197**. An actual tie for most targets counts as
correct; forecast ties break by player ID. This tests target leaders, not
start/sit decisions or who would have seen more targets if everyone were
healthy. [Picks, changes and targets missed](target-leaders.csv)

## What was knowable before kickoff?

- A **regular** is an RB, WR or TE in the audited candidate pool, selected or
  not, with at least 10% of that club's official targets across its previous
  five regular-season games and a target in at least two of them. Byes do
  not count as games; prior-season games can inform openers. Work for a
  different club does not create targets to redistribute on this club.
- Use only a timestamped `Out` designation or inactive announcement for the
  out group. Keep a workload-cap announcement or a limited-practice report
  separate from an absence. An undated report, a later report, and a
  postgame zero are not pregame evidence. In the actual test sample there
  were **767** qualifying out reports and **1,673** limited-practice reports;
  none of the qualifying cases was an announced game workload cap. Limited
  practice therefore must not be read as a promise of fewer game snaps.
- Of **6,302 team-games**, **717** had an out regular and **1,200** had only
  limited regulars. At the affected *recipient* level this yielded 4,295
  scored RB/WR/TE games with an out **teammate**, and 6,276 with only limited
  teammates. A player is never his own missing teammate. Not every injury,
  IR move, late scratch or new signing is covered by these sources and the
  recent-usage rule. **No qualifying timestamped report was available in
  2025**; forecasts for all 544 team-games that season stayed unchanged.
  No report is not evidence of full health. [Coverage by season](coverage.csv)

## How the adjustment was tested

Fit the same per-position, opportunity-weighted target-share ridge on verified
players who played in prior seasons. Add the combined past target shares of
*other* regulars reported out and limited, separately for RB/WR/TE. Interact
each of the six inputs with the recipient's recent target share and whether
he ranks first at his position. The model learns these associations rather
than assuming every target from an unavailable WR goes to a WR.

Use the refit only for selected pass catchers with a qualifying teammate.
Preserve everyone else's allocation, multiply conditional shares by the same
play probabilities, and cap the affected group's total at the remaining team
budget. If they need less than the budget, the rest stays available to other
players. Change only expected targets and the receiving yards/TDs priced from
them, not player selection, RB carries, attempts, or prediction intervals.
The 107 control columns reproduce the published pass-workload forecast
Parquet hash exactly; the team budget Parquet is identical as well.

```console
uv run --locked ffpred weekly-kickoff --arm target-inheritance --end-year 2025
```

The [manifest](manifest.json) records sources and outputs. The full forecasts
and game-specific report timestamps and prior shares are written locally to
`artifacts/weekly-kickoff/target-inheritance/`. For a further test, prefer
pregame evidence of **which player already ran the missing receiver's routes**
to a blanket increase for players with the same position label.
