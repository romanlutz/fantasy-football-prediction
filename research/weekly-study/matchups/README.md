# Expected matchup points, player uncertainty and starter choices

**Use the ensemble for expected points and the component model to explain the miss.**
Across 44,108 selected player-weeks and 209 regular-season weeks in 2014-2025,
the equal-weight ensemble had the lowest RMSE at all four positions. Forecasting
stats alone did not consistently beat forecasting points directly. It did
explain why RB forecasts miss by nearly five points: roughly two points from
rushing yards, another two from rushing TDs, and most of the rest from receiving.

[Example: every forecast and starter pick for 2025 Week 10](example-week.csv) ·
[Pooled scores](scores.csv) · [Season scores](annual.csv) · [Opener and prior-absence scores](cohorts.csv)

## The point forecast

Every model predicts the same eligible players, including zeros and negatives.
The table is **MAE / RMSE**, in non-PPR points. RMSE is the main comparison for
expected points; it rewards getting the conditional mean right.

| Model | QB | RB | WR | TE |
|---|---:|---:|---:|---:|
| Trailing ten | 7.200 / 8.982 | 5.235 / 6.943 | 4.463 / 5.874 | 3.705 / 4.940 |
| Direct boosting, broad training pool | 6.748 / 8.450 | 4.952 / 6.591 | **4.312** / 5.694 | 3.674 / 4.873 |
| Direct boosting, projected-role training | 6.800 / 8.487 | 4.985 / 6.618 | 4.343 / 5.700 | 3.696 / 4.890 |
| Direct ridge, projected-role training | 6.760 / 8.401 | 4.937 / 6.559 | 4.342 / 5.696 | 3.619 / 4.790 |
| Conditional stats, linear | 6.724 / 8.418 | 4.942 / 6.568 | 4.323 / 5.704 | **3.617** / 4.800 |
| Conditional stats, forest | 6.717 / 8.379 | 4.967 / 6.581 | 4.330 / 5.676 | 3.637 / 4.796 |
| Equal-weight ensemble | **6.701 / 8.352** | **4.929 / 6.541** | 4.314 / **5.665** | 3.620 / **4.781** |

Direct ridge, direct squared-error boosting, and both conditional-stat models
make up the ensemble. The weights were fixed at one quarter each, not learned
from the test seasons. Its RMSE beat role-trained direct boosting in 11/12 QB
seasons and 12/12 at RB, WR and TE. The differences are small; this is not
proof that stacking or optimizing component-specific weights would generalize.
The controlled **training population** comparison favors keeping the broad
pool: for the same boosting configuration and forecast rows, excluding all
historically nonselected players made both MAE and RMSE worse at every
position. These are *pregame projected* roles, not realized starters.

## Where the errors come from

We first predict the probability of **playing at least one snap**, including
special teams, not merely dressing or recording a touch. A logistic model uses
only pre-lock features. Among those who played, we predict attempts, carries
and targets, then yards and TDs per opportunity for each channel. For example,
expected rushing yards are play probability times expected carries if playing
times a predicted yards-per-carry rate, plus a small zero-carry yardage term
when the data warrant one. The rate is fitted **with opportunities as weights**:
a one-carry game does not set the rate as strongly as a 20-carry game. The
estimand is expected total yards divided by expected volume given the inputs
and participation, not the product of arbitrary independent averages.

Expected interceptions, lost fumbles and remaining scoring (return/recovery TDs
and conversions) are estimated given participation. Add the expected stats
with the league scoring weights. A prediction of 0.4 rushing TD is an
expected count, not a claim that a fractional TD will occur. No same-game
snap count, attempts, carries or injury exit enters a forecast. The full
per-player outputs retain conditional volume, unconditional volume, efficiency
and component means. Linear fits use ridge; the alternative is a 24-tree
extremely randomized forest with minimum leaf size 20 and depth cap 10.

For the **forest component model**, the following are contributions to its
held-out MAE, **in fantasy points**, not raw stat errors:

| Position | Total MAE | Largest contributions |
|---|---:|---|
| QB | 6.717 | Passing TDs 3.075; passing yards 2.284; rushing TDs 0.603; rushing yards 0.431 |
| RB | 4.967 | Rushing yards 1.969; rushing TDs 1.811; receiving yards 0.680; receiving TDs 0.478 |

Of the RB rushing-yard contribution, **1.354 points came from the carry
forecast and 0.615 from remaining yardage at that workload**. The model
missed by 4.59 carries and 24.91 rushing yards on average. When a back had
at least one carry, its rate missed by 1.64 yards/carry. At QB the analogous
passing-yard split was 1.397 points from attempts and 0.886 from yield. This
points toward better workload and goal-line/red-zone share forecasts, not
just a more elaborate yards-per-carry predictor.

**How the attribution adds up:** multiply each component's signed point error
by the sign of the player's *total* point error, then average. These contributions
sum to the model's MAE; they can be negative where one mistake offsets another.
The yardage split prices the volume error at the predicted rate; the remainder
is yield (including rare yards without a recorded opportunity). This is an
accounting identity, not a causal claim. The table doesn't retroactively
explain the earlier study's 6.6/4.8 numbers. For the direct booster, the
[component ledger](components.csv) offers an explicit bridge: add the
direct-minus-component forecast difference in the direction of the *direct*
model's error. This reconciles to direct MAE but does not expose internal
components in a model trained only on points.

The models differ by statistic as well as position: linear conditional
models had lower RB carry RMSE (5.877 vs. 5.910) and rushing-yard RMSE
(33.029 vs. 33.178); the forest had slightly lower rushing-TD RMSE
(0.553 vs. 0.554). These differences are too small to select a specialist
for each target from these same holdouts. See [all stat errors](components.csv)
and [workload vs. yield](yardage.csv).

## How uncertain is one player's score?

Publish **50% and 80% prediction intervals** for the individual game's points.
A confidence interval for the average forecast would answer the wrong question.
Four quantile boosters get pregame features including the player's own trailing
ten-observation variability, workload, role and availability evidence. For test
year Y, they fit on projected roles through Y-2; observed errors from Y-1
calibrate each band with a finite-sample rank. Bands can expand, never shrink,
and the outer band must contain the inner band. No test-year outcome is used
to set a test-year band. Drift, discrete scores and finite samples mean these
are empirical coverage levels, not individual guarantees.

| Position | 50% band: observed coverage | 80% band: observed coverage | Mean width of 80% band |
|---|---:|---:|---:|
| QB | 50.2% | 79.8% | 21.17 points |
| RB | 51.4% | 84.5% | 15.04 |
| WR | 51.6% | 85.8% | 13.16 |
| TE | 50.7% | 86.6% | 10.86 |

The bands pick up meaningful differences in risk: the narrowest quarter of RB
bands averaged **9.71** points wide and those forecasts missed by **3.28**;
the widest quarter averaged **20.37** wide and missed by **6.39**. That does
not mean a historically steady player never has an outlier. In the example
week, TreVeyon Henderson's forecast was **7.2** with an 80% band of **1.4-15.5**;
he scored **27.0**. The skill-position outer bands are conservative, especially
near zero. Report coverage alongside width and endpoint pinball loss, not
coverage alone. [Interval scorecard](intervals.csv) ·
[Probability of playing](participation.csv) · [Probability calibration by bin](participation-calibration.csv)

## Did we start the right players?

One hour before the first kickoff of each NFL week, rank all projected roles
by expected points and take **12 QBs, 24 RBs, 24 WRs and 12 TEs**. No flex,
individual fantasy rosters, waiver restrictions, or later-week substitutions.
The full-league hindsight comparison includes breakout players **not** on the
forecast board. Position comes from the pregame role for selected players,
and from game-roster/box labels for other recorded producers. The restricted
comparison asks what perfect ranking could do with just our selected roles.
Ties at the true cutoff get credit up to the available tied slots; ties in
forecasts use stable player ID.

| Position | Actual top starters picked, per week | Points lost vs. full-league optimum | Points lost vs. optimum inside selected roles |
|---|---:|---:|---:|
| QB | 6.05 / 12 | 69.42 | 64.07 |
| RB | 13.33 / 24 | 103.38 | 86.81 |
| WR | 9.90 / 24 | 136.07 | 123.91 |
| TE | 4.91 / 12 | 51.99 | 35.55 |

The ensemble identified **34.20 of 72** starters and scored **785.69 points**
per week across the 12-team league. Perfect hindsight scored **1,146.56**.
Of the **360.87** points left on the table, **50.52** were unobtainable from
the selected roles even with perfect ranking. The trailing-ten baseline lost
**389.50**; the ensemble recovered **28.63 points per league-week**, or about
2.39 per team if spread evenly. This is a league-wide decision-quality
benchmark, **not** the score a particular manager could achieve by rearranging
a fixed roster. [Weekly picks/hits/regret counts](starters-weekly.csv) ·
[All models vs. both optima](starters.csv)

## Eligibility, openings and remaining limits

The frozen board has one team and position per player-week. Dated NFL trades
are matched by PFR/GSIS identity, not player name. Date-only trades become
eligible at midnight US Eastern the next day. A traded draft pick sometimes
carries the ID of the eventual draftee: that is **not** a player transfer.
We remove superseded affiliations and resolve conflicts using the available
evidence before filling slots. None of the five previously double-selected
trade weeks is counted twice. McCaffrey's 2022 trade date was Thursday; the
week locked before we could use that date-only record, so the forecast
still says Carolina. Changing it to San Francisco after seeing the box would
be hindsight. [Removed affiliations](affiliations.csv)

All **384 team-openers** are included. We filled **2,687 of 2,688** opening
slots using last season's final regular-season roster (or individual prior
box for missing roster codes), the draft, dated player transfers and whatever
eligible depth/practice/news evidence existed by the common lock. No preseason
statistics or target-week un-timestamped roster is used. The history features
span seasons; rookies receive train-only imputation plus projected-role
features rather than invented last-game production. In all, the model covers
**6,302 team-games**, filling 44,108 of 44,114 slots.

**Openers are forecastable; opening rosters are not fully reconstructed.**
Correct projected QBs: **281/384 (73.2%)** on openers vs. 87.7% later.
Opening precision was 56.9% at RB, 60.1% at WR and 55.7% at TE. There were
**272 opening picks** whose box was for a different team, versus **10** later.
We retain the player's actual fantasy points for the week in those cases;
the projected *matchup* was wrong. The `matchup_target` rows in the
[cohort scorecard](cohorts.csv) give no credit for points scored for another
team: with the same forecasts, opening QB MAE rises from 7.69 to 7.88 and
opening RB MAE from 4.79 to 4.96. A carried-over player who never played
has no box team to compare. Complete dated free-agent signings, cuts, PUP/IR
changes and final opening depth charts remain the most important data gap.
The frozen previous benchmark used individual kickoff deadlines and omitted
openers, so its MAEs are not direct measures of progress against this study.
[Selection by position](selection.csv)

Snap coverage starts in **2013** in the source actually retrieved. A
positive offensive, defensive or special-teams snap, or a stat event, proves
participation. Absence of such evidence is labeled nonparticipation only with
a covered team-game and a usable player ID mapping; otherwise the label is
unknown, not automatically zero. All held-out selections had known labels.
Selection, forecasting and availability probabilities do not identify in-game
injury exits. Those games remain in the manager-facing scorecard. Earlier
weekly injury/depth records and the two-item announcement ledger are not a
comprehensive history of reports at the decision time; 2025's undated injury
release does not enter current-week features.

## Reproduce

```console
uv run --locked ffpred weekly-matchups --end-year 2025
```

Season 2009 warms up the player and opponent histories. Point and participation
models train from 2013 through the previous season, separately by position;
quantile training can start in 2010 because it requires points, not snap labels.
All inputs in a test week use either strictly earlier games or evidence dated by
the common lock. Earlier weeks in the held-out year update histories, not model
weights. All scoring uses the corrected non-PPR preset. No 2026 games were used.
The [manifest](manifest.json) records input/output hashes. The local output
directory `artifacts/weekly-matchups` contains the full board, predictions and
independent scoring universe as Parquet alongside all scorecards.
