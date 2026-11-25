# Forecast each player at his own kickoff

Christian McCaffrey's 2022 Week 7 Carolina forecast is **not** a miss against
San Francisco's 6.2 points, and it is **not** a Carolina zero. The CAR forecast
was reasonable on Thursday. By Sunday the trade was public, and the model
could forecast the actual SF–KC matchup. The same timing correction led to a
different decision for Carlos Hyde: **no forecast to score**.

| Player and week | Common weekly freeze (control) | Just before his team's kickoff (audited) |
|---|---|---|
| McCaffrey, 2022 W7 | CAR vs TB: 13.90 projected; old-team target withheld. | SF vs KC: 13.49 projected, 6.20 actual, 7.29 points over. The [trade](https://www.panthers.com/news/panthers-trade-christian-mccaffrey-to-49ers) admitted SF before its 20:25 UTC kickoff; the later box was used only to score it. |
| Hyde, 2018 W7 | CLE at TB: 9.94 projected; old-team target withheld. | The [Friday trade](https://www.clevelandbrowns.com/news/browns-trade-rb-carlos-hyde-to-jacksonville) admitted JAX. The Jaguars [announced their inactives](https://www.houstontexans.com/news/texans-inactives-vs-jaguars) at 15:31 UTC, before the 17:00 kickoff; [Hyde was on the list](https://www.nbcsports.com/nfl/profootballtalk/rumor-mill/news/carlos-hyde-not-playing-for-jaguars). He had zero for JAX, but was not selected. Neither a CLE zero nor a JAX forecast enters the error metric. |

[Control trade-week rows](weekly/trade-weeks.csv) · [Final trade-week rows](kickoff-audited/trade-weeks.csv)

## What the clocks change

The control freezes *everyone* 60 minutes before the first game of the week,
often on Thursday. The kickoff arms use information strictly before **each
game's scheduled kickoff**, including Sunday or Monday inactives. One game
cannot revise an already-issued forecast for another. All arms hold out each
season from 2014 through 2025, use the same position roles and model families,
and give RBs the published lagged carry-share features. They consume no
current-week performance as a feature. This is a forecast study: ranking the
whole week's *final* forecasts is **not** a playable locked lineup. No starter
picks or regret metrics are computed from that ranking. To measure start/sit
decisions, a separate replay must compare **all still-available alternatives**
using what was known at each kickoff, then lock only the players whose games
start. Sunday news cannot change a Thursday start or bench decision.

| Arm | Forecasts issued | Matchups scored | Correct team roles / scored | Withheld |
|---|---:|---:|---:|---:|
| [Weekly control](weekly/scores.csv) | 44,110 | 43,983 | 35,635 / 43,983 | 127 |
| [Kickoff, same source set](kickoff/scores.csv) | 44,109 | 43,978 | 36,799 / 43,978 | 131 |
| [Kickoff, audited](kickoff-audited/scores.csv) | 44,109 | 44,081 | 36,866 / 44,081 | 28 |

Ensemble errors on each arm's own verified forecasts (non-PPR points, **MAE / RMSE**):

| Position | Weekly control | Kickoff, same sources | Kickoff, audited |
|---|---:|---:|---:|
| QB | 6.711 / 8.351 | 6.613 / 8.215 | 6.617 / 8.221 |
| RB | 4.932 / 6.524 | 4.885 / 6.455 | 4.873 / 6.443 |
| WR | 4.317 / 5.660 | 4.314 / 5.625 | 4.308 / 5.620 |
| TE | 3.655 / 4.793 | 3.672 / 4.781 | 3.674 / 4.783 |

The clearer benefit was **choosing whom to forecast**. The final kickoff arm
has 1,853 picks not on the weekly board and drops 1,854 weekly picks. On the
42,245 verified game-team-player rows common to both, weekly-to-final MAE was
6.632 to 6.630 (QB), 4.907 to 4.933 (RB), 4.325 to 4.361 (WR), and 3.656 to
3.715 (TE). Improved coverage and role selection do not establish more accurate
individual forecasts on an unchanged population.

Follow-up on the identical audited matchups: [does separately predicting
completions, catches and yards per catch improve the point forecast?](catch-stages/README.md)
That experiment identified attempts and targets as the largest contributors to
yardage error. The [team passing volume and player share experiment](pass-workload/README.md)
then tested whether forecasting those counts differently improves fantasy points.
The [teammate availability experiment](target-inheritance/README.md) tests the
next question: can pregame reports tell us which pass catchers get extra targets?

## Which games count

An exact game, team and player box is scored as recorded. Without a box, a
positive mapped snap for that side, a game-specific inactive announcement, or a
unique matching weekly roster can establish the matchup. With complete team
boxes and no scoring event, its fantasy score is zero. **Affiliation does not
imply playing a snap.** An unknown participation label stays unknown. Raw
snap/box conflicts, a known post-freeze departure, another matchup, or no
corroboration never produce a zero by default. Such rows keep their forecasts
in the archive but have null targets and are excluded from fitting, calibration
and scoring. Stronger evidence that he actually played for the predicted side
takes priority. Actual boxes can contribute to the player's *later* history,
never to a forecast for another matchup that week.

In the [final evidence counts](kickoff-audited/evidence.csv), 40,328 forecasts
have matching boxes, 939 more have matching snaps and 2,814 more have matching
rosters. The scored set includes **5,422 zeros and 402 negative scores**.
None had an observed other-team matchup. The [28 unverified picks](kickoff-audited/unscored.csv)
are all in-season; 14 are from 2014–16. Their outcomes are withheld, not
assumed to be missed games or injuries.

The [111-game eligibility audit](../opportunity/transactions/historical-availability.md)
found that 96 exclusions were knowable before even the weekly freeze; the
other 15 became usable by the player's kickoff.
The [case ledger](../../../src/ffpred/study/historical_availability.csv)
is applied only on the audited kickoff arm, before team-role ranking. Seven of
the 111 old control selections had already disappeared just by moving the
deadline. The remaining 104 disappear with the ledger; **none of the 111**
is a target for the wrong matchup. This is retrospective repair, not a complete
historical news feed or an independent test of the audit's benefit.

Five additional pilot cases are **separate** from the 111. The two strict
clocks share the [game-specific news](../../../src/ffpred/study/kickoff_news.csv):
Tim Hightower (NO 2015 W1) and DuJuan Harris (SF 2016 W1) had been released
by kickoff; Chase Claypool (MIA 2023 W5) had been declared inactive. On the
audited arm, [two pregame waiver claims](../../../src/ffpred/study/kickoff_claims.csv)
also disqualify Houston's Jaelen Strong (2017 W3) and Oakland's Trevor Davis
(2019 W14). Neither made his new club's projected roles. Relative to the
same-source kickoff arm, the 104 ledger removals plus two wrong-club removals
change 106 selections. Of their replacements, 86 have a matching box, six a
matching snap, 11 a matching roster, and three remain unverified.

## Reproduce

```console
uv run --locked ffpred weekly-kickoff --arm weekly --end-year 2025
uv run --locked ffpred weekly-kickoff --arm kickoff --end-year 2025
uv run --locked ffpred weekly-kickoff --arm kickoff-audited --end-year 2025
```

The command writes separate `artifacts/weekly-kickoff/<arm>/` directories with
the full forecast Parquet (including unscored forecasts), annual and pooled
scores, interval coverage, role counts, trade cases, and hashed inputs and
outputs. Source CSV hashes normalize line endings to LF so Windows checkouts
can verify the same evidence. [Final manifest](kickoff-audited/manifest.json) ·
[Annual errors](kickoff-audited/annual.csv) · [Interval coverage](kickoff-audited/intervals.csv).
See the [earlier weekly-lock experiment](../opportunity/transactions/README.md)
for the independent control's original roster audit. Its historical scores
used player-week, not strict matchup targets.
