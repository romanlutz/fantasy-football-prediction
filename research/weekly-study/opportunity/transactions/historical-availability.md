# Why 111 selected players had no record in the forecast matchup

All **111 player-games** had evidence, available before the player's scheduled
kickoff, that he was not eligible for the forecast team's active lineup. That
is a selection error to repair, not 111 poor fantasy performances. The cohort
contains 97 distinct players: 39 selections in 2014, 54 in 2015 and 18 in 2016.
It excludes openers. The 11 later-season cases among the original 122 are not
part of this count.

| Pregame reason | 2014 | 2015 | 2016 | Total |
|---|---:|---:|---:|---:|
| Injured reserve, including IR designated for return | 28 | 44 | 0 | 72 |
| Waived or released by the forecast team | 7 | 3 | 11 | 21 |
| Suspended | 1 | 4 | 2 | 7 |
| Non-football injury or illness reserve | 1 | 1 | 1 | 3 |
| Practice squad rather than active roster | 0 | 2 | 1 | 3 |
| Retired at that date | 0 | 0 | 2 | 2 |
| Commissioner exempt list | 1 | 0 | 0 | 1 |
| Traded away | 1 | 0 | 0 | 1 |
| Reserve/did not report | 0 | 0 | 1 | 1 |
| **Total** | **39** | **54** | **18** | **111** |

The [case ledger](../../../../src/ffpred/study/historical_availability.csv)
gives the exact game, forecast team, GSIS player ID, reason, primary source
and date, conservative UTC availability time, and corroboration where available.
Its keys match the [original audit](unresolved.csv) filtered to `board=transactions`,
`opener=false`, `evidence=no_record`, and `season<=2016`. A positive transaction
or announcement, **not** absence from a box, snap file, or roster, establishes
the reason. No row supplies an independently corroborated game-day inactive
zero. Reserve status is not an injury exit in the game being forecast.

**What changed with the clock?** On the recorded availability times, 96 of the
111 exclusions were already knowable at the historical common cutoff. The other
15 became usable between that cutoff and the player's game. Moving the deadline
alone is therefore not the whole fix: most of the missing information was a
dated status change the board did not ingest. These counts describe the frozen
cohort, not the number of selections that change in a full replay, where one
replacement can affect other slots.

**Dating and returns.** For date-only news we wait until midnight Eastern *after*
the supported date. A club's retrospective transaction table can label a move a
day early; the source date is preserved, but a later corroborated date controls
availability. For example, the [Vikings' dated announcement](https://www.vikings.com/news/mckinnon-placed-on-injured-reserve-14514024)
places Jerick McKinnon on IR Saturday, not Friday as the annual table suggests.
The [Colts' Vick Ballard](https://www.colts.com/team/transactions/2015) was on IR at
Monday's game; the subsequent release is not the reason for that forecast.
Corroborating retrospective histories can verify the sequence, but later news
is not an input before publication.

Two cases illustrate why the exclusions are bounded to a **game, team and
player**, not a player or season:

- Tony Romo went on [short-term IR in September](https://www.espn.com/nfl/story/_/id/13717602/tony-romo-dallas-cowboys-placed-short-term-injured-reserve),
  played again in November, and went on [IR again in December](https://www.espn.com/nfl/story/_/id/14436303/dallas-cowboys-put-qb-tony-romo-injured-reserve).
  The two missed matchups are excluded, not the return games in between.
- Percy Harvin was traded from Seattle to the Jets in 2014 **after New York
  had already played that week**. There is no replacement Jets matchup to
  score. His 2015 Buffalo IR stint and 2016 retirement are separate facts; the
  latter does not exclude his subsequent comeback.

Apply the exclusions **before** ranking replacements on the kickoff board. Do
not edit the historical picks, transfer fantasy points between teams, or fill
unverified outcomes with zero. This is a targeted retrospective repair, not a
complete historical transaction feed or an independent test of forecasting skill.
