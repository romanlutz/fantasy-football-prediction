# Which team could we have forecast at lineup lock?

**Follow-up:** [What changes when the player locks at his own kickoff?](../../kickoff/README.md)
The later study keeps this Thursday control, forecasts McCaffrey for SF,
rules Hyde out for JAX, and scores neither superseded old-team prediction.

Of **30 in-season picks** that disagreed with a game box, a positive snap or
a unique weekly roster, **28 were preventable**. Ten of those players still
made the role cutoff for their new team; 18 did not. Two trades were genuinely
too late. The 216 previously unverified opening picks tell a different story:
212 had the expected team's roster entry, one player was already suspended,
and the other three were released after the common lock.

## What the 216 openers mean

The previous board filled 2,688 opening slots. A same-team box corroborated
2,377 and a positive snap corroborated 95 more. Among the remaining 216,
**212** had a same-team weekly roster record: 156 `A01`, 46 `I01`, nine other
non-null codes and one missing code. A roster entry establishes neither a
snap nor an in-game injury. We did not delete these games or change their
fantasy points. [Roster codes and counts](roster-status.csv)

| No weekly roster, box or positive snap | What was knowable before the common lock | Decision |
|---|---|---|
| Ace Sanders, JAX 2014 | The [NFL had suspended him for the first four games](https://www.espn.com/nfl/story/_/id/11256064/nfl-suspends-jacksonville-jaguars-receiver-ace-sanders-4-games), despite the active listing in the opening roster snapshot. | Exclude for those four games, not for the season. Allen Hurns took the opening slot and scored 23.0 non-PPR points. |
| Louis Murphy, TB 2014 | Tampa [listed him as not practicing on September 3](https://www.buccaneers.com/news/buccaneer-quotes-september-3-13638236). The [release was September 6](https://www.buccaneers.com/news/murphy-returns-in-bucs-receiver-swap-13839461), after the September 4 lock. | Keep the affiliation; an injury concern was knowable, the cut was not. |
| Tim Hightower, NO 2015 | On the [September 6 club roster](https://www.neworleanssaints.com/news/new-orleans-saints-53-man-roster-by-position-as-of-sunday-september-6-15823347); [cut September 12](https://www.nbcsports.com/nfl/profootballtalk/rumor-mill/news/tim-hightowers-comeback-ends-for-now), after the September 10 lock. | Keep the locked pick. |
| DuJuan Harris, SF 2016 | [Signed September 5](https://www.49ers.com/news/49ers-sign-rb-dujuan-harris-place-cb-will-redmond-on-injured-reserve-17624602), then [cut September 12](https://www.49ers.com/news/49ers-re-sign-rb-harris-waive-wr-harper-17679881), after the September 8 lock. | Keep the locked pick. |

Correct opening roles rose from **2,080 to 2,081 of 2,688**. Of the repaired
opening picks, 2,378 have a same-team box, 95 a snap only, 212 a roster only
and three none of the three. None has an observed other-team record.

## What changed after openers

| Evidence for a different team, strongest available first | Published club board | With dated transactions |
|---|---:|---:|
| Player box | 10 | 1 |
| Positive mapped snap, no box | 2 | 0 |
| Unique weekly roster, no box or positive mapped snap | 18 | 1 |
| **Total** | **30** | **2** |

The two retained forecasts are **Christian McCaffrey for Carolina** (2022 W7;
[trade announced late Thursday](https://www.panthers.com/news/panthers-trade-christian-mccaffrey-to-49ers))
and **Carlos Hyde for Cleveland** (2018 W7; [trade announced Friday](https://www.clevelandbrowns.com/news/browns-trade-rb-carlos-hyde-to-jacksonville)).
McCaffrey later had a San Francisco box; Hyde had only a Jacksonville roster
record that week. Both moves came after the Thursday lock. Conversely, an
agreement for Mohamed Sanu and a reported trade for Mark Ingram were public
*before* lock, despite later official signings. We admit those destinations
without promising a snap. A release alone can rule out the old team: Kelvin
Benjamin had already left Buffalo, but a Kansas City workout was not a deal.
The first cited signing report was 01:03 UTC, after the 00:20 UTC lock.

[All 30 players/weeks and resulting selections](reviewed-mismatches.csv) ·
[All affiliation evidence, including the unknowns](affiliation-tiers.csv) ·
[Role selections by position](selection.csv)

Across the **41,422** non-opening picks on the repaired board, **33,554**
matched the actual starting QB or a top workload at the position, up from
33,537 of 41,421. The different denominator is one newly filled RB slot.
Correcting a player's team does not mean he immediately gets a starting role.

**Effect on RB forecasts:** with the same carry-share ensemble, point MAE
went from **4.924 to 4.920** and RMSE from 6.521 to 6.520. Across 209 weeks,
the average number of actual top-24 RBs chosen was 13.44 on both boards.
The points lost to the optimal 24 rose slightly, from **101.62 to 101.70**
per league-week. The repair matters for individual matchups, but it does
not explain the roughly five-point RB forecasting error.

## How to use this comparison

Both boards freeze the entire league one hour before its first weekly kickoff.
The [source ledger](../../../../src/ffpred/study/transaction_news.csv) records
the player IDs, origin and destination, type of move, public date or UTC time,
and source. Date-only news becomes usable at the following midnight Eastern;
an explicit timestamp can establish a same-day move. For Martellus Bennett
the clock is the [embedded reporter post](https://twitter.com/AdamSchefter/status/928732442722230272)
at 21:14 UTC, not the team recap's earlier page timestamp. A new acquisition
can enter the pool at the first eligible lock for his team, including after
a bye. A practice-squad agreement is not an active-team seed. A later dated
identity, depth chart, or earlier game box can establish a return to a former
team; a stale lagged roster alone cannot. Boxes, snaps and rosters for the
week being forecast are used **only to evaluate** its selections.

This is **retrospective error repair**, not a complete league transaction feed
or an independent estimate of future accuracy. We chose cases after observing
disagreements. The model family, carry features, rolling season splits, role
limits and non-PPR scoring are unchanged; the selected players and training
examples may differ. The club-only RB control reproduces the previous study's
forecast file byte for byte. In the full-league top-24 comparison, the oracle
can also select players outside our role pool.

```console
uv run --locked ffpred weekly-opportunity --transactions --end-year 2025 --output-dir artifacts/weekly-transactions
```

[RB scores](scores.csv) · [Annual errors](annual.csv) ·
[Top-24 RB picks and points lost](starters.csv) · [Reproduction manifest](manifest.json)

**Still unknown:** the [unresolved file](unresolved.csv) lists every pick on
both boards lacking all three evidence sources or having conflicting records.
Of the **122 non-opening picks** on the repaired board with none of the three
records, **111 are from 2014-2016**. Start with those seasons' official gamebooks
and game-day inactive lists before assigning a reason. Neither absence from all three sources nor low
usage proves an injury. A separate timestamped injury-exit dataset is needed
to evaluate injury-shortened games; the real outcomes stay in the primary
manager-facing scorecard.
