# Who belongs on the opening board, and how many carries will they get?

**Use dated affiliations for team-openers and recent carry shares for RB forecasts.**
Team roster revisions recovered 155 more correct opening role picks than the
earlier directory-based board and eliminated all 65 observed wrong-team boxes.
Carry shares helped on the *same* RBs in 11 of 12 seasons by RMSE, but reduced their
average point error by only 0.014. Adding inside-five usage improved one
touchdown model slightly, not the overall point forecast or starter decisions.
These are different findings: picking the right player does not imply being
able to predict his points much more closely.

**Follow-up:** The [transaction audit](transactions/README.md) accounts for
the 216 openers without a box or positive snap, checks the rest of the season
against weekly rosters, and replays both boards with the same RB model.

## Opening roles: did we pick the right people?

The repaired board filled all 2,688 slots across 384 team-openers: one QB, two
RBs, three WRs and one TE per team. A correct QB is the actual starter. At the
other positions, a correct pick is among the team's highest positive workloads
(carries plus targets for RB, targets for WR/TE), including ties at the boundary.
These are players to forecast, not yet the fantasy league's start/sit decisions.

| Position | Correct: no directories | Correct: directories only | Correct: with club rosters |
|---|---:|---:|---:|
| QB | 281 (73.2%) | 308 (80.2%) | 322 (83.9%) |
| RB | 437 (56.9%) | 524 (68.2%) | 583 (75.9%) |
| WR | 692 (60.1%) | 829 (72.0%) | 892 (77.4%) |
| TE | 214 (55.7%) | 264 (68.8%) | 283 (73.7%) |
| All | 1,624 / 2,687 (60.4%) | 1,925 / 2,687 (71.6%) | 2,080 / 2,688 (77.4%) |

**The two missing seasons:** 2015 improved from 129 to 168 correct opening
picks, and wrong-team boxes fell from 31 to zero. For 2018 the changes were
121 to 172 correct and 28 to zero; the previously empty WR slot was filled.
Across all 12 test years, wrong-team boxes fell from **272 to 65 to zero**
across the three boards. No non-opening pick changed. [Before/after by season
and position](roster-repair.csv) · [Pooled results](selection.csv)

**Zero observed mismatches is not perfect verification.** Of 2,688 repaired
opening picks, 2,377 had a box for the predicted team. Positive snap counts
verified another 95 who had no box. That leaves **216 unverified**, not 216
wrong, confirmed inactive, or injured. No boxless opener with an unambiguous
positive snap played for another team; two had done so on the directory-only
board. After openers, **10** wrong-team boxes and **two** additional wrong-team
snap-only records remain. Actual snaps and box teams are audit labels only.
[Independent affiliation audit](affiliation-audit.csv)

**Sources and cutoff.** We took the last publicly timestamped revision of each
franchise's Wikipedia roster template at or before the league lock, for all
32 clubs in 2013 training and 2014-2025 evaluation. All 416 snapshots were
within 81 hours of lock. They are community-maintained, not official game-day
inactives or depth charts. We use only names, positions, and active, reserve
or practice-squad listings. Each [revision](club-revisions.csv) links to its
contributors' version and records its UTC edit time, content hash and linkage
counts. The [extractor](extract_club_rosters.py) reproduces the bundled facts
from immutable revision IDs, not today's rendered templates. No player prose
or depth ordering is used.

We linked 9,220 of 9,469 listings to GSIS IDs using exact normalized names.
For namesakes we use only prior-season rosters and the draft, never the team
for which someone later played. Unresolved names and conflicting club records
are not guessed. Earlier [player directories](editions.csv) remain fallback
evidence. Explicit reserve or practice-squad status is distinguished from a
missing listing; absence alone never means released. A newer dated move,
applicable report or depth chart can supersede a roster. Confirmed affiliation
matters for filling slots, but is neither a starter announcement nor evidence
of health. Use `--directory-only` to reproduce the
[previous study](https://github.com/romanlutz/fantasy-football-prediction/tree/3ce61ee/research/weekly-study/opportunity).

## RB forecasts: do workload features help for the same player?

Each column below uses exactly the same 12,602 RB player-weeks, teams, targets
and deadlines from the [published matchup board](../matchups/README.md).
The no-new-features run reproduced its RB forecasts exactly. Only model
inputs vary, not the estimators, chronological folds, training populations,
ensemble weights or scoring. **Carry + inside-five** adds goal-line features
on top of the carry block. Errors include zero-touch games.

| On the original board | Existing features | Add carry shares | Carry + inside-five |
|---|---:|---:|---:|
| Ensemble point MAE | 4.929 | **4.915** | 4.918 |
| Ensemble point RMSE | 6.541 | **6.527** | 6.532 |
| Signed point bias (forecast minus actual) | -0.114 | -0.101 | -0.106 |
| Conditional linear model: carry MAE, carries | 4.537 | **4.496** | 4.503 |
| Conditional forest: carry MAE, carries | 4.587 | **4.557** | 4.566 |
| Conditional forest: rushing-TD RMSE, TDs | 0.5525 | 0.5528 | **0.5520** |
| Actual top-24 RBs picked per week | 13.33 | **13.37** | 13.33 |
| Points lost to full-league top 24 per week | 103.38 | **102.75** | 103.24 |

Carry features improved point MAE in 10/12 seasons and RMSE in 11/12.
The inside-five variant beat the original on both in 8/12 but did worse
than carry alone on both pooled measures. Adding inside-five information to
the carry block reduced forest rushing-TD RMSE by less than 0.001 TD, while
the linear model's rushing-TD RMSE increased slightly. The forest's
rushing-yard MAE contribution attributable to workload only fell from
1.354 to 1.347 fantasy points with carry features. Historical share is useful
but does not resolve surprise workloads. The refit 80% RB outcome bands with
carry features covered 84.1% of outcomes at 15.03 points mean width, versus
84.5% at 15.04 before. The small mean improvement is not a reason to promise
narrower player ranges. [Scores and bias](scores.csv) · [By season](annual.csv)
· [Opening vs. later games](opener-scores.csv)

**Combined decision.** On the *repaired* board, adding carry features reduced
point RMSE from 6.537 to **6.521**, MAE from 4.936 to **4.924**, and weekly
top-24 RB regret from **102.27 to 101.62 points** (13.44 actual top-24 picks).
Adding the goal-line block raised RMSE to 6.527 and regret to 101.81. Regret
is the total for all 24 RB slots in a 12-team league, not a single manager's
loss. Pool changes and model changes should not both be credited to carry
shares. For example, in **Week 1**, perfect ranking of the directory-only
board could score 329.02 RB points on average; on the repaired board that
ceiling was 338.73. With baseline features the actual ensemble picks scored
222.88 versus 231.49. With carry features, they scored 216.54 versus 231.65.
These are outcomes for *different selections*, not improvements measured
on identical players.
[Top-24 decisions](starters.csv) · [Annual decisions](starter-annual.csv)

## Workload data and reproduction

Use regular-season play-by-play from 2009 through 2025 (8,862 offensive
team-games). Count a run when `play_type=run` and `rush_attempt=1`, excluding
kneels. An inside-five chance starts at the opponent's 1 through 5, inclusive:
count runs and passes with an identified target separately. Exclude deleted
plays, `no_play` penalties and two-point tries. Count runs by **all** positions
in the team denominator even if the individual ID is missing. A game without
an inside-five run is zero, not missing; this occurred in 3,510 team-games.
These opportunity counts are inputs, not replacements for official box-score
carries, TDs or corrected non-PPR points. [Play coverage](play-coverage.csv)

Team volume is the mean of up to eight preceding team-games. A player's
carry share is their fraction of the team's runs across up to five preceding
appearances. An appearance with a mapped positive snap but no usage is a true
zero; before snap coverage, runs or identifiable targets establish appearances.
When snap-only records disagree on team we do not invent a zero. Histories skip
byes and absences, cross seasons, and follow the player to a new team. Projected
volume multiplies player share by the **current** team's prior volume. We also
include other non-ruled-out RB candidates' summed historical carry shares;
this measures competition and need not sum to one. Inside-five run and target
shares each shrink toward overall carry share with three prior opportunities.
Missing history has its own indicator and train-only imputation, not invented
rookie yardage. Five selected RB player-weeks on the original board and 11
on the repaired board lacked player history;
all had team history. [History coverage](history-coverage.csv)

```console
uv run --locked ffpred weekly-opportunity --end-year 2025
```

This writes six RB forecast Parquets, compact tables and a [manifest](manifest.json)
to `artifacts/weekly-opportunity` without changing `weekly-matchups`. Folds
train on earlier seasons; within-season games update inputs, not fitted
weights. The common lock is one hour before the first weekly kickoff. In-game
injury exits stay in the manager-facing evaluation; they cannot be identified
reliably from low usage. No 2026 games or preseason performance were used.

**Next experiment:** forecast the team's rushing volume, then allocate that
volume among available rushers with shares that add up. Test how a teammate's
confirmed absence or return redistributes carries, rather than assuming
everyone keeps their past share. Keep these exact RB rows and top-24 metric
as controls, and report injury-return weeks separately. The [in-season
affiliation audit](transactions/README.md) distinguishes transactions known
by the weekly lock from moves that happened later. A separate historical injury-exit
ledger would be needed to verify mid-game injuries.
