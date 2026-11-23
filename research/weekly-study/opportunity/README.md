# Who belongs on the opening board, and how many carries will they get?

**Use dated affiliations for team-openers and recent carry shares for RB forecasts.**
Affiliations recovered 301 correct opening role picks across 2014-2025. Carry
shares helped on the *same* RBs in 11 of 12 seasons by RMSE, but reduced their
average point error by only 0.014. Adding inside-five usage improved one
touchdown model slightly, not the overall point forecast or starter decisions.
These are different findings: picking the right player does not imply being
able to predict his points much more closely.

## Opening roles: did we pick the right people?

Both boards filled 2,687 of 2,688 slots across 384 team-openers: one QB, two
RBs, three WRs and one TE per team. A correct QB is the actual starter. At the
other positions, a correct pick is among the team's highest positive workloads
(carries plus targets for RB, targets for WR/TE), including ties at the boundary.
These are players to forecast, not yet the fantasy league's start/sit decisions.

| Position | Opening picks | Correct: previous board | Correct: dated board |
|---|---:|---:|---:|
| QB | 384 | 281 (73.2%) | 308 (80.2%) |
| RB | 768 | 437 (56.9%) | 524 (68.2%) |
| WR | 1,151 | 692 (60.1%) | 829 (72.0%) |
| TE | 384 | 214 (55.7%) | 264 (68.8%) |
| All | 2,687 | 1,624 (60.4%) | 1,925 (71.6%) |

Of the 576 opening picks replaced, only 46 were correct; 347 of their
replacements were correct. For example, the dated board correctly picked
Derek Carr for New Orleans and David Montgomery for Detroit in their 2023
openers. Picks with a statistical box for a **different team fell from 272 to
65**. A missing box is unverified, not a successful team match: there were 599
missing boxes before and 461 after. No non-opening role pick changed.

The archived directories cover **10 of 12 test seasons**, plus 2013 training.
On the covered openers alone, correct picks rose from 1,374 to 1,675 of 2,240;
wrong-team boxes fell from 213 to 6. Nine covered years gained correct picks;
2025, which already had dated depth charts, tied. For **2015 and 2018** we
found no sufficiently recent edition: all 447 picks are unchanged, including
59 of the remaining 65 wrong-team boxes. [By season](opening-annual.csv) ·
[Pooled and non-opening controls](selection.csv)

**What the sources establish.** We use pinned, pre-lock editions of `nflgame`
player files and the DynastyProcess player directory, not modern downloads
assumed to describe the past. These are not official 53-man rosters. A commit
time is evidence the file was available, not the date the player signed.
Only editions within 30 days before the first opening lock qualify. We use
GSIS IDs, team, position, explicit active/reserve statuses where available,
and rookie draft year, not fantasy rankings. We discard ambiguous ID listings.
A crosswalk alone can retain retirees, so new candidates need an existing
candidacy, a box in the preceding two seasons, a current rookie flag or
explicit status. A missing listing is **not** a cut. Explicit free-agent or
unavailable evidence can supersede a stale affiliation, and a later dated
trade, depth chart or applicable report can supersede the directory. An
affiliation confirms a team, not the starting role or health of a player.
[Archived editions, dates and SHA-256 hashes](editions.csv)

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

**Combined decision.** On the *dated* board, adding carry features reduced
point RMSE from 6.538 to **6.522**, MAE from 4.936 to **4.922**, and weekly
top-24 RB regret from **103.17 to 102.47 points** (13.40 actual top-24 picks).
Adding the goal-line block raised RMSE to 6.527 and regret to 102.79. Regret
is the total for all 24 RB slots in a 12-team league, not a single manager's
loss. Pool changes and model changes should not both be credited to carry
shares. For example, in **Week 1**, even perfect ranking of the old board
could score only 321.49 RB points on average; on the dated board that ceiling
was 329.02, against an unchanged league-wide best of 365.96. With baseline
features the actual ensemble picks scored 211.98 versus 222.88. Opening RB
MAE nevertheless *rose* from 4.79 to 4.84 on the changed population.
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
rookie yardage. Only five selected RB player-weeks lacked player history;
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
as controls, and report injury-return weeks separately. For opener selection,
dated cuts, activations and depth charts for the two uncovered years are a
more targeted investment than changing model families.
