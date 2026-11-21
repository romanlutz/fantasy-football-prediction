# Weekly fantasy points: from the 2015 paper to four positions

The question is how many non-PPR fantasy points a player will score this week,
including the possibility of zero. The forecasts cover completed regular-season
games through **2025**. They are not preseason projections; a lineup decision
can use these forecasts downstream.

## Primary experiment: team roles, one hour before kickoff

We select one projected starting QB, two RBs, three WRs and one TE for each
team playing that week. No players are assigned to a bye. The team's **first
game of the season** is outside this historical replay, since it has no prior
in-season roster. There are two distinct questions: did we select the right
players, and how accurately did we predict points for the players we selected?
A wrong pick, a late scratch and a legitimate zero all stay in the forecast
scorecard.

### Results, 2014-2025

The backtest filled **41,419 of 41,426 slots** across 5,918 team-games.
There was a QB pick for every team-game; five RB and two TE slots went
unfilled. The league-wide caps select 23,640 player-games instead. These are
different populations, not competing models scored on identical players.

| Position | Picks | Correct role picks | Returning role-holders selected | Trailing-ten MAE | Linear SVR MAE | Boosting MAE |
|---|---:|---:|---:|---:|---:|---:|
| QB | 5,918 | 90.6% | 52 / 66 (78.8%) | 7.08 | 6.80 | **6.61** |
| RB | 11,831 | 82.4% | 136 / 197 (69.0%) | 5.12 | **4.79** | 4.83 |
| WR | 17,754 | 83.5% | 238 / 314 (75.8%) | 4.38 | **4.17** | 4.25 |
| TE | 5,916 | 79.6% | 64 / 101 (63.4%) | 3.67 | **3.48** | 3.58 |

MAE is in non-PPR fantasy points, with the paper-style defense features held
fixed for the fitted models. Boosting also had the lowest RMSE at every
position. Both fitted models beat the trailing average on MAE. The additional
opponent-adjusted defense block changed pooled MAE by less than 0.01 point
at every position for both models.

**Injury returns:** we selected 490 of 678 documented returning role-holders
(72.3%). Of the 188 misses, **135 were already in the candidate list** but
ranked below the team cutoff; **53 never entered the list** at the relevant
position. Both are selection errors; neither can be found by looking at point
error only. Christian McCaffrey was correctly selected as San Francisco's RB1
in 2024 Week 10 after his Saturday activation, despite being absent from the
lagged roster. His activation and Joe Flacco's 2024 Week 9 starter announcement
are hand-verified examples, not independent evidence of archive-wide accuracy.

[Pooled selection audit](team-roles/selection-pooled.csv) ·
[Selection by season](team-roles/selection-by-season.csv) ·
[Pooled forecast scores for both cohorts](team-roles/scores-pooled.csv) ·
[Forecast scores by season](team-roles/scores-by-season.csv) ·
[Run manifest](team-roles/manifest.json)

**What the selector could know.** The cutoff is the scheduled kickoff in US
Eastern time minus 60 minutes, converted to UTC game by game. The initial pool
uses the team's prior-game roster. A dated depth chart, timestamped full or
limited practice report, or a sourced announcement can add a name absent from
that roster. Practicing is evidence of a *possible* return, not confirmation.
The selector excludes a known Out or inactive player, treats Questionable as
eligible, and ranks doubtful players behind other eligible candidates barring
an explicit starter announcement. It ranks an announced QB first, then prefers
a current QB1 depth listing when available. Other roles are ranked by the mean
of the player's trailing-three and trailing-ten workload averages: pass attempts
for QBs, carries plus targets for RBs, targets for WRs and TEs. A documented
missed game is not a zero in that workload history. A known game-specific
opportunity limit caps the workload estimate. Depth rank then player ID break
ties. Insufficient candidates leave a vacancy, not a retrospective substitution.

**What is missing.** The archive is not a complete historical feed of official
starters, activations or gameday inactives. Through 2024, depth charts have week
labels, not publication times, so only the prior team-game's chart can be used.
In 2025 the latest whole team depth snapshot no more than seven days old is
eligible at the cutoff; a removed player is not carried forward from an older
snapshot. Conversely, the 2025 injury release has no report modification time
and is not used for that game's selection. For older injury reports we use the
recorded modification time, not a guarantee we have every published revision.
No status from the target game's final roster or box score is used to select.
The [official announcement ledger](../../src/ffpred/study/role-news.csv)
contains two verified historical examples, not systematic coverage.

**How selection is scored.** The independent evaluation population is the
schedule's recorded starting QB and the team's players with the most positive
realized opportunities at each skill position, with everyone tied at the last
slot credited. This measures realized workload, not which eleven players took
the first snap. Precision counts correct picks divided by all picks. Return
recall counts selected returning role-holders divided by **all** realized
returning role-holders, including those never in the candidate pool. A marked
return requires no player box in the previous scheduled team-game and either
an injured-reserve roster code (`R01`) or an Out report with an injury field
for that game. A bye alone does not count, and unreported injuries are not
inferred. These outcome labels never enter selection for the same game.

Rebuild with `uv run --locked ffpred weekly-roles --end-year 2025`.
`artifacts/weekly-roles` contains the candidate list (including nonselections
and evidence times), predictions, realized roles, and per-season and pooled
scorecards. Use `--role-news PATH` to substitute a sourced game-specific CSV
using the bundled schema. `starter` is QB-only; `activated` establishes
eligibility, not a starting job; `inactive` rules a player out; `limited`
requires a numerical `max_opportunities`. A late report is ignored. The
previous league caps (24/36/36/24, ranked by prior ten-observation workload
only) are a separate comparison. Both cohorts use the same fitted forecasts
and broad-roster training history. Injury evidence changes *selection*, not
the estimators' point-prediction features in this comparison.

## Full-roster results: twelve held-out seasons, 2014-2025

**Player history does most of the work.** On 94,361 forecasts for 2,422
distinct players, linear SVR beats the trailing average on MAE at all four
positions. Gradient boosting beats both on RMSE at every position. The table
holds the defense inputs fixed at the paper-style last-game/last-ten block.
Each cell is **MAE / RMSE**, in fantasy points; lower is better.

| Position | Held-out player-games | Trailing ten | Linear SVR | Gradient boosting |
|---|---:|---:|---:|---:|
| QB | 14,788 | 4.289 / 6.715 | 3.652 / 6.232 | 3.770 / 6.095 |
| RB | 24,680 | 3.426 / 5.334 | 3.009 / 5.207 | 3.212 / 5.018 |
| WR | 34,828 | 3.113 / 4.675 | 2.805 / 4.708 | 2.997 / 4.494 |
| TE | 20,065 | 2.062 / 3.366 | 1.825 / 3.460 | 2.019 / 3.270 |

The choice of metric matters. The SVR forecasts are low by 0.74 to 1.22 points
on average; boosting's mean errors are within 0.05 of zero. RMSE penalizes
large misses more heavily. The estimators also optimize different losses, so
this comparison does not isolate nonlinearity alone. In a separate [2014 run
with the original libsvm solver](scores-exact-2014.csv), all twelve
position-by-defense SVR MAEs were within 0.003 of the scalable implementation.

**Does knowing how the defense plays help?** The entries below are the change
in pooled MAE when only the opponent features change (negative is better).

| Position | SVR: paper minus none | SVR: adjusted minus paper | Boost: paper minus none | Boost: adjusted minus paper |
|---|---:|---:|---:|---:|
| QB | -0.0064 | -0.0010 | -0.0120 | +0.0012 |
| RB | -0.0055 | +0.0007 | -0.0106 | +0.0041 |
| WR | -0.0007 | -0.0005 | +0.0008 | +0.0028 |
| TE | -0.0007 | -0.0019 | -0.0017 | -0.0023 |

For QBs and RBs, simple defense history beat no defense in **all 12 seasons**
with SVR, but by about 0.006 points pooled. The extra recency-weighted,
opponent-adjusted rates offer no general improvement over the paper block.
TE is the only position where they lower pooled MAE for **both** estimators,
by about 0.002 points. These are paired, retrospective differences, not
evidence that matchup is irrelevant or that small differences are significant.

**Do not compare the new QB MAE of 3.652 directly with the paper's 6.238.**
The new population includes backups: 7,879 of 14,788 QB weeks scored zero
(53%); across all positions 41,239 of 94,361 scored zero (44%). Among just
the 6,438 QB weeks with at least five attempts in the player's **previous**
observation, MAE is **6.918** for trailing ten, **6.835** for SVR, and
**6.641** for boosting with the same simple defense features. The ordering
changes without using the target game's attempts to select players.

The full 2010-2025 panel has 125,312 rows. In the held-out portion 31,431
candidates had no box, 2,436 came from the prior-box fallback and 70 player-weeks
with a box had a different team than forecast. The pool missed 3,445 nonzero
player-weeks in Week 1 and 1,277 in later weeks (104 QB, 434 RB/FB, 521 WR,
218 TE). The [coverage table](coverage.csv) makes these exclusions visible by
season and position. For a follow-up, the higher-priority experiment is
**pregame opportunity**: add prior snap/carry/target shares and timestamped
role or availability information. Report likely contributors, defined *before*
the game, alongside the full roster. Historical weather forecasts or DVOA
would need the version available at the weekly cutoff, not final weather or
an end-of-season rating.

## What changed from the paper?

| | 2015 experiment | Weekly follow-up |
|---|---|---|
| Data | Game Center boxes packaged with `nflgame`, 2009-2014 | nflverse schedules, weekly rosters, player and team boxes, 2009-2025; historical Game Center boxes for the 2014 audit |
| Players | Passers with at least five attempts **in the game being predicted**, subject to the original team-membership check | QB, RB (including FB/HB), WR and TE, selected from the **previous completed team-game**; no target-week opportunity requirement |
| Target | QB passing and rushing, penalizing every fumble | All positions receive credit for the same scoring plays; only a **lost** fumble costs two points |
| Features | 34 numeric features, including last-game and last-ten player and opposing-defense statistics | Player usage and scoring history; a controlled comparison with no defense, the paper-style defense history, and additional opponent-adjusted efficiency |
| Methods | Linear/kernel support-vector regression, feature selection and a shallow neural net; one 2014 test | Trailing-ten baseline, linear SVR and histogram gradient boosting; one expanding training window per position and held-out season |

The [paper](https://arxiv.org/abs/1505.06918) reported 6.238 QB mean absolute
error (MAE) in 2014. The historical-style nflverse builder and all-feature
linear SVR give 6.250 on 552 games. That small gap is **not** a measure of
progress: neither the modern boxes nor their membership are identical. In
2010-2013 the single-team check dropped 22 games for Trent Edwards (4), Kyle
Orton (8), Matt Flynn (6) and Josh Freeman (4) that nflverse restores; the
box-position filter loses 11 Terrelle Pryor games labeled WR. That explains
2,167 versus 2,178 training rows. In 2014 the only membership difference
between those two historical-style samples is Logan Thomas (553 versus 552).

### Accounting for 2014, player by player

The [2014 reconciliation](reconciliation-2014.csv) replays all **553**
original passer-games. Of those, **517** are also in the new pregame pool, **33**
are in Week 1, and just **three** later passers were not on the prior eligible
roster: Matt Simms (Week 9), Case Keenum (Week 16) and Connor Shaw (Week 17).
The new pool also contains 7,074 other player-games, for **7,591** total:
1,194 QB, 2,100 RB, 2,728 WR and 1,569 TE. Those are player-games, not people.

On the 553 games, changing the source while keeping the old formula changes 37
scores. On modern boxes, deducting points only for lost fumbles changes 150;
other scoring changes affect six. These groups overlap and corrections can
cancel: **164** end scores differ between the archived boxes under the old
rules and the modern boxes under the new rules. The CSV gives the three
additive deltas per game, not just a count of discrepancies.

Two rows illustrate why the distinction matters. Logan Thomas had eight
attempts in Arizona's Week 5 game. His box says TE, his prior roster says QB,
and all scoring methods give **7.24**. His was an eligibility difference.
Andy Dalton's Week 3 box records an 18-yard receiving touchdown. His score
goes from **5.06 to 12.86**, entirely a scoring difference.

## Pregame data, eligibility and scoring

Each forecast has a stable player ID, season, week, expected team/opponent and
game. The weekly roster has no reliable publication timestamp, so eligibility
comes from the team's last **completed** game in the same season. A bye is not a
zero. Week 1 has no prior in-season roster and is outside the forecast; the same
rule applies to teams whose opening game was delayed. No target-week roster,
starter designation or eventual touches enter the model.

Use the prior game's `status_description_abbr` (`A01`, `I01`, `I02`), not the
coarser `status`: Ryan Fitzpatrick's 2014 Houston records, for example, say
`RES` even in weeks he started, while the game-specific code says `A01`. When
a player has no weekly code for that team, include them only if they had a box
for the team in that prior game, preferring the roster position. If a weekly
code exists it takes precedence. This matters for 2016, when many rosters are
only partially coded. The fallback is marked `prior_box`; it can miss backups.
New signings and promotions can be missed and departed players can remain.
Historical records can be revised;
lagging observations does not reconstruct a timestamped news feed.

Score the **player-week**, even if the player changes teams before kickoff. A
candidate with no box in a week with a completed scheduled game gets **zero**,
not an inferred injury or inactivity label. The panel preserves actual team,
game and `has_box` separately. A missing team-game in the box feeds fails the
build rather than assigning an entire team zeros.

The fixed non-PPR preset awards 1 point per 25 passing yards, 4 per passing TD,
1 per 10 rushing or receiving yards, 6 per rushing, receiving, return or
fumble-recovery TD, and 2 per successful passing, rushing or receiving
two-point conversion. Interceptions and **lost** fumbles cost 2 each;
receptions alone earn 0. The code computes this from components, not from a
provider's premade fantasy-points column. The 2014 audit also independently
scores the contemporaneous boxes under the original rules, which remain
available in [`legacy-2015`](https://github.com/romanlutz/fantasy-football-prediction/tree/legacy-2015).

## Defense experiment and evaluation

Player features are a count of prior observations and the last observation and
last ten observations of points, pass attempts, carries, targets, receptions
and passing/rushing/receiving yards. History includes earlier recorded boxes
and previously forecast roster weeks that scored zero. The trailing baseline
uses the available history, or the training-position mean for a debut.

| Defense block | What was known before the target week |
|---|---|
| None | Player history only. Team names and IDs are not predictors. |
| Paper-style | Add the opponent's previous-game and last-ten averages for scoreboard points, passing yards, rushing yards and turnovers allowed/forced. Scoreboard points include all scoring, not just defensive responsibility. |
| Adjusted | Keep all eight paper-style features. Add the defense's estimated allowed rushing yards, EPA and 10+ yard runs per carry, and passing yards, EPA and 20+ yard passes per approximated dropback (attempts plus sacks). |

EPA means expected points added. For each past game, estimate the offense's
expected rate from its earlier games, shrunk toward the prior league rate.
Subtract expected production from observed production and accumulate these
residuals for the opposing defense. The pregame defensive rate is the league
rate plus the defense's residual per play, shrunk toward zero with 50 prior
plays. Offense and defense history have a four-game half-life; 25% carries
across each offseason. All teams' snapshots for a week are formed **before**
that week's observations update either side. These are public-data estimates,
not DVOA. No observed game-day weather, injuries, betting lines or end-of-season
ratings are model inputs.

Use 2009 to initialize features, train on eligible rows from 2010 onward, then
predict 2014 from 2010-2013, 2015 from 2010-2014, and so on through 2025.
Within a held-out season earlier completed weeks update features, not model
weights. Imputation and SVR scaling fit on training rows only. The benchmark
uses linear epsilon-SVR with the paper's `C=0.25`, `epsilon=0.25`, using the
scalable liblinear solver by default. The boosted model is scikit-learn's
squared-error histogram gradient boosting with 80 iterations, 15 leaves,
minimum leaf size 30 and learning rate 0.07. Hyperparameters are fixed across
seasons and positions. Each model forecasts exactly the same player-games.
Scores include MAE, root mean squared error (RMSE), and signed bias
(forecast minus actual); pooled scores weight player-games, not seasons, equally.

## Reproduce and inspect

```console
uv sync --locked
uv run --locked ffpred weekly-study --end-year 2025
```

The default output is `artifacts/weekly-study`. Set `FFPRED_CACHE_MODE=filesystem`
to cache nflverse downloads. To use the paper's slower libsvm linear-kernel
solver on the **new** panel:

```console
uv run --locked ffpred weekly-study --end-year 2014 --exact-svr --output-dir artifacts/weekly-study/exact-2014
```

This changes only the SVR solver, not the rows, targets or features. The separate
`build-dataset` and `train-svr` commands reproduce the historical-style QB task.

| Output | Use |
|---|---|
| `player-games.parquet` | One row per eligible player-week, with target, pregame features and source/lineage fields |
| `predictions.parquet` | Actual points and all seven forecasts on each held-out player-week |
| [`scores-by-season.csv`](scores-by-season.csv) / [`scores-pooled.csv`](scores-pooled.csv) | Position-by-model MAE, RMSE, bias and counts |
| [`reconciliation-2014.csv`](reconciliation-2014.csv) | Union of every original qualifying passer and every new candidate, keyed by player-week; separates source, fumble and other scoring differences |
| [`coverage.csv`](coverage.csv) | Candidate, zero, missing-box, fallback and team-change counts; nonzero box players missed in Week 1 vs later weeks |
| [`manifest.json`](manifest.json) | Scoring, source-frame and archive hashes, model inputs, dependency versions and output row counts/hashes for this run |

Data: [nflverse releases](https://github.com/nflverse/nflverse-data/releases)
via `nflreadpy`, and [nflgame 1.2.15](https://pypi.org/project/nflgame/1.2.15/)
Game Center boxes for the original scores. The tables here are derived
research outputs; see [Data licensing](../../README.md#data-licensing).
