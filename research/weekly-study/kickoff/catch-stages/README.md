# Do completions and catches improve fantasy-point forecasts?

**Adding a catch stage explained the yardage miss but did not materially
improve the points forecast.** We replayed all 12 regular seasons from 2014
through 2025: 44,109 forecasts issued, the same 44,081 verified matchups
scored, and the same 28 outcomes withheld. No 2026 or preseason games.

| Position | Scored games | Direct ridge MAE | Per-opportunity linear MAE | Catch-stage linear MAE | Original ensemble MAE | Catch-stage ensemble MAE |
|---|---:|---:|---:|---:|---:|---:|
| QB | 6,301 | 6.6577 | 6.6439 | 6.6452 | 6.6171 | 6.6181 |
| RB | 12,588 | 4.8848 | 4.8870 | 4.8869 | 4.8733 | 4.8734 |
| WR | 18,894 | 4.3259 | 4.3186 | 4.3182 | 4.3083 | 4.3088 |
| TE | 6,298 | 3.6696 | 3.6814 | 3.6814 | 3.6742 | 3.6742 |

The direct column is the linear points-only comparator, not a selection of
the best direct model by position. The two ensembles each average the same
two direct models and the corresponding linear and forest component models.
For the ensemble, neither MAE nor RMSE changed by as much as 0.003 points at
any position. All 12 paired MAE comparisons (three model pairs, four positions)
had 95% season-bootstrap intervals spanning zero. With only 12 seasons this
does not prove equivalence, but it gives no reason to replace the point model.
[All models, including direct boosters and component forests](scores.csv) ·
[Annual scores](annual.csv) · [Paired differences and intervals](paired.csv)

## Where the yardage error came from

On every scored game, including games with no usage, we split the linear
model's absolute yardage error into contributions from volume, catch rate
and yield per catch. All four columns below are **yards per player-game**,
not fantasy points or errors conditional on playing.

| Position and channel | Total yardage MAE | Attempts or targets | Completion or catch rate | Yards per catch and residual |
|---|---:|---:|---:|---:|
| QB passing | 71.65 | 48.05 | 8.82 | 14.79 |
| RB receiving | 12.93 | 7.24 | 1.39 | 4.30 |
| WR receiving | 26.98 | 13.93 | 5.17 | 7.88 |
| TE receiving | 21.95 | 12.19 | 4.05 | 5.71 |

The same ordering held for the forests. For touchdowns, the pattern reversed:
for QBs, 0.73 of the linear model's 0.93 passing-TD MAE was the TD yield
after completions; for WRs, 0.34 of 0.39 receiving-TD MAE was yield after
catches. The linear model missed QB completions by 5.94 per game and WR
receptions by 1.79. This points to workload as a target for improving yards,
not a claim that perfectly forecasting workload would eliminate that many
yards of error. [Stage and TD diagnostics in their original units](stages.csv)

**How to read the decomposition.** Price the opportunity-count error at the
forecast catch rate and yield. Next, at the actual number of opportunities,
price the catch-count miss at the forecast yield. What remains is the yield
miss, including rare production with no catch. Multiply each signed term by
the sign of that game's total stat error before averaging. The terms sum to
stat MAE before rounding; offsetting mistakes can contribute negatively. This
is an accounting breakdown, not a causal analysis or three separate MAEs.

## What was held fixed

- The [audited kickoff board](../README.md), pregame inputs, previous-season
  training cutoff, position-specific models, play probability, and predicted
  passing attempts, carries and targets are unchanged. Only passing and
  receiving production are replaced. Rushing, interceptions, lost fumbles,
  miscellaneous scoring and non-PPR weights are the original forecasts and
  rules; catches themselves earn no points.
- Among players who played, fit completions per attempt and catches per
  target, weighted by attempts or targets. Fit yards and TDs per completed
  pass or reception, weighted by completions or receptions. Multiply these
  rates by the forecast volumes, applying play probability only once. Include
  a training-only allowance for yards or TDs credited with no recorded catch
  (for example, laterals). Actual counts for the forecast game are outcomes
  only, never predictors. No new catch-rate features were introduced, so this
  measures the effect of changing the prediction targets, not of richer inputs.
- Resample the 12 paired held-out seasons 4,000 times, preserving player-game
  weights within each draw. The original seven models reproduce every
  published annual and pooled score. Selecting the original 80 columns of the
  forecast Parquet reproduces its [published SHA-256](../kickoff-audited/manifest.json)
  exactly, although some raw source-frame fingerprints differ between runs.

```console
uv run --locked ffpred weekly-kickoff --arm catch-stages --end-year 2025
```

The [manifest](manifest.json) records source and output hashes. Full
per-player projections, including the predicted catch counts, rates, yards,
touchdowns and expected points, are written to
`artifacts/weekly-kickoff/catch-stages/forecast.parquet`.
