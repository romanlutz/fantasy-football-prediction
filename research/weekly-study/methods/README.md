# Do newer models beat the 2015 paper's methods?

**Not across the board.** On the same pregame selections, additive boosting
and a paper-style sigmoid network both beat linear SVR for QBs in all twelve
seasons. For TEs the linear SVR still has the lowest mean absolute error
(MAE). For RBs and WRs the best MAE gains over SVR are just 0.010 and 0.021
points. The bigger decision is whether the published number is a **typical
score** or **expected points**. An estimator can win on the first and
systematically undershoot the second.

## Results on the same team roles, 2014-2025

Each cell is pooled **MAE / RMSE** in non-PPR points, in that order. Lower
is better. Bold identifies the lowest value in its position and metric, not
an independently established winner.

| Method | QB | RB | WR | TE |
|---|---:|---:|---:|---:|
| Trailing ten | 7.080 / 8.863 | 5.122 / 6.892 | 4.383 / 5.831 | 3.666 / 4.943 |
| Paper-style linear SVR | 6.795 / 8.636 | 4.794 / 6.947 | 4.170 / 6.042 | **3.484** / 5.171 |
| Approximate RBF SVR | 7.384 / 9.397 | 4.962 / 7.257 | 4.238 / 6.185 | 3.614 / 5.406 |
| Paper-style sigmoid net | 6.614 / **8.305** | 4.832 / 6.588 | 4.253 / 5.695 | 3.576 / **4.813** |
| ReLU / Adam net | 6.638 / 8.346 | 4.826 / 6.604 | 4.260 / 5.709 | 3.577 / 4.866 |
| Ridge | 6.665 / 8.369 | 4.817 / **6.586** | 4.240 / 5.702 | 3.549 / 4.823 |
| Squared-error tree boosting | 6.614 / 8.379 | 4.829 / 6.600 | 4.246 / 5.699 | 3.579 / 4.853 |
| Absolute-error tree boosting | 7.700 / 10.079 | **4.784** / 6.969 | **4.150** / 6.022 | 3.713 / 5.475 |
| Additive EBM | **6.607** / 8.358 | 4.847 / 6.601 | 4.247 / **5.694** | 3.593 / 4.854 |

**QB is the clearest improvement.** Additive boosting improves MAE by 0.188
points versus linear SVR (paired season-bootstrap 95% interval for the
improvement: 0.128 to 0.254). But its lead over the sigmoid net is only
0.007; it wins that comparison in seven of twelve seasons, and the paired
interval for EBM minus sigmoid spans -0.059 to +0.046. The sigmoid net also
has the best QB RMSE. There is evidence against linear SVR as the sole QB
model, not evidence that a more complicated model beats the old neural-net
family. Exact RBF beat its approximation in 2014 at every position, but
still lost to linear SVR on MAE at all four; do not generalize from one
kernel width or treat the approximation as an exact 12-season test.

**The skill positions illustrate the cost of picking one metric.** Absolute
boosting's RB MAE gain over linear SVR has a paired interval spanning zero
(-0.042 to +0.025 for new minus SVR). The WR difference is small but more
consistent (-0.041 to -0.004); both won in nine seasons. Linear SVR has the
best TE MAE but forecasts **1.94 points below actual on average**, versus
0.36 low for the sigmoid net. For expected points, ridge is a useful simple
control: its RB RMSE beats the sigmoid net by just 0.002, and its TE RMSE
trails by 0.010. EBM's WR RMSE beats the sigmoid net by only 0.001. These
gaps are not reasons to maintain four different production models.

**The training population matters.** Absolute-error boosting predicts exactly
zero for 1,163 QB selections and 1,580 TE selections, versus 490 and 780
observed zeros respectively. Every selected TE even received a zero forecast
in 2018 Week 5. Learning a median from a roster full of backups is a plausible
explanation, not a proven cause. Test training on historically projected
roles, or including the pregame role probability, against the same held-out
slots. Do not train on *realized* starters only.

**Weekly rankings tell a different story again.** Averaging Spearman
correlations within position and week, ReLU improves QB ranking from 0.329
(linear SVR) to 0.370 despite not winning MAE. The best values at RB, WR
and TE are about 0.52, 0.43 and 0.36. Each defined week has equal weight;
the constant TE week is undefined for absolute boosting (196 ranked weeks
instead of 197). This ranks players at their respective kickoff cutoffs; a
lineup decision made on Thursday needs forecasts frozen as of Thursday.
The uncertainty intervals above resample the 12 whole seasons together for
both models (4,000 draws, player-slot-weighted errors), not 41,419 supposedly
independent players. They are descriptive with only 12 seasons, overlapping
training sets and multiple model comparisons.

## What the paper actually found

The [published paper](https://arxiv.org/pdf/1505.06918) (Tables 1 and 2)
trained on 2010-2013, using 2009 for history, and tested on 2014 QBs. Its
24-player subset was selected using **2014 season scores**, unlike an
eligibility rule you could have applied before each game.

| Published 2014 QB method | All passers: MAE / RMSE | Selected 24: MAE / RMSE |
|---|---:|---:|
| Linear SVR, all features | 6.238 / 7.815 | 6.265 / 7.925 |
| Linear SVR, recursive feature elimination | **6.221 / 7.759** | 6.248 / **7.833** |
| Linear SVR, manual feature removal | 6.224 / 7.796 | 6.256 / 7.914 |
| Shallow net, 50 sigmoid units, 50-epoch setting | 6.441 / 8.061 | **6.235** / 7.868 |

The standout neural net beat the best SVR on **top-24 MAE by 0.013**, not
on all-passer MAE or top-24 RMSE. Most nets did worse. The paper explored
linear, radial-basis, polynomial and sigmoid SVR kernels; the selected
kernel was linear, with `C=.25, epsilon=.25`. PyBrain tried one hidden layer
with 10/25/50/100 units, sigmoid or tanh and several epoch limits. The
historical-style nflverse all-feature SVR result was **6.250 on 552 games**;
the [player-by-player reconciliation](../README.md#accounting-for-2014-player-by-player)
explains why that is a replication check, not a claim of improvement.

We do **not** preserve two aspects of the archived
[SVR script](https://github.com/romanlutz/fantasy-football-prediction/blob/legacy-2015/models.py):
it standardized concatenated train and test features, and its optional search
did not validate forward in time. The [network script](https://github.com/romanlutz/fantasy-football-prediction/blob/legacy-2015/neural_net.py)
used raw inputs and backpropagation with a validation-based stopping rule.

## The matched experiment

Every model predicts the **same 41,419 pregame team slots** across 197 NFL
weeks, 2014-2025: 5,918 QB, 11,831 RB, 17,754 WR and 5,916 TE. Each team is
scored from its second regular-season game onward. One hour before kickoff,
the selector chooses up to 1/2/3/1 at the four positions. Targets use the
same corrected non-PPR scoring. Neither an eventual touch nor surviving the
entire game is an entry requirement.

Each position and test year gets its own model. Fit on the broad-roster
panel starting in 2010 and ending the **previous season**; earlier games in
the test season update the predictors, not the weights. The fitted methods
get exactly 25 inputs: 17 player-history features and eight last-game/last-ten
opponent statistics. These are *not* the original 34 QB features. Imputation
and, where applicable, scaling use training data only. Role, injury and news
evidence select who is forecast but do not enter these regressors. The
reference forecasts include players added from outside the prior roster;
there is no inner join to only the players who were easiest to model.

| Model | What is held fixed or varied |
|---|---|
| Trailing ten | Mean points over last ten observations; training-position mean if none. No fitted matchup adjustment. |
| Linear SVR | Scaled linear epsilon-SVR (`C=.25`, `epsilon=.25`), using the scalable solver rather than the 2015 libsvm implementation. |
| RBF SVR | Nonlinear family from the paper's search, not its winner. Standardized inputs, `gamma=.05`, 128 Nyström landmarks sampled from training only, same linear SVR loss and `C`. Also ran **exact** libsvm RBF on all four positions for 2014 only. |
| Sigmoid network | Mirrors the paper's standout 50-unit, 50-epoch architecture: SGD, rate .01, no momentum or weight penalty. Sklearn minibatches, standardized inputs and a fixed epoch budget replace PyBrain's training and stopping; not an exact replay. |
| ReLU network | Same width and 50-epoch budget, Adam at .001, weight penalty .0001, ReLU activation. A modern shallow-net alternative, not an isolated activation test. |
| Ridge | Standardized linear least squares with L2 penalty 10. A simple expected-points control versus SVR's different loss. |
| Squared / absolute boosting | Histogram gradient boosting: 80 rounds, 15 leaves, minimum 30 records per leaf, rate .07, no early stopping. Change **only** the loss: squared error targets the conditional mean; absolute error targets the median. |
| Additive EBM | Explainable Boosting Machine: no interactions, 64 bins, 200 rounds, rate .04, minimum leaf 10, one bag, no validation holdback or smoothing rounds. Tests additive nonlinear effects, not the best possible tuned EBM. |

All stochastic estimators use seed 42. We fixed these settings before running
the comparison and did not tune them on the test folds. We had already
examined these seasons for the reference models: a winner identified here
is a retrospective finding, not an independently confirmed choice. The
neural nets are fixed-budget runs, not claims of convergence. The 2014
exact-kernel check is not a 12-season result.

## Data and evaluation decisions for the next paper

1. **Fix roster identity before changing the model again.** All game/team/player
   keys are unique, but there are **41,414 distinct player-weeks for 41,419
   slots**. Five trades selected one player for both teams: Josh Gordon
   (2018 W3), Christian McCaffrey (2022 W7), Davante Adams (2024 W7), Tank
   Bigsby (2025 W2) and Rashid Shaheed (2025 W10). An old-team lagged roster
   and new-team practice report or dated chart coexist. The fantasy target is
   the player-week, so it is scored twice with different matchup features.
   Freeze this cohort for model comparisons; for the next version, invalidate
   the old affiliation only once a sourced transaction is known by that
   specific kickoff, select the next eligible teammate, and publish the
   five-case bridge. Removing both entries for all five cases changes no
   model/position MAE by more than 0.002 points; it matters for lineup
   integrity more than for these rankings. Do not use the eventual box
   team to repair past inputs.
2. **Keep the failures that a manager experiences.** There are **5,127 zero
   and 371 negative scores**. Of 6,770 picks that did not match a realized
   role, 4,416 scored zero (65.2%); among 34,649 correct picks, 711 did
   (2.1%). Neither a zero nor a low snap count establishes an injury. Keep
   healthy blowouts, benchings, late scratches, real negative scores, and
   multi-TD games. Exclude byes, unplayed games and genuinely corrupt or
   uncovered feeds; distinguish a covered game with no player box from a
   missing game. Audit scoring components, game IDs, team aliases, position
   changes and corrected stats separately instead of trimming by points.
3. **Make opportunity and injury evidence explicit.** The fixed slot caps
   describe an actionable population, not an oracle for actual usage. Keep
   the broader roster and league caps as sensitivity cohorts. Only **490/678**
   documented returning role-holders were selected: 135 misses were ranked
   too low and 53 were not in the pool. Improve timestamped activations,
   inactives, starter changes and snap limits. Separate the chance of
   playing from expected carries, routes, targets or dropbacks if active,
   then model points conditional on workload. Use pregame estimates, never
   realized snaps, to predict. For the next model, lag separate *availability*
   and *production when playing* histories; the present point histories
   count some no-box weeks as zero even though the selector skips documented
   injury absences when ranking workload. There were **1,169** selected slots
   after a documented missed team-game; **644 scored zero**. Squared-error
   boosting underpredicted the 98 such QB picks by **5.20 points on average**.
   Restricting to the 52 who actually regained the starting job increases
   that figure to **11.94**. The first group includes unsuccessful returns;
   the second is a hindsight diagnostic, not the expected error of a player
   announced as returning. Neither group represents in-game injury exits.
4. **Distinguish injury exits from ordinary playing time.** Keep an all-game
   scorecard for the deployment question. Add a second one excluding only
   sourced in-game exits with no return; record left-and-returned and unknown
   separately. Where the exit time and expected role are credible, report an
   exposure-adjusted diagnostic, not a replacement fantasy score. Playing
   50% of team snaps may be normal for RB2; it does not imply losing half his
   workload. No comprehensive exit ledger is available in this run.
5. **Restore openers and respect source changes.** Openers are absent because
   there is no previous *in-season roster*, not because every player lacks
   history. Of 384 actual opening starting QBs, 360 had a box the previous
   season. Seed eligibility from a pregame roster/depth snapshot; carry over
   veteran usage with shrinkage and use earlier-season position/role priors
   for rookies. Verify and score opening eligibility, rookies, team changers
   and established starters separately. Weekly depth charts through 2024 lack
   publication times; 2025 has dated charts but no usable injury-report
   modification times. A missing report is not evidence of health. Track
   coverage by season; compare recent vs expanding training windows, including
   the 17-game era, using earlier rolling seasons for any tuning.
6. **Use the metric that answers the question.** Points MAE measures a typical
   absolute miss and rewards median forecasts. RMSE emphasizes costly large
   misses and rewards expected-points forecasts. Pair these with signed bias
   (forecast minus actual), weekly within-position rank correlation, and
   calibration by projected-points bands. Absolute/relative percentage error
   is unsuitable around zero or negative values; the old implementation
   divides by the prediction, not the actual. Next add calibrated quantiles:
   report pinball loss and both coverage **and width** of 50%/80% intervals,
   including on returns. For lineup value, compare feasible choices from the
   **same roster and decision time**; ranking the whole league is not itself a
   start/sit decision. Historical weather forecasts, roof announcements and
   opponent-adjusted ratings can follow, if the version actually available at
   kickoff is archived. Final weather and end-of-season ratings are not valid
   substitutes.

## Reproduce

```console
uv run --locked ffpred weekly-roles --end-year 2025 --compare-models --output-dir artifacts/weekly-methods
```

The output includes the keyed game-level `method-predictions.parquet` and
the original forecast, candidate and realized-role files. The run used
scikit-learn 1.9.0, InterpretML 0.7.8, NumPy 2.5.2 and Polars 1.43.2. When
compared by key, **every field** in the original candidate, realized-role
and reference prediction tables matched the previous publication. Four
source-frame IPC hashes differed between manifests; this verifies the
evaluated data and forecasts, not that the two raw feeds were byte-identical.
Compact scorecards:
[pooled](methods-pooled.csv), [by season](methods-by-season.csv),
[paired MAE differences versus linear SVR](methods-paired-mae.csv),
[exact versus approximate RBF in 2014](methods-exact-rbf-2014.csv),
[trade, role and previous-absence diagnostics](methods-role-audit.csv) and
[source and output hashes](manifest.json).
