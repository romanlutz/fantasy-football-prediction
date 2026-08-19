# Fantasy Football Prediction

This project revisits the original 2015 experiment that used support vector
regression and neural networks to predict weekly NFL quarterback fantasy
scores. The original implementation is preserved at the
[`legacy-2015`](https://github.com/romanlutz/fantasy-football-prediction/tree/legacy-2015)
tag.

The modern pipeline uses Python 3 and
[`nflreadpy`](https://github.com/nflverse/nflreadpy) instead of the abandoned
`nflgame` package. Core play-by-play, player, team, schedule, and roster data
comes from automated
[`nflverse-data`](https://github.com/nflverse/nflverse-data/releases)
releases.

## Setup

Install the pinned runtime and development dependencies with
[`uv`](https://docs.astral.sh/uv/):

```console
uv sync
```

## Rebuild the paper datasets

```console
uv run python create_datasets.py
```

This loads regular-season data from 2009 through 2014, trains on games from
2010–2013, and creates a 2014 test set. It writes:

- `train.npy`
- `test.npy`
- `dataset-manifest.json`

The manifest records the nflreadpy version, source, generation time, row
counts, and output checksums. Use the command-line options to evaluate a newer
season:

```console
uv run python create_datasets.py --history-start 2018 --train-start 2019 --test-year 2025
```

Data generation downloads full play-by-play files because the original feature
set included passing and rushing two-point *attempts*. Subsequent runs can use
nflreadpy's filesystem cache:

```powershell
$env:NFLREADPY_CACHE = "filesystem"
uv run python create_datasets.py
```

## Train the models

Run the support vector regressor:

```console
uv run python models.py
```

Run the neural-network experiments:

```console
uv run python neural_net.py
```

PyBrain is no longer maintained, so the neural-network implementation now uses
scikit-learn's `MLPRegressor`. Both model scripts fit feature scaling on the
training set only.

The reported mean relative error is the mean absolute error divided by the
absolute observed score. Games with an observed score of zero are excluded
because relative error is undefined for them. This corrects the original
implementation, which divided by the prediction, so MRE values are not directly
comparable with the 2015 results.

## Feature layout

Each generated row retains the original 37-column shape:

| Columns | Values |
|---|---|
| 0–1 | GSIS quarterback ID and display name |
| 2–3 | Age on game day and completed NFL seasons |
| 4–15 | Quarterback statistics from the previous game |
| 16–27 | Average quarterback statistics from the previous ten games |
| 28–31 | Opposing defense statistics from its previous game |
| 32–35 | Average opposing defense statistics from its previous ten games |
| 36 | Standard-scoring fantasy points for the current game |

The nflverse tables provide explicit player teams and opponents, eliminating
the original single-team-per-season inference. Defense-allowed statistics are
constructed from the opponent's offensive totals.

## Data licensing and reproducibility

`nflreadpy` is MIT-licensed. Most nflverse data is CC BY 4.0; specialized FTN
data uses CC BY-SA 4.0. This project uses the standard player, team, schedule,
roster, and play-by-play datasets rather than FTN charting.

nflverse release assets may be corrected or refreshed after publication. For
a reproducible experiment, preserve the downloaded source files or cache
alongside `dataset-manifest.json`; output checksums alone establish the
generated dataset's identity but cannot reconstruct changed upstream inputs.

## Tests

The tests use local fixture frames and do not download NFL data:

```console
uv run pytest
```
