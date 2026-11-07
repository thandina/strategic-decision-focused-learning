# Conservation Security Game (Etosha): code behind Table 1 and Figure 2

This folder is the code that produced the conservation-game column of Table 1
and the per-day utilities plotted in Figure 2 of *Strategic Decision Focused
Learning* (NeurIPS 2026). Four models (PTO, ZOC, KKT and Hybrid, the paper's
Algorithm 1) predict daily elephant counts in 9 spatial cells of Etosha National
Park; each prediction is scored by the defender's utility in a Stackelberg
security game played against the true counts.

## Requirements

Python 3.10 or newer (3.11 tested) with `torch` (2.2 or newer), `numpy`, `pandas`, `scipy` and
`scikit-learn`. KKT and Hybrid also need `cvxpy` and `cvxpylayers` (they solve
with SCS); without them the two modules still import, but training them raises
`ImportError`. The device is chosen automatically (CUDA, then Apple MPS, then CPU);
set `SDFL_DEVICE=cpu` to force the CPU, which is advisable on a Mac (see the
top-level README on MPS).

## Data

The training file `data_ETOSHA_CELL_9.csv` is not shipped with the code. Put it
at `../data/data_ETOSHA_CELL_9.csv` relative to this folder, or pass `--data`.
It has one row per (day, cell) with columns `time_period, cell_id,
elephant_count, day_of_week, month, week_of_year`: 9 cells, 2008-10-28 to
2014-03-28. The loader adds 6 lagged counts per cell, holds out every day from
2013-02-24 as the test set (396 days) and uses the earliest 15% of the remaining
days as the validation set.

## Commands

```bash
# Table 1: all four models, seeds 1-5 in parallel, 500 epochs (hours on a GPU box)
python train.py

# A quick subset
python train.py --models pto,zoc --seeds 1 2 --epochs 50 --out results_quick

# Figure 2 input: per-day utilities re-solved from the saved predictions
python reconstruct_per_window.py            # reads and writes results/
```

`train.py` options: `--models pto,zoc,kkt,hybrid`, `--seeds 1 2 3 4 5`,
`--epochs 500`, `--data PATH`, `--out DIR` (default `results/`), `--workers N`
(default: one process per seed, capped at the CPU count). Both scripts can be
run from any working directory.

## Outputs

Written to `--out` with the same names as the paper's stored results
(`label` is `pto`, `zo3` for ZOC, `kkt` or `hybrid`):

| File | Content |
|---|---|
| `{label}_gsg_results.json` | `{"<seed>": {"utility": u, "mae": m}}`; Table 1 is the mean and sample standard deviation over seeds |
| `{label}_gsg_seed{s}_preds.csv` | test predictions (`time_period, cell_id, elephant_count`) |
| `{label}_gsg_epoch_maes.json` | test MAE of the stored checkpoint at epochs 50, 100, ..., 500 |
| `per_window_utility_from_preds_gsg.csv` | from `reconstruct_per_window.py`: one row per (day, model, seed) |

`utility` is the sum of the per-day defender utilities over the test days divided
by the number of test days; `mae` is the mean absolute error over all test rows.

## What to expect from a rerun

* The torch random generator is never seeded (as in the paper's run), so
  initial weights, dropout and input noise differ between runs; `--seeds` only
  fixes the numpy `RandomState` used by KKT and Hybrid for their inner split and
  day sampling. Seed-level numbers are therefore not bit-reproducible; the
  5-seed means should land within the standard deviations reported in Table 1.
* ZOC's inner backprop / evaluation split uses a fixed `RandomState(42)`
  (KKT and Hybrid use the run seed). This is intentional: it is what produced
  the paper's numbers.
* 0 is the best possible utility in this game (the attacker is deterred onto an
  empty cell). Hybrid and KKT reach it on most test days because the SSE places
  coverage at the attacker's indifference point under the predicted counts, so
  over-predicting the few occupied cells deters the attacker; treat the utility
  gap as a prediction-scale effect of a small sparse game rather than as better
  localisation.
* PTO and ZOC seeds take minutes; KKT and Hybrid seeds take hours (one
  cvxpylayers solve per training day and step).

## File map (original name -> this folder)

| Original | Here | Role |
|---|---|---|
| `experiments/_train_multi_seed.py` (gsg path) | `train.py` | multi-seed runner, Table 1 outputs |
| `ml_predict_vgsg2.py` (loader and metric only) | `data_pipeline.py` | data loading, features, split, MAE |
| `game_run_vgsg2.py` | `game.py` | payoff matrices, strategic evaluator |
| `solvers/nfg_sse_lp.py` | `solvers/nfg_sse_lp.py` | SSE LP (scipy/HiGHS) and attacker best response |
| `reconstruct_per_window_gsg_from_preds.py` | `reconstruct_per_window.py` | per-day utilities for Figure 2 |
| `utils.py` | `utils.py` | device selection |
| `models/regression_model_gsg2.py` | `models/pto.py` | PTO (`REGRESSION_MODEL_GSG2`) |
| `models/regression_zo3_model_gsg2.py` | `models/zoc.py` | ZOC (`REGRESSION_ZO3_MODEL_GSG2`) |
| `models/regression_e2e_kkt_gsg2.py` | `models/kkt.py` | KKT (`REGRESSION_E2E_KKT_GSG2`) |
| `models/regression_hybrid_gsg2.py` | `models/hybrid.py` | Hybrid, Algorithm 1 (`REGRESSION_HYBRID_GSG2`) |

Class names and all numerical behaviour are unchanged; only file names, import
paths, path handling and the command-line interface differ from the original.
