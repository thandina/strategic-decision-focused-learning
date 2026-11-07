# Infrastructure Security Game (NYC subway): Table 1 and Figure 2(b)

Code for the Infrastructure Security Game experiment of *Strategic Decision Focused Learning*
(Handina, Diao, Wierman, Mazumdar; NeurIPS 2026): 123 Manhattan subway station complexes,
8-hour windows, training/validation on 2022-2023, test on the 1,098 windows of 2024.
The paper's numbers come from run id `paper_20260630_030550` of exactly this code; only file
names, imports, path handling and comments differ from the files that ran (map below).

## What it reproduces

| Paper | Produced by |
|---|---|
| Table 1, Infrastructure game rows: utility (mean per-window defender utility over the test year) and MAE, mean and sample std over seeds 1-5 | `train.py` (one model, one seed), launched for all models and seeds by `launch_all_seeds.sh` |
| Figure 2(b): monthly mean defender utility over the test year, per model | `reconstruct_per_window.py` on the checkpoints of that run (one utility per window, model and seed); the monthly aggregation and the plot are not part of this folder |

Five-seed means of the paper run (per-window utility / MAE): PTO -13.611 +/- 0.210 / 0.409 +/- 0.009;
ZOC -13.160 +/- 0.355 / 0.559 +/- 0.062; KKT -12.829 +/- 0.465 / 1.104 +/- 0.467;
Hybrid -12.578 +/- 0.383 / 0.736 +/- 0.028. The utility of the true ridership (oracle) is -4.553
per window. Reruns are not bit-identical (see Notes); five-seed means should land within the
standard deviations above.

## Requirements

Python 3.10 or newer (3.11 tested) with `torch` (>= 2.2), `numpy`, `pandas`, `scipy`, `scikit-learn`, and
`cvxpy` + `cvxpylayers` for `kkt_v3` / `hybrid_v3` (training, and reloading their checkpoints).
`pto` and `zoc` run without cvxpy. The paper run used a plain venv with numpy 2.x. The compute
device is chosen automatically (CUDA, else Apple MPS, else CPU; `utils.py`); set
`SDFL_DEVICE=cpu` to force the CPU, which is what the paper run used and is advisable on a
Mac (see the top-level README on MPS).

## Data

`../data/data_MTA_8H_Manhattan.csv` (or `--data PATH`): columns `time_period` (window start,
00:00 / 08:00 / 16:00), `station_complex_id`, `ridership` (thousands of riders in the window);
123 Manhattan station complexes, 2022-01-01 16:00 to 2024-12-31 16:00 (403,469 rows). Built from
the public "MTA Subway Hourly Ridership: 2020-2024" table (data.ny.gov): Manhattan rows from
2022-01-01, summed per station and 8-hour bin, divided by 1,000. Features (six lags, month,
day of week, hour, one-hot station) and the split are built in `data_pipeline.py`.

## Commands

Activate the Python environment first. Scripts work from any directory.

```bash
# one model, one seed (what launch_all_seeds.sh runs 20 times)
MTA_EVAL_WORKERS=6 python train.py --model pto    --seed 1 --epochs 500 --run_id RUN
MTA_EVAL_WORKERS=6 python train.py --model zoc    --seed 1 --epochs 500 --run_id RUN
MTA_EVAL_WORKERS=6 python train.py --model kkt_v3    --seed 1 --epochs 500 --top_k 20 --lp_workers 6 --huber_weight_end 3.0 --run_id RUN
MTA_EVAL_WORKERS=6 python train.py --model hybrid_v3 --seed 1 --epochs 500 --top_k 20 --lp_workers 6 --huber_weight_end 3.0 --run_id RUN
#   --data PATH   ridership CSV (default ../data/data_MTA_8H_Manhattan.csv)
#   --out DIR     results root (default results/ next to train.py); the run id folder goes under it

# the full Table 1 run: 4 models x 5 seeds as 20 background processes (needs a many-core machine;
# about 2.7 h per hybrid_v3 seed on a 32-core box with everything running, minutes per pto/zoc seed)
bash launch_all_seeds.sh                 # prints RUN_ID; logs in logs/, results in results/RUN_ID/

# per-window utilities (Figure 2(b)) from a finished run's checkpoints
python reconstruct_per_window.py --run_dir results/RUN_ID [--data PATH] [--out CSV]
```

`MTA_EVAL_WORKERS` is the size of the process pool that solves the test windows (and the inner
evaluation windows during ZOC / KKT / Hybrid training); `--lp_workers` is the number of threads
for the KKT / Hybrid target enumeration. Both change speed only, up to solver thread
nondeterminism.

## Outputs

```
results/<run_id>/
  <model>_seed_<s>.json        model, seed, epochs, top_k, utility (sum over the 1,098 test
                               windows), per_window (= utility / 1098: the Table 1 number),
                               mae (over all test rows), elapsed_s (training time), run_id
  summary.json                 per model: n_seeds, per_window_mean/std, mae_mean/std (sample
                               std, ddof = 1), seeds, individual records; rewritten after every seed
  checkpoints/<model>_seed<s>.pt   model_state_dict, scaler, params (the hyperparameters that
                               ran), feature_cols
  per_window_utility_reconstructed.csv   from reconstruct_per_window.py:
                               time_period, defender_utility, model, seed
```

## Model keys

| key | paper | file / class | objective and selection rule | hyperparameters that ran |
|---|---|---|---|---|
| `pto` | PTO | `models/pto.py` `REGRESSION_MODEL_MTA_V2` | Huber loss only, full batch; final-epoch weights | hidden 256, lr 0.01, noise_std 0.01, 500 epochs |
| `zoc` | ZOC | `models/zoc.py` `REGRESSION_ZO3_MODEL_MTA_V2` | Huber blocks of 50 epochs; each block accepted or reverted on the game utility of an inner 15% evaluation split; best accepted block | as PTO + n_store 50, split 0.85 (fixed `RandomState(42)`) |
| `kkt_v3` | KKT | `models/kkt.py` `REGRESSION_E2E_KKT_MTA_V3` | -utility through the differentiable SSE LP (cvxpylayers) over the 20 stations with the largest predicted ridership + huber_weight x Huber over all 123 stations; huber_weight 5.0 -> 3.0 linearly over epochs 1-300, then 3.0; best inner-split utility (checked every 5 epochs) restored | hidden 256, lr 5e-4, batch 8 windows, 32 windows/epoch, top_k 20, split 0.85, SCS eps 1e-4, lp_workers 6, 500 epochs |
| `hybrid_v3` | Hybrid | `models/hybrid.py` `REGRESSION_HYBRID_MTA_V3` | blocks of 50 epochs alternating Huber (odd; block 1 is a warm-up) and the KKT step (even), accept-or-revert after each block as ZOC; Huber weight clock starts after the warm-up: 5.0 at epoch 51 -> 3.0 at epoch 350, then 3.0 | as KKT with noise_std 0.01, n_store 50, SCS eps 1e-2 |

All four share the MLP 132 -> 256 -> 128 -> 1 (LayerNorm, SiLU, Dropout 0.2, ReLU output),
AdamW with weight decay 0.01, ReduceLROnPlateau on the validation Huber loss, and the same
test-time evaluation: SSE strategy from the predicted payoffs, attacker best response under
the true payoffs (`game.py`, `solvers/nfg_sse_lp.py`).

## File map (paper run -> this folder)

| original (`neurips2026_Additional_experiments/`) | here |
|---|---|
| `run_paper.py` | `train.py` (four paper models only; `--data`, `--out` added) |
| `launch_paper.sh` | `launch_all_seeds.sh` (four paper models only; no environment activation) |
| `ml_predict_vmta_v2.py` | `data_pipeline.py` (loader and metrics only; `data_path` argument) |
| `game_run_vmta_final.py` | `game.py` |
| `reconstruct_per_window.py` | `reconstruct_per_window.py` (run directory and data path as arguments) |
| `utils.py` | `utils.py` |
| `solvers/nfg_sse_lp.py` | `solvers/nfg_sse_lp.py` (unused non-MTA variants and self-test removed) |
| `models/regression_model_mta_v2.py` | `models/pto.py` |
| `models/regression_zo3_model_mta_v2.py` | `models/zoc.py` |
| `models/regression_e2e_kkt_mta_v3.py` | `models/kkt.py` |
| `models/regression_hybrid_mta_v3.py` | `models/hybrid.py` |

Model keys were kept (`kkt_v3` = KKT, `hybrid_v3` = Hybrid in the paper) because the shipped
result files and checkpoints use them.

## Notes

* Seeds. `--seed` sets `random_seed`, which `kkt_v3` / `hybrid_v3` use for their inner 85/15
  split and per-epoch window sampling; `pto` / `zoc` do not read it (ZOC's inner split uses a
  fixed `RandomState(42)`). torch's RNG is never seeded, so weight initialisation, dropout and
  the input noise differ between runs even at a fixed seed. The train/validation/test split is
  fixed (`np.random.seed(42)` inside the loader).
* A seed whose json already exists in the run folder is skipped by `train.py`.
* KKT's cvxpylayers solve uses SCS tolerance 1e-4 (hard-coded); Hybrid's uses its
  `solver_eps` = 1e-2. A failed cvxpylayers solve contributes zero utility (no gradient) for
  that window; the counts are printed during training.
* The paper run's `kkt_v3` checkpoints record `lp_workers` = 14 while the launch script passes
  6; this affects speed only.
* Reloading a `kkt_v3` / `hybrid_v3` checkpoint instantiates its class, which requires cvxpy
  and cvxpylayers even though inference does not use the LP layer.
* The loader also accepts a raw hourly export with a `transit_timestamp` column, which it sums
  into 4-hour bins; that branch is kept verbatim from the original code and is not the paper's
  8-hour setting. Use the 8-hour file.
