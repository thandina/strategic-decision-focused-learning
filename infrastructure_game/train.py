"""
train.py -- single-model, single-seed runner for the Infrastructure Security
Game of "Strategic Decision Focused Learning" (Table 1, and the checkpoints
behind Figure 2(b)). Originally run_paper.py; the paper's numbers are run id
paper_20260630_030550, launched by launch_all_seeds.sh with the flags below.

Model keys (name in the paper in brackets):
    pto       [PTO]     models/pto.py     REGRESSION_MODEL_MTA_V2
    zoc       [ZOC]     models/zoc.py     REGRESSION_ZO3_MODEL_MTA_V2
    kkt_v3    [KKT]     models/kkt.py     REGRESSION_E2E_KKT_MTA_V3
    hybrid_v3 [Hybrid]  models/hybrid.py  REGRESSION_HYBRID_MTA_V3

Usage:
    MTA_EVAL_WORKERS=6 python train.py --model pto    --seed 1 --epochs 500 --run_id RUN
    MTA_EVAL_WORKERS=6 python train.py --model kkt_v3 --seed 1 --epochs 500 \
        --top_k 20 --lp_workers 6 --huber_weight_end 3.0 --run_id RUN
    --data PATH  ridership CSV (default ../data/data_MTA_8H_Manhattan.csv next
                 to this package); --out DIR results root (default results/
                 next to this script). Works from any working directory.

Per call the script loads the split (data_pipeline.load_and_prep_mta), trains
the model, predicts the 1,098 test windows of 2024, solves the game on the
predictions (game.evaluate_game_performance, a process pool of
MTA_EVAL_WORKERS workers; speed only), computes the MAE over all test rows
after merging predictions and truth on (time_period, station_complex_id), and
writes
    <out>/<run_id>/<model>_seed_<seed>.json  model, seed, epochs, top_k, utility
                                             (sum over windows), per_window
                                             (= utility / 1098: the Table 1
                                             number), mae, elapsed_s, run_id
    <out>/<run_id>/checkpoints/<model>_seed<seed>.pt
                                             model_state_dict, scaler, params,
                                             feature_cols (reloaded by
                                             reconstruct_per_window.py)
    <out>/<run_id>/summary.json              per model: mean and sample std
                                             (ddof = 1) of per_window and mae
                                             over the seed files present
A seed whose json already exists is skipped (summary is still rewritten).

Hyperparameters that ran (also stored in each checkpoint's params): the
model defaults from get_runtime_hyperparams() with epochs = 500 and
random_seed = seed for every model, and for kkt_v3 / hybrid_v3 additionally
top_k = 20, max_periods_per_epoch = 32, lp_workers = 6, huber_weight = 5.0,
huber_weight_end = 3.0, huber_anneal_epochs = 300.

Seeding: --seed becomes params["random_seed"] (read by kkt_v3 and hybrid_v3
for their inner 85/15 split and per-epoch window sampling; pto and zoc do not
read it) and seeds numpy's global RNG, from which no model draws. torch's RNG
is not seeded anywhere, so weight initialisation, dropout and the Gaussian
input noise vary from run to run even at a fixed seed. The data split itself
is fixed (np.random.seed(42) inside the loader).
"""

import os, sys, json, argparse, time, importlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_pipeline import load_and_prep_mta, DEFAULT_DATA_PATH
from game          import evaluate_game_performance

DEFAULT_RESULTS_ROOT = ROOT / "results"

# model key -> (module under models/, class name). kkt_v3 is "KKT" and
# hybrid_v3 is "Hybrid" in the paper; the keys are kept because the shipped
# result files use them.
MODEL_MAP = {
    "pto":       ("models.pto",    "REGRESSION_MODEL_MTA_V2"),
    "zoc":       ("models.zoc",    "REGRESSION_ZO3_MODEL_MTA_V2"),
    "kkt_v3":    ("models.kkt",    "REGRESSION_E2E_KKT_MTA_V3"),
    "hybrid_v3": ("models.hybrid", "REGRESSION_HYBRID_MTA_V3"),
}

# Row order of the printed summary table only.
MODEL_ORDER = ["pto", "zoc", "kkt_v3", "hybrid_v3"]


def parse_args():
    """Command line of one training run (see the module docstring)."""
    p = argparse.ArgumentParser()
    p.add_argument("--model",            type=str,   required=True, choices=list(MODEL_MAP))
    p.add_argument("--seed",             type=int,   default=1)
    p.add_argument("--epochs",           type=int,   default=500)
    p.add_argument("--top_k",            type=int,   default=20,
                   help="Stations entering LP per step (kkt_v3 and hybrid_v3 only)")
    p.add_argument("--lp_workers",       type=int,   default=0,
                   help="Parallel LP threads (kkt_v3 and hybrid_v3 only; 0=auto)")
    p.add_argument("--huber_weight_end", type=float, default=3.0,
                   help="Final huber weight after annealing (kkt_v3 and hybrid_v3 only)")
    p.add_argument("--run_id",           type=str,   default=None)
    p.add_argument("--data",             type=str,   default=str(DEFAULT_DATA_PATH),
                   help="Ridership CSV (default: ../data/data_MTA_8H_Manhattan.csv relative to this script)")
    p.add_argument("--out",              type=str,   default=str(DEFAULT_RESULTS_ROOT),
                   help="Results root; the run id subfolder is created under it (default: results/ next to this script)")
    return p.parse_args()


def _write_summary(out_dir):
    """
    Collate every <model>_seed_<s>.json in out_dir: print a per-model table
    (mean +/- sample std, ddof = 1 when n > 1) and write summary.json with
    n_seeds, per_window_mean/std, mae_mean/std, seeds and the individual
    records per model. Rewritten after every seed; oracle_per_window is the
    utility of the true ridership, recorded for reference.
    """
    files   = sorted(out_dir.glob("*_seed_*.json"))
    results = [json.loads(f.read_text()) for f in files]
    if not results:
        return

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    lines = ["", "=" * 62,
             f"  Paper Results  |  run={out_dir.name}",
             f"  {'Model':<12} {'n':>2}  {'Per-window':>14}  {'MAE':>10}",
             "  " + "-" * 46]

    for m in MODEL_ORDER:
        rs = by_model.get(m)
        if not rs:
            continue
        pws  = [r["per_window"] for r in rs]
        maes = [r["mae"] for r in rs]
        n    = len(pws)
        ddof = min(1, n - 1)
        lines.append(
            f"  {m:<12} {n:>2}  "
            f"{np.mean(pws):>+8.4f} ± {np.std(pws, ddof=ddof):<5.4f}  "
            f"{np.mean(maes):>8.4f} ± {np.std(maes, ddof=ddof):<5.4f}"
        )
    lines.append("=" * 62)
    for line in lines:
        print(line)

    summary = {"run_id": out_dir.name, "oracle_per_window": -4.553, "by_model": {}}
    for m, rs in by_model.items():
        pws  = [r["per_window"] for r in rs]
        maes = [r["mae"] for r in rs]
        n    = len(pws)
        ddof = min(1, n - 1)
        summary["by_model"][m] = {
            "n_seeds":          n,
            "per_window_mean":  float(np.mean(pws)),
            "per_window_std":   float(np.std(pws, ddof=ddof)),
            "mae_mean":         float(np.mean(maes)),
            "mae_std":          float(np.std(maes, ddof=ddof)),
            "seeds":            sorted(r["seed"] for r in rs),
            "individual":       rs,
        }
    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Summary → {out_dir / 'summary.json'}")


def main():
    """Train one model for one seed, evaluate it, write the result files (module docstring)."""
    args = parse_args()

    mod_file, class_name = MODEL_MAP[args.model]
    mod      = importlib.import_module(mod_file)
    ModelCls = getattr(mod, class_name)

    run_id  = args.run_id or f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_file = out_dir / f"{args.model}_seed_{args.seed}.json"
    if seed_file.exists():
        print(f"[{args.model}] seed_{args.seed} already exists — skipping.")
        _write_summary(out_dir)
        return

    print(f"\n[{args.model}] run_id={run_id} | seed={args.seed} | epochs={args.epochs}")

    train_set, val_set, test_set, station_ids, feature_cols = load_and_prep_mta(args.data)
    n_windows = int(test_set["time_period"].nunique())

    # Model defaults, then the overrides of the paper run.
    params = mod.get_runtime_hyperparams()
    params["epochs"]      = args.epochs
    params["random_seed"] = args.seed

    if args.model in ("kkt_v3", "hybrid_v3"):
        params["top_k"]                 = args.top_k
        params["max_periods_per_epoch"] = 32

    if args.model in ("kkt_v3", "hybrid_v3"):
        params["lp_workers"]          = args.lp_workers
        params["huber_weight"]        = 5.0
        params["huber_weight_end"]    = args.huber_weight_end
        params["huber_anneal_epochs"] = 300

    np.random.seed(args.seed)
    model = ModelCls(params, feature_cols, window_size=6)

    t0 = time.time()
    model.train(train_set, val_set)
    elapsed = time.time() - t0

    # Test: predict every 2024 row, solve the game per window (sum of the
    # defender's true utilities), MAE over all test rows.
    preds   = model.generate(test_set)
    utility = evaluate_game_performance(preds, test_set)

    merged = preds.merge(
        test_set[["time_period", "station_complex_id", "ridership"]],
        on=["time_period", "station_complex_id"],
        suffixes=("_pred", "_actual"),
    )
    mae        = float(np.mean(np.abs(merged["ridership_pred"] - merged["ridership_actual"])))
    per_window = utility / n_windows

    print(f"\n[{args.model}] seed={args.seed} | utility={utility:.4f} | "
          f"per_window={per_window:.4f} | MAE={mae:.4f} | {elapsed:.0f}s")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.model.state_dict(),
        "scaler":           model.scaler,
        "params":           params,
        "feature_cols":     feature_cols,
    }, ckpt_dir / f"{args.model}_seed{args.seed}.pt")

    result = {
        "model":      args.model,
        "seed":       args.seed,
        "epochs":     args.epochs,
        "top_k":      params.get("top_k", None),
        "utility":    float(utility),
        "per_window": float(per_window),
        "mae":        float(mae),
        "elapsed_s":  float(elapsed),
        "run_id":     run_id,
    }
    seed_file.write_text(json.dumps(result, indent=2))
    print(f"[{args.model}] Result → {seed_file}")

    _write_summary(out_dir)


if __name__ == "__main__":
    main()
