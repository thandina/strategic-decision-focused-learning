"""
reconstruct_per_window.py -- the per-window defender utilities behind
Figure 2(b) of "Strategic Decision Focused Learning".

train.py records only each seed's total utility. This script reloads every
checkpoint of a run directory (checkpoints/<model>_seed<s>.pt for the models
pto, zoc, kkt_v3, hybrid_v3 and seeds 1..5; missing ones are reported and
skipped), regenerates the test-set predictions (deterministic: eval mode plus
the saved scaler) and re-solves the game for each of the 1,098 test windows
with the test-time solver (game._solve_snapshot_utility), writing one row per
(window, model, seed):

    time_period, defender_utility, model, seed

Usage:
    python reconstruct_per_window.py --run_dir results/<run_id> \
        [--data ../data/data_MTA_8H_Manhattan.csv] \
        [--out  <run_dir>/per_window_utility_reconstructed.csv]

The sum over windows of a (model, seed) series equals that seed's `utility`
in <model>_seed_<s>.json up to solver tolerance (the paper run matched to
~1e-10). The script resumes: (model, seed) pairs already complete in the
output CSV are kept and skipped. Windows are solved by a pool of 9 worker
processes (speed only). Progress lines go to stdout and are appended to
reconstruct_progress.txt next to the output CSV.

Loading a checkpoint instantiates its model class, so cvxpy + cvxpylayers
must be installed for the kkt_v3 / hybrid_v3 checkpoints even though no LP
layer is used for inference. Checkpoints saved under numpy 2.x / torch >= 2.3
(as the paper run was) need the same or newer to unpickle.
"""
import sys, os, time, importlib, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_pipeline import load_and_prep_mta, DEFAULT_DATA_PATH
from game import gen_mta_utility_matrices, _solve_snapshot_utility

# model key -> (module under models/, class name); same keys as train.py.
MODEL_MAP = {
    "pto":       ("models.pto",    "REGRESSION_MODEL_MTA_V2"),
    "zoc":       ("models.zoc",    "REGRESSION_ZO3_MODEL_MTA_V2"),
    "kkt_v3":    ("models.kkt",    "REGRESSION_E2E_KKT_MTA_V3"),
    "hybrid_v3": ("models.hybrid", "REGRESSION_HYBRID_MTA_V3"),
}

_progress_fh = None   # opened in main() once the output location is known


def log(msg):
    """Print a time-stamped line and append it to the progress file."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _progress_fh.write(line + "\n")


def parse_args():
    """Command line: run directory (required), data CSV and output CSV (optional)."""
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run directory written by train.py (contains checkpoints/)")
    p.add_argument("--data",    type=str, default=str(DEFAULT_DATA_PATH),
                   help="Ridership CSV (default: ../data/data_MTA_8H_Manhattan.csv relative to this script)")
    p.add_argument("--out",     type=str, default=None,
                   help="Output CSV (default: <run_dir>/per_window_utility_reconstructed.csv)")
    return p.parse_args()


def main():
    """Reconstruct the per-window utilities of every checkpoint in --run_dir (module docstring)."""
    global _progress_fh
    args     = parse_args()
    run_dir  = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    out_csv  = Path(args.out).resolve() if args.out else run_dir / "per_window_utility_reconstructed.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _progress_fh = open(out_csv.parent / "reconstruct_progress.txt", "a", buffering=1)

    t_start = time.time()
    log("Loading MTA data / rebuilding test split ...")
    train_set, val_set, test_set, station_ids, feature_cols = load_and_prep_mta(args.data)
    n_windows = int(test_set["time_period"].nunique())
    log(f"test windows={n_windows}, test rows={len(test_set)}")

    # Resume: keep rows of (model, seed) pairs that already have every window.
    all_records = []
    already_done = set()
    if out_csv.exists():
        prior = pd.read_csv(out_csv)
        counts = prior.groupby(["model", "seed"]).size()
        for (model_key, seed), cnt in counts.items():
            if cnt >= n_windows:
                already_done.add((model_key, seed))
        # Only keep rows for (model,seed) pairs we're treating as complete,
        # and only for models still in MODEL_MAP.
        prior = prior[prior.apply(
            lambda r: (r["model"], r["seed"]) in already_done and r["model"] in MODEL_MAP,
            axis=1,
        )]
        all_records = prior.to_dict("records")
        log(f"Resuming: found {len(already_done)} already-complete (model,seed) pairs "
            f"in existing CSV, keeping {len(all_records)} rows.")

    checkpoints = []
    for model_key, (mod_file, class_name) in MODEL_MAP.items():
        for seed in range(1, 6):
            if (model_key, seed) in already_done:
                continue
            ckpt_path = ckpt_dir / f"{model_key}_seed{seed}.pt"
            if ckpt_path.exists():
                checkpoints.append((model_key, mod_file, class_name, seed, ckpt_path))
            else:
                log(f"WARNING missing checkpoint {ckpt_path}")

    total_ckpts = len(checkpoints)
    total_windows_all = total_ckpts * n_windows
    completed_windows = 0
    log(f"{len(already_done)} checkpoints already done. "
        f"{total_ckpts} remaining -> {total_windows_all} window-solves. "
        f"Using 9 worker processes.")

    sanity = []

    times_sorted = sorted(test_set["time_period"].unique())

    with ProcessPoolExecutor(max_workers=9) as pool:
        for idx, (model_key, mod_file, class_name, seed, ckpt_path) in enumerate(checkpoints, 1):
            t0 = time.time()
            mod = importlib.import_module(mod_file)
            ModelCls = getattr(mod, class_name)

            # Rebuild the model exactly as trained: saved params, feature
            # columns, weights and the fitted scaler; eval mode (dropout off).
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = ModelCls(ckpt["params"], ckpt["feature_cols"], window_size=6)
            model.model.load_state_dict(ckpt["model_state_dict"])
            model.scaler = ckpt["scaler"]
            model.model.eval()

            preds = model.generate(test_set)

            # Same payload construction as game.evaluate_game_performance:
            # per window, truth and forecast sorted by station id.
            payloads_with_t = []
            for t in times_sorted:
                ta = test_set[test_set["time_period"] == t].sort_values("station_complex_id")
                tp = preds[preds["time_period"] == t].sort_values("station_complex_id")
                D_gen, A_gen   = gen_mta_utility_matrices(tp["ridership"])
                D_real, A_real = gen_mta_utility_matrices(ta["ridership"])
                payloads_with_t.append((t, D_gen, A_gen, D_real, A_real))

            futures = {
                pool.submit(_solve_snapshot_utility, (D_gen, A_gen, D_real, A_real)): t
                for t, D_gen, A_gen, D_real, A_real in payloads_with_t
            }

            results_by_t = {}
            n_done_this_ckpt = 0
            report_every = max(1, n_windows // 4)
            for fut in as_completed(futures):
                t = futures[fut]
                results_by_t[t] = fut.result()
                n_done_this_ckpt += 1
                completed_windows += 1
                if n_done_this_ckpt % report_every == 0:
                    elapsed_total = time.time() - t_start
                    frac = completed_windows / total_windows_all
                    eta_min = (elapsed_total / frac - elapsed_total) / 60 if frac > 0 else float("nan")
                    log(f"  [{idx}/{total_ckpts}] {model_key} seed{seed}: "
                        f"{n_done_this_ckpt}/{n_windows} windows | "
                        f"overall {completed_windows}/{total_windows_all} ({frac*100:.1f}%) | "
                        f"elapsed={elapsed_total/60:.1f}m ETA={eta_min:.1f}m")

            # Rows are appended in completion order (as_completed); sort by
            # time_period downstream if a time series is needed.
            snapshots = [
                {"time_period": t, "defender_utility": u, "model": model_key, "seed": seed}
                for t, u in results_by_t.items()
            ]
            all_records.extend(snapshots)
            util_total = float(np.sum([r["defender_utility"] for r in snapshots]))
            per_window = util_total / n_windows
            sanity.append((model_key, seed, per_window))

            log(f"[{idx}/{total_ckpts}] {model_key} seed{seed} DONE "
                f"per_window={per_window:.4f} ({time.time()-t0:.0f}s)")

            # Checkpoint the CSV after every (model, seed) so the run can resume.
            pd.DataFrame(all_records).to_csv(out_csv, index=False)

    log("Sanity check vs stored summary.json (per-seed per_window):")
    for model_key, seed, pw in sanity:
        log(f"  {model_key:10s} seed{seed}: reconstructed={pw:.4f}")

    log(f"ALL DONE. Saved {len(all_records)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
