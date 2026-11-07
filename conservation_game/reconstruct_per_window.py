"""
Per-window (per-day) defender utility reconstructed from the saved test
predictions: the data behind the conservation panel of Figure 2.

Original file: reconstruct_per_window_gsg_from_preds.py.

train.py stores only per-seed totals in {label}_gsg_results.json, but it also
saves every seed's test predictions ({label}_gsg_seed{s}_preds.csv). This
script rebuilds the test split with data_pipeline.load_and_prep_gsg2, re-solves
the game for every test day from those predictions with the same evaluator
train.py used (game.evaluate_game_performance, collect_snapshots=True) and
checks that sum / n_days reproduces each seed's utility in the results json
(the paper's stored run matched to within 3e-10).

Output: <results>/per_window_utility_from_preds_gsg.csv with columns
        time_period, defender_utility, model, seed, run_id
        (model is pto / zoc / kkt / hybrid; the monthly means of
        defender_utility per model and seed are what Figure 2 plots).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_pipeline import load_and_prep_gsg2
from game import evaluate_game_performance

# CLI name -> prefix of the files written by train.py (ZOC files are "zo3_*").
PREFIX = {"pto": "pto", "zoc": "zo3", "kkt": "kkt", "hybrid": "hybrid"}
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_DATA = HERE.parent / "data" / "data_ETOSHA_CELL_9.csv"
DEFAULT_RESULTS = HERE / "results"
OUT_NAME = "per_window_utility_from_preds_gsg.csv"
# Value of the run_id column. Kept equal to the identifier used in the paper's
# stored per-window file so that the two files can be compared directly.
RUN_ID = "final_code_download_table_run"


def main(argv=None):
    """
    Re-solve the game per test day for every (model, seed) prediction file in
    --results, print the per-seed check against the results json and write
    the per-window CSV into --results. Side effects: one CSV written, stdout.
    """
    ap = argparse.ArgumentParser(
        description="Per-day defender utilities from the saved test predictions (Figure 2 input)."
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="path to data_ETOSHA_CELL_9.csv (default: ../data/data_ETOSHA_CELL_9.csv next to this folder)")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="directory holding train.py's outputs; the CSV is written there (default: results/)")
    ap.add_argument("--models", default="pto,zoc,kkt,hybrid",
                    help="comma-separated subset of pto,zoc,kkt,hybrid (default: all four)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="seeds to reconstruct (default: 1 2 3 4 5)")
    args = ap.parse_args(argv)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in PREFIX]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {list(PREFIX)}")
    res = Path(args.results).resolve()
    out = res / OUT_NAME

    _, _, test_set, _, _ = load_and_prep_gsg2(str(Path(args.data).resolve()))
    n_periods = test_set["time_period"].nunique()
    print(f"\ntest periods = {n_periods}")
    records, max_abs_diff = [], 0.0
    for label in model_keys:
        short = PREFIX[label]
        ref = json.loads((res / f"{short}_gsg_results.json").read_text())
        for s in args.seeds:
            preds = pd.read_csv(res / f"{short}_gsg_seed{s}_preds.csv")
            preds["time_period"] = pd.to_datetime(preds["time_period"])
            preds["cell_id"] = preds["cell_id"].astype(str)
            # Same evaluator and test split as train.py; snaps holds one record per day.
            total, snaps = evaluate_game_performance(preds, test_set, collect_snapshots=True)
            util = float(total) / n_periods
            ref_u = ref[str(s)]["utility"]
            d = abs(util - ref_u)
            max_abs_diff = max(max_abs_diff, d)
            print(f"{label:7s} seed {s}: reconstructed {util:.6f}  results.json {ref_u:.6f}  "
                  f"diff {d:.2e}  windows {len(snaps)}")
            for r in snaps:
                records.append({
                    "time_period": pd.Timestamp(r["time_period"]).strftime("%Y-%m-%d"),
                    "defender_utility": r["defender_utility"],
                    "model": label, "seed": s, "run_id": RUN_ID,
                })
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nmax |reconstructed - results.json| over {len(model_keys) * len(args.seeds)} model-seeds = {max_abs_diff:.2e}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
