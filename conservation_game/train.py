"""
Multi-seed training runner for the conservation game: produces the numbers
of Table 1 (conservation game column) and the per-seed prediction files from
which reconstruct_per_window.py derives the Figure 2 series.

Original file: experiments/_train_multi_seed.py (its "gsg" path; the "mta"
path was a different, superseded code generation and is not included).

For each requested model, all requested seeds are trained in parallel
processes (one process per seed); each process loads the data, trains the
model for --epochs epochs, predicts the test set and computes
    utility = total defender utility over the test days / number of test days
    mae     = mean absolute error of the per-cell counts on the test rows,
then re-evaluates the MAE at every stored epoch checkpoint.

Outputs in --out (names identical to the paper's stored results):
    {label}_gsg_results.json     {"<seed>": {"utility": u, "mae": m}, ...}   (Table 1)
    {label}_gsg_seed{s}_preds.csv  test predictions: time_period, cell_id, elephant_count
    {label}_gsg_epoch_maes.json  {"<seed>": {"<epoch>": mae, ...}, ...}  (epoch curves)
with label = pto, zo3 (ZOC), kkt, hybrid.

Seeding, as in the paper's run: each worker calls np.random.seed(seed) and
passes random_seed = seed to the model. KKT and Hybrid use it (RandomState(seed)
for the backprop / D_eval split and the per-epoch sampling of days); ZOC uses
a fixed RandomState(42) for its split; PTO uses no seed. torch is never seeded,
so weight initialisation, dropout and input noise differ from run to run and
seed-level numbers are not bit-reproducible; the 5-seed means are expected to
land within the standard deviations reported in Table 1.
"""
import argparse
import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# CLI name -> (module, class). Class names are the original ones.
MODELS = {
    "pto":    ("models.pto",    "REGRESSION_MODEL_GSG2"),
    "zoc":    ("models.zoc",    "REGRESSION_ZO3_MODEL_GSG2"),
    "kkt":    ("models.kkt",    "REGRESSION_E2E_KKT_GSG2"),
    "hybrid": ("models.hybrid", "REGRESSION_HYBRID_GSG2"),
}
# Output-file prefixes. ZOC keeps its original prefix "zo3" so that the file
# names match the paper's stored results (zo3_gsg_results.json etc.).
OUTPUT_LABEL = {"pto": "pto", "zoc": "zo3", "kkt": "kkt", "hybrid": "hybrid"}
SUFFIX = "gsg"

DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_EPOCHS = 500
DEFAULT_DATA = HERE.parent / "data" / "data_ETOSHA_CELL_9.csv"
DEFAULT_OUT = HERE / "results"


# ---------------------------------------------------------------------------
# Worker function: must be top-level for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def _seed_gsg(args):
    """
    Train and evaluate one (model, seed) pair in a worker process.

    Input:   args = (model_key, seed, epochs, data_path).
    Returns: (seed, utility, mae, preds, epoch_maes) where utility is the
             per-day mean defender utility on the test set, mae the test MAE,
             preds the test-prediction DataFrame and epoch_maes
             {epoch: test MAE of the model's stored checkpoint at that epoch}.
    Side effects: training output on stdout; nothing written to disk.
    """
    model_key, seed, epochs, data_path = args
    import sys, importlib, numpy as np
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    np.random.seed(seed)
    from data_pipeline import load_and_prep_gsg2, calculate_gsg2_metrics, WINDOW_SIZE
    from game import evaluate_game_performance
    train_set, val_set, test_set, _, feature_cols = load_and_prep_gsg2(data_path)
    module_name, class_name = MODELS[model_key]
    mod = importlib.import_module(module_name)
    ModelClass = getattr(mod, class_name)
    params = mod.get_runtime_hyperparams()
    params["epochs"] = epochs
    params["random_seed"] = seed
    inst = ModelClass(params, feature_cols, WINDOW_SIZE)
    inst.train(train_set, val_set)

    # Final test evaluation
    preds = inst.generate(test_set)
    n_periods = test_set['time_period'].nunique()
    util = float(evaluate_game_performance(preds, test_set)) / n_periods
    mae, _ = calculate_gsg2_metrics(preds, test_set)

    # Epoch-checkpoint MAEs: reload each stored checkpoint, predict the test
    # set, then restore the final weights.
    epoch_maes = {}
    if hasattr(inst, 'epoch_checkpoints') and inst.epoch_checkpoints:
        orig_state = {k: v.detach().clone() for k, v in inst.model.state_dict().items()}
        for ep in sorted(inst.epoch_checkpoints):
            inst.model.load_state_dict(inst.epoch_checkpoints[ep])
            ep_preds = inst.generate(test_set)
            ep_mae, _ = calculate_gsg2_metrics(ep_preds, test_set)
            epoch_maes[ep] = float(ep_mae)
        inst.model.load_state_dict(orig_state)

    return seed, util, float(mae), preds, epoch_maes


# ---------------------------------------------------------------------------
# Parallel runner: one process per seed, all seeds of a model run at once
# ---------------------------------------------------------------------------

def _run_model(model_key, seeds, epochs, data_path, out_dir, n_workers):
    """
    Train all `seeds` of one model in parallel and write its three output
    files to out_dir. A seed whose worker raises is recorded with NaN utility
    and MAE (the traceback is printed) and has no predictions file.
    Seeds appear in the results JSON in completion order, as in the original.
    """
    module_name, class_name = MODELS[model_key]
    print(f"\n  {class_name} [{model_key}]  ({len(seeds)} seeds, {n_workers} parallel workers)")
    results = {}
    epoch_maes_all = {}   # {seed: {epoch: mae}}
    all_preds = {}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_seed_gsg, (model_key, s, epochs, data_path)): s for s in seeds}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                seed, util, mae, preds, epoch_maes = fut.result()
                results[seed] = {"utility": util, "mae": mae}
                epoch_maes_all[seed] = epoch_maes
                all_preds[seed] = preds
                print(f"    seed={seed}  utility={util:.4f}  mae={mae:.4f}  ckpts={list(epoch_maes)}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [Error] {class_name} seed={s}: {e}")
                results[s] = {"utility": float("nan"), "mae": float("nan")}

    label = OUTPUT_LABEL[model_key]

    # Per-seed test predictions (input of reconstruct_per_window.py / Figure 2)
    for s, pred_df in all_preds.items():
        pred_df.to_csv(out_dir / f"{label}_{SUFFIX}_seed{s}_preds.csv", index=False)

    # Epoch MAE curves
    emaes_out = out_dir / f"{label}_{SUFFIX}_epoch_maes.json"
    emaes_out.write_text(json.dumps(
        {str(s): {str(ep): v for ep, v in emap.items()} for s, emap in epoch_maes_all.items()},
        indent=2
    ))
    print(f"  [Saved] {emaes_out}")

    # Final results (Table 1)
    out = out_dir / f"{label}_{SUFFIX}_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  [Saved] {out}")


def main(argv=None):
    """Parse the command line and run the requested models one after another."""
    ap = argparse.ArgumentParser(
        description="Train PTO / ZOC / KKT / Hybrid on the conservation game over several seeds "
                    "(Table 1, conservation column) and write results, predictions and epoch MAEs."
    )
    ap.add_argument("--models", default="pto,zoc,kkt,hybrid",
                    help="comma-separated subset of pto,zoc,kkt,hybrid (default: all four, in this order)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="seeds to train, one worker process each (default: 1 2 3 4 5)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="training epochs per seed (default: 500, the paper's setting)")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="path to data_ETOSHA_CELL_9.csv (default: ../data/data_ETOSHA_CELL_9.csv next to this folder)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory (default: results/ next to this script)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel seed processes (default: min(number of seeds, CPU count))")
    args = ap.parse_args(argv)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {list(MODELS)}")

    data_path = str(Path(args.data).resolve())
    if not os.path.exists(data_path):
        ap.error(f"data file not found: {data_path}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.workers is not None and args.workers > 0:
        n_workers = args.workers
    else:
        n_workers = min(len(args.seeds), os.cpu_count() or 4)

    for m in model_keys:
        _run_model(m, args.seeds, args.epochs, data_path, out_dir, n_workers)


if __name__ == "__main__":
    main()
