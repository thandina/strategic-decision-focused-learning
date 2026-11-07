"""
plot_instance_panels.py -- Figure 3(a): instance-level panels for the
Conservation Security Game (Etosha National Park, 9 cells).

For one test day it draws five 3x3 heatmaps of elephant counts per cell:
Ground Truth | PTO | ZOC | KKT | Hybrid. The winning method's title and frame
are green and the defender's utility u of each method's induced patrol
strategy on that day is printed below its panel. Two days are selected
automatically (see _select_instances):

    gsg_<day>_e2ewins.{pdf,png}   the day on which an end-to-end method (ZOC,
                                  KKT or Hybrid) beats PTO by the most, among
                                  days where PTO has the lowest MAE of all
                                  methods and its argmax cell is the true
                                  argmax cell (so PTO "looks" best but loses)
    gsg_<day>_ptowins.{pdf,png}   the day on which PTO beats the best
                                  end-to-end method by the most, same filter

The paper's Figure 3(a) is gsg_20130612_000000_e2ewins.pdf (seed-1
predictions of the paper's conservation run). A LaTeX figure environment for
the panels is printed at the end.

Inputs
    --results-dir   folder with {pto,zo3,kkt,hybrid}_gsg_seed<s>_preds.csv, the
                    saved test-set predictions of the paper's run
                    (default ../results/paper/conservation/)
    --data          data_ETOSHA_CELL_9.csv (default ../data/data_ETOSHA_CELL_9.csv);
                    only its test split (from 2013-02-24) is used, loaded through
                    conservation_game.data_pipeline.load_and_prep_gsg2
    --seed          which seed's prediction files to read (default 1)
    --out           output folder (default ./out/)
    --show          also open the figures interactively

Per-day utilities are recomputed with conservation_game.game
.evaluate_game_performance (the same SSE solver and evaluator as the
experiments), one solve per day and method.

Run from anywhere, e.g. from the repository root:
    python figures/plot_instance_panels.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
# conservation_game's modules import their siblings by bare name (for example
# `from solvers.nfg_sse_lp import ...`), so the package folder itself is put on
# the path as well as the repository root.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "conservation_game"))

from conservation_game.data_pipeline import load_and_prep_gsg2
from conservation_game.game import evaluate_game_performance

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "paper" / "conservation"
DEFAULT_DATA        = REPO_ROOT / "data" / "data_ETOSHA_CELL_9.csv"
DEFAULT_OUT         = SCRIPT_DIR / "out"

MODELS       = ["pto", "zo3", "kkt", "hybrid"]     # zo3 = ZOC in the paper
E2E_MODELS   = ["zo3", "kkt", "hybrid"]
MODEL_LABELS = {"pto": "PTO", "zo3": "ZOC", "kkt": "KKT", "hybrid": "Hybrid"}
SEED         = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_preds(results_dir, model, domain, seed=SEED):
    """Read <results_dir>/<model>_<domain>_seed<seed>_preds.csv (None if absent); ids as str."""
    p = Path(results_dir) / f"{model}_{domain}_seed{seed}_preds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["time_period"])
    for col in ("cell_id", "station_complex_id"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-period utility computation
# ---------------------------------------------------------------------------

def _compute_utility_per_period_gsg(pred_df, actual_df):
    """{day: defender utility} of the SSE strategy induced by pred_df, scored on actual_df."""
    _, snapshots = evaluate_game_performance(pred_df, actual_df, collect_snapshots=True)
    # Normalise keys to pd.Timestamp so dict.get() works after DataFrame indexing
    return {pd.Timestamp(r["time_period"]): r["defender_utility"] for r in snapshots}


# ---------------------------------------------------------------------------
# Instance selection: one E2E-wins period, one PTO-wins period
# ---------------------------------------------------------------------------

def _all_models_mae(preds_by_model, actual_df, entity_col, value_col, entity_ids):
    """Returns ({model: {period: mae}}, {period: pto_argmax_matches_gt})."""
    maes   = {m: {} for m in MODELS}
    argmax = {}
    for tp in actual_df["time_period"].unique():
        key = pd.Timestamp(tp)
        act = (actual_df[actual_df["time_period"] == tp]
               .set_index(entity_col).reindex(entity_ids)[value_col].fillna(0).values)
        for m in MODELS:
            prd = (preds_by_model[m][preds_by_model[m]["time_period"] == tp]
                   .set_index(entity_col).reindex(entity_ids)[value_col].fillna(0).values)
            maes[m][key] = float(np.mean(np.abs(act - prd)))
        pto_prd = (preds_by_model["pto"][preds_by_model["pto"]["time_period"] == tp]
                   .set_index(entity_col).reindex(entity_ids)[value_col].fillna(0).values)
        argmax[key] = int(np.argmax(act)) == int(np.argmax(pto_prd))
    return maes, argmax


def _pto_best_mae_periods(mae_by_model):
    """Returns set of periods where PTO has strictly the lowest MAE of all models."""
    all_periods = set(mae_by_model["pto"].keys())
    for m in E2E_MODELS:
        all_periods &= set(mae_by_model[m].keys())
    return {
        tp for tp in all_periods
        if mae_by_model["pto"][tp] == min(mae_by_model[m][tp] for m in MODELS)
    }


def _select_instances(utilities_by_model, mae_by_model=None, pto_argmax_match=None):
    """Returns (tp_e2e, e2e_winner_model, tp_pto) — any may be None.
    Restricts to periods where PTO has best MAE AND its argmax matches ground truth
    (falling back to the unfiltered set when the filter leaves nothing). tp_e2e is
    the period with the largest (best E2E utility - PTO utility) if positive;
    tp_pto the period with the largest (PTO utility - best E2E utility) if positive."""
    common = set.intersection(*[set(u.keys()) for u in utilities_by_model.values()])
    if not common:
        return None, None, None

    if mae_by_model is not None:
        pto_best = _pto_best_mae_periods(mae_by_model)
        filtered = common & pto_best
        # Additionally require PTO argmax matches ground truth
        if pto_argmax_match is not None:
            aligned = {t for t in filtered if pto_argmax_match.get(t, False)}
            filtered = aligned if aligned else filtered  # fall back if none align
        common = filtered if filtered else common

    if not common:
        return None, None, None

    records = [{"time_period": tp, **{m: utilities_by_model[m][tp] for m in MODELS}}
               for tp in sorted(common)]
    df = pd.DataFrame(records).set_index("time_period")

    e2e_best = df[E2E_MODELS].max(axis=1)

    delta_e2e = e2e_best - df["pto"]
    tp_e2e    = delta_e2e.idxmax() if delta_e2e.max() > 0 else None
    e2e_model = df.loc[tp_e2e, E2E_MODELS].idxmax() if tp_e2e is not None else None

    delta_pto = df["pto"] - e2e_best
    tp_pto    = delta_pto.idxmax() if delta_pto.max() > 0 else None

    return tp_e2e, e2e_model, tp_pto


# ---------------------------------------------------------------------------
# Single-panel heatmap helper
# ---------------------------------------------------------------------------

WINNER_COLOR = "#1a7a3a"  # dark green


def _heatmap_3x3(ax, values_9, title, vmin, vmax, cmap, is_winner=False):
    """Draw one 3x3 cell grid with its values printed in each cell (white on dark cells)."""
    grid = np.array(values_9, dtype=float).reshape(3, 3)
    im   = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    for r in range(3):
        for c in range(3):
            val = grid[r, c]
            if val >= 0.095:
                fmt = f"{val:.1f}"
                fs = 12.0
            elif val > 1e-9:
                fmt = f"{val:.3f}"
                fs = 10.5
            else:
                fmt = "0.0"
                fs = 12.0
            ax.text(c, r, fmt, ha="center", va="center",
                    fontsize=fs,
                    color="white" if val > 0.62 * vmax else "#1a1a1a")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title, fontsize=10, pad=4,
        fontweight="bold" if is_winner else "normal",
        color=WINNER_COLOR if is_winner else "#1a1a1a",
    )
    if is_winner:
        for spine in ax.spines.values():
            spine.set_edgecolor(WINNER_COLOR)
            spine.set_linewidth(3.5)
    return im


# ---------------------------------------------------------------------------
# Figure builder (5 panels + colorbar)
# ---------------------------------------------------------------------------

def _make_figure(tp, actual_df, preds_by_model, entity_ids,
                 utils_at_tp, entity_col, value_col, cmap, winner_models):
    """One figure for period tp: Ground Truth + four method panels sharing a colour scale."""
    mpl.rcParams.update({"font.family": "serif"})

    actual_vals = (actual_df[actual_df["time_period"] == tp]
                   .set_index(entity_col).reindex(entity_ids)[value_col]
                   .fillna(0).values)
    pred_vals = {
        m: (preds_by_model[m][preds_by_model[m]["time_period"] == tp]
            .set_index(entity_col).reindex(entity_ids)[value_col]
            .fillna(0).values)
        for m in MODELS
    }

    all_vals = list(actual_vals) + [v for pv in pred_vals.values() for v in pv]
    vmin, vmax = 0.0, max(float(np.max(all_vals)), 1e-6)

    # 5 data columns + 1 narrow colorbar column
    fig = plt.figure(figsize=(10.5, 2.1))
    gs  = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.07],
                            wspace=0.10, left=0.01, right=0.97,
                            top=0.88, bottom=0.18)
    axes   = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cbar_ax = fig.add_subplot(gs[0, 5])
    fig.patch.set_facecolor("white")

    # Ground Truth (no utility label below)
    im = _heatmap_3x3(axes[0], actual_vals, "Ground Truth", vmin, vmax, cmap)
    axes[0].set_xlabel("", fontsize=7)  # blank to keep vertical alignment

    # Method panels
    for ci, m in enumerate(MODELS):
        is_winner = (m in winner_models)
        im = _heatmap_3x3(axes[ci + 1], pred_vals[m], MODEL_LABELS[m],
                          vmin, vmax, cmap, is_winner=is_winner)
        u = utils_at_tp.get(m, float("nan"))
        axes[ci + 1].set_xlabel(
            f"$u = {u:.3f}$",
            fontsize=11,
            fontweight="bold" if is_winner else "normal",
            color=WINNER_COLOR if is_winner else "#555555",
            labelpad=4,
        )

    # Dedicated colorbar axes — no panel gets stolen from
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)

    return fig


# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------

def _tp_str(tp):
    """Timestamp -> compact file-name token, e.g. 2013-06-12 00:00:00 -> 20130612_000000."""
    s = str(tp).replace(" ", "_").replace(":", "").replace("-", "")
    return s[:s.index(".")] if "." in s else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_instance_panels(results_dir, data_path, out_dir, save=True, show=False, seed=SEED):
    """
    Load the Etosha test split and the four models' seed-`seed` predictions,
    recompute every day's utility per model, select the e2ewins / ptowins
    days and write gsg_<day>_<scenario>.pdf/.png to out_dir. Prints the
    LaTeX figure environment at the end.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {"gsg": {}, "mta": {}}

    # ---- GSG ----
    try:
        _, _, test_gsg, _, _ = load_and_prep_gsg2(str(data_path))
        test_gsg["cell_id"] = test_gsg["cell_id"].astype(str)
        cell_ids = sorted(test_gsg["cell_id"].unique())

        gsg_preds = {m: _load_preds(results_dir, m, "gsg", seed) for m in MODELS}
        if all(v is not None for v in gsg_preds.values()):
            print("Computing GSG per-period utilities…")
            utils_gsg = {m: _compute_utility_per_period_gsg(gsg_preds[m], test_gsg)
                         for m in MODELS}
            mae_gsg, argmax_gsg = _all_models_mae(gsg_preds, test_gsg,
                                                  "cell_id", "elephant_count", cell_ids)
            tp_e2e, e2e_model, tp_pto = _select_instances(utils_gsg, mae_gsg, argmax_gsg)

            for scenario, tp, winner in [
                ("e2ewins", tp_e2e, e2e_model),
                ("ptowins", tp_pto, "pto"),
            ]:
                if tp is None:
                    continue
                utils_at_tp = {m: utils_gsg[m].get(tp, float("nan")) for m in MODELS}
                if scenario == "e2ewins":
                    best_u = max(utils_at_tp[m] for m in E2E_MODELS)
                    winner_models = {m for m in E2E_MODELS
                                     if abs(utils_at_tp[m] - best_u) < 1e-9}
                else:
                    winner_models = {"pto"}
                fig = _make_figure(tp, test_gsg, gsg_preds, cell_ids,
                                   utils_at_tp, "cell_id", "elephant_count",
                                   "YlOrRd", winner_models)
                stem = f"gsg_{_tp_str(tp)}_{scenario}"
                if save:
                    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
                    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
                    print(f"[Saved] {out_dir / stem}.{{pdf,png}}")
                saved["gsg"][scenario] = (stem, winner, utils_at_tp, tp)
                if show:
                    plt.show()
                plt.close(fig)
        else:
            missing = [m for m in MODELS if gsg_preds[m] is None]
            print(f"[GSG] Missing prediction files for {missing} in {results_dir} "
                  f"(expected <model>_gsg_seed{seed}_preds.csv) — nothing drawn.")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[GSG] Error: {e}")

    # ---- LaTeX ----
    _print_latex(saved)


def _winner_label(m):
    return MODEL_LABELS[m]


def _print_latex(saved):
    """Print the LaTeX figure environment used in the paper for the saved panels."""
    def subcap(domain_label, scenario, stem, winner, subdir):
        wl = _winner_label(winner)
        cap = f"{domain_label}: {wl} wins" if scenario == "e2ewins" else f"{domain_label}: PTO wins"
        lbl = f"fig:{subdir.replace('_heatmap','').replace('gsg','gsg').replace('mta','mta')}_{scenario}"
        return cap, lbl, stem, subdir

    rows = []
    for scenario in ("e2ewins", "ptowins"):
        if scenario in saved["gsg"]:
            stem, winner, _, _ = saved["gsg"][scenario]
            rows.append(subcap("Conservation", scenario, stem, winner, "gsg_heatmap"))
        if scenario in saved["mta"]:
            stem, winner, _, _ = saved["mta"][scenario]
            rows.append(subcap("Infrastructure", scenario, stem, winner, "mta_heatmap"))

    if not rows:
        return

    gsg_e2e = next((r for r in rows if "Conservation" in r[0] and "PTO" not in r[0]), None)
    gsg_pto = next((r for r in rows if "Conservation" in r[0] and "PTO" in r[0]), None)
    mta_e2e = next((r for r in rows if "Infrastructure" in r[0] and "PTO" not in r[0]), None)
    mta_pto = next((r for r in rows if "Infrastructure" in r[0] and "PTO" in r[0]), None)

    def block(entry, last_in_row=False, last=False):
        cap, lbl, stem, subdir = entry
        sep = "\\\\" + "[0.3em]" if last_in_row else "%"
        hfill = "" if last_in_row or last else "\n    \\hfill"
        return (
            f"    \\subcaptionbox{{{cap}\\label{{{lbl}}}}}[0.495\\textwidth]{{%\n"
            f"        \\includegraphics[width=0.495\\textwidth]{{plots/{subdir}/{stem}.pdf}}%\n"
            f"    }}{sep}{hfill}"
        )

    lines = ["\\begin{figure*}[ht]", "    \\centering"]
    if gsg_e2e: lines.append(block(gsg_e2e))
    if gsg_e2e: lines.append("    \\hfill")
    if gsg_pto: lines.append(block(gsg_pto, last_in_row=True))
    if mta_e2e: lines.append(block(mta_e2e))
    if mta_e2e: lines.append("    \\hfill")
    if mta_pto: lines.append(block(mta_pto, last=True))
    lines += [
        "    \\caption{Instance-level state predictions for selected evaluation periods. "
        "Top row: Conservation Security Game (Etosha); "
        "bottom row: Infrastructure Security Game (Manhattan MTA). "
        "Winner method shown in \\textcolor{green!50!black}{\\textbf{bold green}}; "
        "utility $u$ reported below each prediction grid.}",
        "    \\label{fig:heatmap_comparison}",
        "\\end{figure*}",
    ]
    print("\n" + "=" * 60)
    print("LATEX:")
    print("=" * 60)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Figure 3(a): Etosha instance panels.")
    ap.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                    help="Folder with <model>_gsg_seed<seed>_preds.csv "
                         "(default ../results/paper/conservation/).")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                    help="Path to data_ETOSHA_CELL_9.csv (default ../data/data_ETOSHA_CELL_9.csv).")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Output folder (default ./out/ next to this script).")
    ap.add_argument("--seed", type=int, default=SEED, help="Seed of the prediction files (default 1).")
    ap.add_argument("--show", action="store_true", help="Also show the figures interactively.")
    args = ap.parse_args()
    plot_instance_panels(args.results_dir, args.data, args.out, show=args.show, seed=args.seed)
