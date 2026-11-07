"""
plot_regime_panels.py -- the regime panels of Appendix B.1
===========================================================

Re-draws the side-by-side prediction grids of the synthetic Conservation
Security Game from a finished grid_experiment.py run (saved weights and test
scenarios), without retraining anything:

    regime_panel_4row.png         one row per qualitative regime
                                  (Hybrid wins / KKT wins / PTO wins / ZOC wins)
    regime_panel_hybrid_kkt.png   two Hybrid wins and one KKT win
    regime_panel_failures.png     rows where KKT or Hybrid is the worst method
    compare_sc<N>_<winner>.png    the ten single-scenario panels the rows above
                                  were chosen from

Layout (one row per scenario): Ground Truth | PTO | ZOC | KKT | Hybrid, all on
the row's shared "Blues" colour scale; the winning method's header, utility
label and frame are green. The utility u below each panel is the defender's
utility of the SSE strategy computed from that method's prediction, evaluated
against the true density (grid_experiment.eval_vs_truth); a scenario's winners
are the methods within 1e-6 of the best utility, so ties share the highlight.

The scenario numbers hard-coded in main() belong to the paper's stored run
(results/paper/synthetic, 30 test scenarios): they were picked by reading the
`winner` column of its results.csv and looking at the per-scenario compare
panels. A fresh grid_experiment.py run has different winners, so re-pick them
from its results.csv / compare_panels/ before pointing this script at it.

Usage:
    python plot_regime_panels.py                              # paper's stored run
    python plot_regime_panels.py --run-dir <dir> --out <dir> [--grid 10]

--run-dir is either a run folder (it contains obs_test.npy, den_test.npy and
the four *_model.pth files) or a folder of runs, in which case its most recent
grid<N>_* sub-folder is used. --out defaults to the run folder itself.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

SCRIPT_DIR      = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR.parent / "results" / "paper" / "synthetic"

sys.path.insert(0, str(SCRIPT_DIR))

# ── model + solvers from the experiment script ───────────────────────────────
from grid_experiment import (
    GridPredictorCNN,
    solve_sse_full_game,
    eval_vs_truth,
    GRID_CONFIGS,
)
from utils import best_device


# ─── locate the run directory ────────────────────────────────────────────────

def resolve_run_dir(path: Path) -> Path:
    """
    Return the run folder to plot from. `path` itself if it holds obs_test.npy,
    otherwise its most recent grid<N>_* sub-folder that does.
    """
    path = Path(path).expanduser()
    if (path / "obs_test.npy").exists():
        return path
    candidates = sorted(path.glob("grid*_*"), reverse=True)
    for c in candidates:
        if (c / "obs_test.npy").exists():
            return c
    raise FileNotFoundError(
        f"{path} is neither a run folder (no obs_test.npy) nor a folder of runs. "
        "Run grid_experiment.py first, or pass --run-dir."
    )


# ─── load data & models ──────────────────────────────────────────────────────

def load_everything(run_dir: Path, grid: int):
    """
    Load the test scenarios and the four trained CNNs of a run.
    Returns (obs_test, den_test, {name: model}, cfg, device); model names are
    pto, zoc, kkt, hyb. The paper's stored run saved Hybrid as hyb_model.pth,
    the current grid_experiment.py saves hybrid_model.pth; both are accepted.
    """
    cfg    = GRID_CONFIGS[grid]
    device = best_device()

    obs_test = np.load(run_dir / "obs_test.npy")
    den_test = np.load(run_dir / "den_test.npy")

    n_obs = cfg["n_obs"]
    models = {}
    for name in ("pto", "zoc", "kkt", "hyb"):
        candidates = [run_dir / f"{name}_model.pth"]
        if name == "hyb":
            candidates.append(run_dir / "hybrid_model.pth")
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No {name} weights in {run_dir} (looked for {[p.name for p in candidates]})")
        m = GridPredictorCNN(n_obs).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models[name] = m

    return obs_test, den_test, models, cfg, device


def predict_all(obs_test, models, device):
    """Run every model on all test scenarios; returns {name: (n_test, G, G) array}."""
    X = torch.tensor(obs_test, dtype=torch.float32).to(device)
    preds = {}
    with torch.no_grad():
        for name, m in models.items():
            preds[name] = m(X).cpu().numpy()
    return preds


def build_scenario(i, obs_test, den_test, preds, cfg):
    """
    Everything one row needs for test scenario i: the true density and its
    optimal (SSE) strategy and utility; for each method its prediction, the
    SSE strategy of that prediction and that strategy's utility against the
    truth; and the set of winners (methods within 1e-6 of the best utility).
    """
    P  = cfg["catch_penalty"]
    gt = den_test[i]

    opt_s, opt_u = solve_sse_full_game(gt, P)
    results = {"gt": gt, "opt_u": opt_u, "opt_s": opt_s}

    for name in ("pto", "zoc", "kkt", "hyb"):
        pred        = preds[name][i]
        strat, _    = solve_sse_full_game(pred, P)
        u           = eval_vs_truth(strat, gt, P)
        results[f"{name}_pred"]  = pred
        results[f"{name}_strat"] = strat
        results[f"{name}_u"]     = u

    # winner among the four learned methods
    utils  = {n: results[f"{n}_u"] for n in ("pto", "zoc", "kkt", "hyb")}
    best_u = max(utils.values())
    eps    = 1e-6
    winners = [n for n, u in utils.items() if u >= best_u - eps]
    results["winners"] = set(winners) if len(winners) > 0 else set()
    return results


# ─── plotting ────────────────────────────────────────────────────────────────

LABEL_MAP = {"pto": "PTO", "zoc": "ZOC", "kkt": "KKT", "hyb": "Hybrid"}
COL_ORDER  = ["pto", "zoc", "kkt", "hyb"]   # order of learned-method columns
N_COLS     = 1 + len(COL_ORDER)              # GT + 4 methods


def _annotate_grid(ax, data, vmin, vmax, cmap, fig=None, add_cbar=False):
    """Imshow with optional colorbar.  No cell text (grid is 10×10 — too dense)."""
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if add_cbar and fig is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    return im


def plot_scenario_row(sc_results, ax_row, scenario_idx, show_colheaders=True, fig=None):
    """
    Fill one row of axes: [GT] [PTO] [ZOC] [KKT] [Hybrid]
    ax_row : list of N_COLS axes
    No scenario titles or Sc. labels — clean panes for paper inclusion.
    Winners get a bold green header, a bold green utility label and a thick
    green frame; the colour scale is shared across the row (0 .. 1.05 x max).
    """
    gt      = sc_results["gt"]
    winners = sc_results["winners"]

    den_max = max(gt.max(),
                  *(sc_results[f"{n}_pred"].max() for n in COL_ORDER)) * 1.05
    CMAP = "Blues"

    # ── Ground Truth ──
    ax = ax_row[0]
    _annotate_grid(ax, gt, 0, den_max, CMAP, fig=fig, add_cbar=False)
    if show_colheaders:
        ax.set_title("Ground Truth", fontsize=11, fontweight="bold", pad=5,
                     color="#222222")

    # ── Learned methods ──
    for ci, name in enumerate(COL_ORDER):
        ax      = ax_row[ci + 1]
        pred    = sc_results[f"{name}_pred"]
        u       = sc_results[f"{name}_u"]
        is_win  = (name in winners)
        is_last = (ci == len(COL_ORDER) - 1)

        _annotate_grid(ax, pred, 0, den_max, CMAP, fig=fig, add_cbar=is_last)

        if show_colheaders:
            ax.set_title(
                LABEL_MAP[name], fontsize=11, pad=5,
                fontweight="bold" if is_win else "normal",
                color="#1a7a1a" if is_win else "#555555",
            )

        ax.set_xlabel(
            f"$u = {u:+.3f}$",
            fontsize=12,
            color="#1a7a1a" if is_win else "#444444",
            fontweight="bold" if is_win else "normal",
            labelpad=3,
        )

        if is_win:
            for spine in ax.spines.values():
                spine.set_edgecolor("#1a7a1a")
                spine.set_linewidth(3.5)


def make_figure(scenario_indices, obs_test, den_test, preds, cfg, out_dir,
                title="", fname="comparison.png"):
    """
    Build and save one figure with a row per scenario index (0-based) to
    out_dir / fname at 180 dpi. Returns the output path.
    """
    n_rows  = len(scenario_indices)
    col_w   = 3.1   # width per column (5 cols → 15.5 in)
    row_h   = 3.0   # height per row
    fig     = plt.figure(figsize=(col_w * N_COLS, row_h * n_rows + 0.5))

    # leave a little extra right margin for the colorbar on the last column
    gs = gridspec.GridSpec(n_rows, N_COLS, hspace=0.50, wspace=0.25,
                           figure=fig, left=0.07, right=0.93)

    for ri, si in enumerate(scenario_indices):
        sc   = build_scenario(si, obs_test, den_test, preds, cfg)
        axes = [fig.add_subplot(gs[ri, ci]) for ci in range(N_COLS)]
        plot_scenario_row(sc, axes, si, show_colheaders=(ri == 0), fig=fig)

    out = out_dir / fname
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)
    return out


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    """Parse the CLI, load the run, draw the ten single panels and the three regime panels."""
    ap = argparse.ArgumentParser(description="Regime panels of Appendix B.1 from a saved run.")
    ap.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR),
                    help="Run folder (obs_test.npy, den_test.npy, *_model.pth) or a folder "
                         "of runs (its latest grid<N>_* sub-folder is used). "
                         "Default: ../results/paper/synthetic, the paper's stored run.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output folder for the PNGs (default: output/regime_panels "
                         "next to this script, which is git-ignored).")
    ap.add_argument("--grid", type=int, default=10,
                    help="Grid size of the run (GRID_CONFIGS key); the paper's run used 10.")
    args = ap.parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir))
    out_dir = (Path(args.out).expanduser() if args.out
               else Path(__file__).resolve().parent / "output" / "regime_panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder : {run_dir}")
    print(f"Output to  : {out_dir}")

    print("Loading data and models…")
    obs_test, den_test, models, cfg, device = load_everything(run_dir, args.grid)

    print("Running inference…")
    preds = predict_all(obs_test, models, device)

    print("\nGenerating comparison figures…")

    # ── Individual scenario panels ─────────────────────────────────────────────
    # (0-based index, file tag); the comments record why each scenario of the
    # paper's stored run was chosen.
    for sc_0idx, tag in [
        (4,  "sc5_hybrid"),    # Hybrid wins — multi-modal blob
        (16, "sc17_hybrid"),   # Hybrid wins — single blob
        (1,  "sc2_kkt"),       # KKT wins, Hybrid worst
        (28, "sc29_kkt"),      # KKT wins — sparse spike, huge spread
        (7,  "sc8_zoc"),       # ZOC wins, KKT worst
        (0,  "sc1_pto"),       # PTO wins, KKT very bad
        (10, "sc11_pto"),      # PTO wins, smooth blob, methods close
        (12, "sc13_kkt"),      # KKT wins, multi-modal
        (17, "sc18_zoc"),      # ZOC wins, close race
        (25, "sc26_zoc"),      # ZOC wins clearly
    ]:
        make_figure(
            scenario_indices=[sc_0idx],
            obs_test=obs_test, den_test=den_test, preds=preds, cfg=cfg,
            out_dir=out_dir, fname=f"compare_{tag}.png",
        )

    # ── Regime panel: one row per qualitative regime ───────────────────────────
    # Row 1: Hybrid wins  (Sc. 5  — multi-modal blob)
    # Row 2: KKT wins     (Sc. 29 — sparse spike)
    # Row 3: PTO wins     (Sc. 1  — KKT fails badly)
    # Row 4: ZOC wins     (Sc. 8  — intermediate)
    make_figure(
        scenario_indices=[4, 28, 0, 7],
        obs_test=obs_test, den_test=den_test, preds=preds, cfg=cfg,
        out_dir=out_dir, fname="regime_panel_4row.png",
    )

    # ── Hybrid-focus panel: two Hybrid wins + one KKT win for contrast ────────
    make_figure(
        scenario_indices=[4, 16, 28],
        obs_test=obs_test, den_test=den_test, preds=preds, cfg=cfg,
        out_dir=out_dir, fname="regime_panel_hybrid_kkt.png",
    )

    # ── Failure-mode panel: where KKT and Hybrid hurt ─────────────────────────
    # Sc. 1  (KKT is worst, PTO wins)
    # Sc. 8  (KKT worst, ZOC wins)
    # Sc. 2  (Hybrid worst, KKT wins)
    make_figure(
        scenario_indices=[0, 7, 1],
        obs_test=obs_test, den_test=den_test, preds=preds, cfg=cfg,
        out_dir=out_dir, fname="regime_panel_failures.png",
    )

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
