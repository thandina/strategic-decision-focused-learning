"""
plot_strategies_mta.py -- Figure 3(b) and 3(c): ridership predictions and
patrol strategies for the Infrastructure Security Game (123 Manhattan subway
stations, eight-hour windows).

Reloads one checkpoint per method from the paper's run, regenerates its
test-set predictions, re-solves the Stackelberg game for a sample of test
windows and draws two kinds of figure for two illustrative windows:

  ridership_<scenario>.png        Ground Truth | PTO | ZOC | KKT | Hybrid as an
                                  11 x 12 grid of all 123 stations (sorted by
                                  mean test ridership, descending; the last 9
                                  cells are padding), PowerNorm(0.5) colours
  strategy_<scenario>.png         True Stackelberg Strategy | PTO | ZOC | KKT |
                                  Hybrid: the SSE patrol probabilities of the
                                  top-9 stations as a 3 x 3 grid with the
                                  probabilities printed, PowerNorm(0.35)

for scenario in {pto_wins, hybrid_wins}, each once with the utility labels
("u = ...") and once without ("_nolabel"), so 8 PNGs in all, plus
strategy_debug.txt listing every station's values for both windows. The
paper's Figure 3(b) is strategy_hybrid_wins.png and Figure 3(c) is
ridership_hybrid_wins_nolabel.png. The winner (green title, frame and label)
is the method with the highest utility in that window.

Window selection: 150 test windows are sampled with random.seed(42); for each
the true SSE strategy and utility and, per method, the SSE strategy of the
predicted ridership and its utility against the true ridership are computed.
"pto_wins" is the window where PTO is the best of the four methods by the
largest margin over second place; "hybrid_wins" likewise for Hybrid (if a
method never wins, the window with the largest -- least negative -- margin
is used instead, with a warning).

Inputs
    --ckpt-dir   folder with pto_seed2.pt, zoc_seed4.pt, kkt_v3_seed4.pt and
                 hybrid_v3_seed1.pt, the seeds used in the paper's figure
                 (default ../results/paper/infrastructure/checkpoints/)
    --data       data_MTA_8H_Manhattan.csv (default ../data/data_MTA_8H_Manhattan.csv)
    --out        output folder (default ./out/)

Uses infrastructure_game.data_pipeline.load_and_prep_mta (test split: 2024),
infrastructure_game.game.gen_mta_utility_matrices and the SSE solver in
infrastructure_game.solvers.nfg_sse_lp; the model classes come from
infrastructure_game.models.{pto,zoc,kkt,hybrid}. Each checkpoint stores the
model's state_dict, its fitted feature scaler, its hyper-parameters and the
feature columns.

Run from anywhere, e.g. from the repository root:
    python figures/plot_strategies_mta.py
Solving 750 Stackelberg games of 123 targets takes a few minutes on a laptop.
"""

import sys, importlib, random, argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
# infrastructure_game's modules import their siblings by bare name (for
# example `from utils import best_device`), so the package folder itself is
# put on the path as well as the repository root.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "infrastructure_game"))

from infrastructure_game.data_pipeline   import load_and_prep_mta
from infrastructure_game.game            import gen_mta_utility_matrices
from infrastructure_game.solvers.nfg_sse_lp import (solve_general_sum_normal_form_mta,
                                                    calculate_actual_attacker_strat_mta)

# ── Global style — match paper (Times New Roman, larger fonts) ─────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "Liberation Serif",
                           "DejaVu Serif"],
    "font.size":          14,
    "axes.titlesize":     14,
    "axes.labelsize":     14,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "figure.dpi":         150,
})

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_CKPT_DIR = REPO_ROOT / "results" / "paper" / "infrastructure" / "checkpoints"
DEFAULT_DATA     = REPO_ROOT / "data" / "data_MTA_8H_Manhattan.csv"
DEFAULT_OUT      = SCRIPT_DIR / "out"

N_STATIONS       = 123
R_ROWS, R_COLS   = 11, 12   # ridership grid  (last 9 cells NaN-padded)
S_ROWS, S_COLS   = 3,  3    # strategy grid   (top-9 stations)
N_STRAT_STATIONS = S_ROWS * S_COLS

GAMMA_R = 0.5    # PowerNorm for ridership
GAMMA_S = 0.35   # PowerNorm for strategy (sparser, needs more stretch)

FIG_SIZE         = (2.5 * 5, 3.0)   # same for both ridership and strategy figures
N_SAMPLE_PERIODS = 150
WINNER_COLOR     = "#2ca02c"         # green for winner border always

# key -> (module under infrastructure_game.models, class name, seed of the checkpoint
# used in the paper's figure). Checkpoint file: <key>_seed<seed>.pt
MODEL_SEEDS = {
    "pto":       ("pto",    "REGRESSION_MODEL_MTA_V2",     2),
    "zoc":       ("zoc",    "REGRESSION_ZO3_MODEL_MTA_V2", 4),
    "kkt_v3":    ("kkt",    "REGRESSION_E2E_KKT_MTA_V3",   4),
    "hybrid_v3": ("hybrid", "REGRESSION_HYBRID_MTA_V3",    1),
}

# Labels used in both figures; ground truth label differs per figure
MODEL_LABELS = {
    "ground_truth": "Ground Truth",          # overridden in strategy figure
    "pto":          "PTO",
    "zoc":          "ZOC",
    "kkt_v3":       "KKT",
    "hybrid_v3":    "Hybrid",
}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(key, ckpt_dir):
    """
    Rebuild the model `key` from <ckpt_dir>/<key>_seed<seed>.pt: instantiate the
    class with the stored hyper-parameters and feature columns (window_size 6),
    load the weights and the fitted scaler, switch to eval mode.
    """
    mod_file, cls_name, seed = MODEL_SEEDS[key]
    path = Path(ckpt_dir) / f"{key}_seed{seed}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mod      = importlib.import_module(f"infrastructure_game.models.{mod_file}")
    ModelCls = getattr(mod, cls_name)
    m        = ModelCls(ckpt["params"], ckpt["feature_cols"], window_size=6)
    m.model.load_state_dict(ckpt["model_state_dict"])
    m.scaler = ckpt["scaler"]
    m.model.eval()
    return m


# ── Game helpers ───────────────────────────────────────────────────────────────

def get_strategy_and_utility(ridership_series):
    """SSE defender strategy (probability per station) and its value for a ridership vector."""
    D, A = gen_mta_utility_matrices(ridership_series)
    _, strat, util = solve_general_sum_normal_form_mta(D, A)
    return np.asarray(strat, dtype=float), float(util)


def eval_utility(pred_strat, true_ridership_series):
    """Defender utility of a committed strategy when the attacker best-responds to the TRUE ridership."""
    D_real, A_real = gen_mta_utility_matrices(true_ridership_series)
    _, util = calculate_actual_attacker_strat_mta(pred_strat, D_real, A_real)
    return float(util)


# ── Data helpers ───────────────────────────────────────────────────────────────

def sort_order(test_set):
    """Station ids sorted by mean test-set ridership, descending (the grid order)."""
    return (test_set.groupby("station_complex_id")["ridership"]
                    .mean()
                    .sort_values(ascending=False)
                    .index.tolist())


def get_series(df, t, station_order):
    """Ridership of every station at window t as a Series in station_order (missing -> 0)."""
    return (df[df["time_period"] == t]
            .sort_values("station_complex_id")
            .set_index("station_complex_id")
            .reindex(station_order)["ridership"]
            .fillna(0))


def to_ridership_grid(vec):
    """123-vector -> 11 x 12 grid, the 9 unused cells NaN."""
    padded              = np.full(R_ROWS * R_COLS, np.nan)
    padded[:N_STATIONS] = vec[:N_STATIONS]
    return padded.reshape(R_ROWS, R_COLS)


def to_strategy_grid(strat_vec):
    """First 9 entries (top-9 stations) of a strategy vector -> 3 x 3 grid."""
    return strat_vec[:N_STRAT_STATIONS].reshape(S_ROWS, S_COLS)


# ── Debug writer ───────────────────────────────────────────────────────────────

def write_debug(path, scenario, rec, station_order, pred_series):
    """Append all 123 stations' true/predicted ridership, the top-9 strategies and the utilities of one window to a text file."""
    true_t   = rec["true_t"]
    col_keys = list(MODEL_SEEDS.keys())
    lines    = []

    lines.append("=" * 82)
    lines.append(f"SCENARIO   : {scenario}")
    lines.append(f"TIME PERIOD: {rec['t']}")
    lines.append("")

    header = (f"  {'Rank':<5} {'Station ID':<34} {'True Ridership':>14}  " +
              "  ".join(f"{k:>12}" for k in col_keys))
    lines.append("ALL 123 STATIONS  (ranked by mean ridership, descending)")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i in range(N_STATIONS):
        sid       = station_order[i]
        preds_str = "  ".join(f"{pred_series[k].values[i]:>12.2f}" for k in col_keys)
        lines.append(f"  {i+1:<5} {str(sid):<34} {true_t.values[i]:>14.2f}  {preds_str}")

    lines.append("")
    lines.append("SSE PATROL STRATEGY — top-9 stations (prob × 100)")
    strat_header = (f"  {'Rank':<5} {'Station ID':<34} {'GT Strategy':>11}  " +
                    "  ".join(f"{k:>12}" for k in col_keys))
    lines.append(strat_header)
    for i in range(N_STRAT_STATIONS):
        sid    = station_order[i]
        gt_p   = rec["strats"]["ground_truth"][i] * 100
        mod_ps = "  ".join(f"{rec['strats'][k][i]*100:>12.2f}" for k in col_keys)
        lines.append(f"  {i+1:<5} {str(sid):<34} {gt_p:>11.2f}  {mod_ps}")

    lines.append("")
    lines.append("UTILITIES THIS PERIOD")
    for label, u in rec["utils"].items():
        lines.append(f"  {label:<14}: {u:.4f}")

    lines.append("=" * 82)
    lines.append("")

    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _apply_winner_border(ax, is_winner):
    if is_winner:
        for sp in ax.spines.values():
            sp.set_edgecolor(WINNER_COLOR)
            sp.set_linewidth(5.0)


def _utility_label(ax, u, is_winner, show=True):
    if show:
        color = WINNER_COLOR if is_winner else "black"
        bold  = "bold"   if is_winner else "normal"
        ax.set_xlabel(f"u = {u:.3f}", fontsize=14, color=color, fontweight=bold)


# ── Ridership figure ───────────────────────────────────────────────────────────

def plot_ridership(rec, scenario, winner, out_dir, show_utility=True):
    """Five 11 x 12 ridership heatmaps for one window -> <out_dir>/ridership_<scenario>[_nolabel].png."""
    panel_keys = ["ground_truth", "pto", "zoc", "kkt_v3", "hybrid_v3"]
    fig, axes  = plt.subplots(1, len(panel_keys), figsize=FIG_SIZE,
                               constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.18)

    gt_vals = rec["ridership"]["ground_truth"]
    r_vmax  = float(np.percentile(gt_vals[gt_vals > 0], 98))
    norm    = PowerNorm(gamma=GAMMA_R, vmin=0, vmax=r_vmax)
    cmap    = plt.cm.Blues

    for ax, key in zip(axes, panel_keys):
        grid = to_ridership_grid(rec["ridership"][key])
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                  interpolation="nearest", aspect="auto")
        ax.set_facecolor("#d8d8d8")
        ax.set_xticks([])
        ax.set_yticks([])
        is_w = (key == winner)
        ax.set_title(MODEL_LABELS[key], fontsize=14, pad=5,
                     color=WINNER_COLOR if is_w else "black",
                     fontweight="bold" if is_w else "normal")
        _apply_winner_border(ax, is_w)
        _utility_label(ax, rec["utils"].get(key, rec["utils"]["ground_truth"]), is_w,
                        show=show_utility)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], fraction=0.046, pad=0.04, label="Ridership")

    suffix = "" if show_utility else "_nolabel"
    out = Path(out_dir) / f"ridership_{scenario}{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Strategy figure ────────────────────────────────────────────────────────────

def plot_strategy(rec, scenario, winner, out_dir, show_utility=True):
    """Five 3 x 3 patrol-probability grids (top-9 stations) -> <out_dir>/strategy_<scenario>[_nolabel].png."""
    panel_keys  = ["ground_truth", "pto", "zoc", "kkt_v3", "hybrid_v3"]
    strat_titles = {
        "ground_truth": "True Stackelberg\nStrategy",
        "pto":          "PTO",
        "zoc":          "ZOC",
        "kkt_v3":       "KKT",
        "hybrid_v3":    "Hybrid",
    }
    fig, axes = plt.subplots(1, len(panel_keys), figsize=FIG_SIZE,
                              constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.18)

    all_nonzero = np.concatenate([
        rec["strats"][k][:N_STRAT_STATIONS][
            rec["strats"][k][:N_STRAT_STATIONS] > 1e-6
        ]
        for k in panel_keys
    ])
    s_vmax = float(np.percentile(all_nonzero, 98)) if len(all_nonzero) else 0.5
    norm   = PowerNorm(gamma=GAMMA_S, vmin=0, vmax=s_vmax)
    cmap   = plt.cm.Blues

    for ax, key in zip(axes, panel_keys):
        grid = to_strategy_grid(rec["strats"][key])
        ax.imshow(grid, cmap=cmap, norm=norm,
                  interpolation="nearest", aspect="auto")
        ax.set_facecolor("#d8d8d8")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, S_COLS - 0.5)
        ax.set_ylim(S_ROWS - 0.5, -0.5)
        is_w = (key == winner)
        ax.set_title(strat_titles[key], fontsize=14, pad=5,
                     color=WINNER_COLOR if is_w else "black",
                     fontweight="bold" if is_w else "normal")

        # Numbers in each cell — raw probabilities (sum to 1)
        for r in range(S_ROWS):
            for c in range(S_COLS):
                val     = grid[r, c]
                txt_col = "white" if norm(val) > 0.55 else "black"
                ax.text(c, r, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=14, color=txt_col, fontweight="bold")

        _apply_winner_border(ax, is_w)
        _utility_label(ax, rec["utils"].get(key, rec["utils"]["ground_truth"]), is_w,
                        show=show_utility)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], fraction=0.046, pad=0.04, label="Patrol prob.")

    suffix = "" if show_utility else "_nolabel"
    out = Path(out_dir) / f"strategy_{scenario}{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(ckpt_dir, data_path, out_dir):
    """Load data and models, score the sampled windows, pick the two scenarios, draw and save."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    _, _, test_set, _, _ = load_and_prep_mta(str(data_path))
    station_order = sort_order(test_set)
    print(f"  {len(station_order)} stations")

    print("Loading models and generating predictions …")
    preds = {}
    for key in MODEL_SEEDS:
        print(f"  {key} (seed {MODEL_SEEDS[key][2]}) …")
        preds[key] = load_model(key, ckpt_dir).generate(test_set)

    all_periods = sorted(test_set["time_period"].unique())
    sample_t    = random.sample(all_periods, min(N_SAMPLE_PERIODS, len(all_periods)))
    print(f"Searching {len(sample_t)} sampled periods …")

    records = []
    for t in sample_t:
        true_t            = get_series(test_set, t, station_order)
        gt_strat, gt_util = get_strategy_and_utility(true_t)

        row = {
            "t":         t,
            "true_t":    true_t,
            "ridership": {"ground_truth": true_t.values.copy()},
            "strats":    {"ground_truth": gt_strat},
            "utils":     {"ground_truth": gt_util},
        }
        for key in MODEL_SEEDS:
            pred_t             = get_series(preds[key], t, station_order)
            m_strat, _         = get_strategy_and_utility(pred_t)
            row["ridership"][key] = pred_t.values.copy()
            row["strats"][key]    = m_strat
            row["utils"][key]     = eval_utility(m_strat, true_t)
            row[f"pred_{key}"]    = pred_t

        records.append(row)

    def best_model(r):
        return max(MODEL_SEEDS.keys(), key=lambda k: r["utils"][k])

    def margin_over_rest(r, target):
        others = [r["utils"][k] for k in MODEL_SEEDS if k != target]
        return r["utils"][target] - max(others)

    # Periods where the target model is genuinely the best across all models,
    # ranked by how much it beats second place
    pto_wins_recs     = [r for r in records if best_model(r) == "pto"]
    hybrid_wins_recs  = [r for r in records if best_model(r) == "hybrid_v3"]

    if not pto_wins_recs:
        print("WARNING: no period found where PTO is the best model — using closest")
        pto_wins_recs = sorted(records, key=lambda r: margin_over_rest(r, "pto"), reverse=True)
    if not hybrid_wins_recs:
        print("WARNING: no period found where Hybrid is the best model — using closest")
        hybrid_wins_recs = sorted(records, key=lambda r: margin_over_rest(r, "hybrid_v3"), reverse=True)

    rec_pto_wins    = max(pto_wins_recs,    key=lambda r: margin_over_rest(r, "pto"))
    rec_hybrid_wins = max(hybrid_wins_recs, key=lambda r: margin_over_rest(r, "hybrid_v3"))

    print(f"\nPTO-wins    t={rec_pto_wins['t']}  "
          f"PTO={rec_pto_wins['utils']['pto']:.3f}  "
          f"Hybrid={rec_pto_wins['utils']['hybrid_v3']:.3f}")
    print(f"Hybrid-wins t={rec_hybrid_wins['t']}  "
          f"PTO={rec_hybrid_wins['utils']['pto']:.3f}  "
          f"Hybrid={rec_hybrid_wins['utils']['hybrid_v3']:.3f}")

    debug_path = out_dir / "strategy_debug.txt"
    debug_path.unlink(missing_ok=True)

    for scenario, rec in [
        ("pto_wins",    rec_pto_wins),
        ("hybrid_wins", rec_hybrid_wins),
    ]:
        winner = max(MODEL_SEEDS.keys(), key=lambda k: rec["utils"][k])
        print(f"  {scenario}: actual winner = {winner} (u={rec['utils'][winner]:.3f})")
        pred_series = {k: rec[f"pred_{k}"] for k in MODEL_SEEDS}
        write_debug(debug_path, scenario, rec, station_order, pred_series)
        plot_ridership(rec, scenario, winner, out_dir, show_utility=True)
        plot_ridership(rec, scenario, winner, out_dir, show_utility=False)
        plot_strategy(rec, scenario, winner, out_dir, show_utility=True)
        plot_strategy(rec, scenario, winner, out_dir, show_utility=False)

    print(f"\nDebug info → {debug_path}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Figure 3(b),(c): MTA ridership and strategy panels.")
    ap.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR),
                    help="Folder with <model>_seed<s>.pt checkpoints "
                         "(default ../results/paper/infrastructure/checkpoints/).")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                    help="Path to data_MTA_8H_Manhattan.csv (default ../data/data_MTA_8H_Manhattan.csv).")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Output folder (default ./out/ next to this script).")
    args = ap.parse_args()
    random.seed(42)
    main(args.ckpt_dir, args.data, args.out)
