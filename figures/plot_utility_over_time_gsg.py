"""
plot_utility_over_time_gsg.py -- Figure 2(a): defender utility over the Etosha test period.

Same method as plot_utility_over_time_mta.py, applied to the conservation security game
(GSG, Etosha, 24 Feb 2013 - 28 Mar 2014): aggregates per-window (per-day) defender utility
to per-day or per-month means, applies any smoothing PER SEED before cross-seed aggregation
(so the SEM band stays statistically honest), then plots one line per model with a shaded
SEM band over the full test period.

Input
-----
A per-window CSV with columns ``time_period, defender_utility, model, seed`` (any extra
columns are ignored). The paper's file is shipped at
``../results/paper/conservation/per_window_utility_from_preds_gsg.csv`` (relative to this
script); it was made by re-solving the game per test day from the saved test predictions
of every (model, seed), with the same evaluator and test split as the training run.
Model names in that file: ``pto``, ``zoc`` (ZOC in the paper), ``kkt``, ``hybrid``.

Outputs
-------
* the plotted aggregate (``model, date, mean, sem, n``), by default written next to the
  input as ``per_day_utility_<models>[_monthly].csv`` (the paper's copy is shipped there);
* the PNG given by ``--out`` (default ``figs/gsg_per_day_utility_<tag>.png`` next to
  this script).

Each monthly point is the mean over the test days of one calendar month, per seed, then
averaged over seeds; the band is the standard error across seeds (sample std, ddof=1).

Command that produced the paper's panel (a) -- the defaults already point at the shipped
input, so ``--in_csv`` may be omitted; ``$'...'`` is bash syntax for an embedded newline:

    python plot_utility_over_time_gsg.py \
        --in_csv ../results/paper/conservation/per_window_utility_from_preds_gsg.csv \
        --models pto,zoc,kkt,hybrid --freq month --title "" --figsize 3.6,2.4 --dpi 300 \
        --end_labels 0 --ylabel $'Defender utility\n(monthly mean)' --legend_pos above \
        --legend_fontsize 8.5 --tick_every 3 --out figs/utility_over_time_gsg.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
DEFAULT_IN_CSV = ROOT.parent / "results" / "paper" / "conservation" / "per_window_utility_from_preds_gsg.csv"

ROLL_WINDOW = 7  # days, centered (only used for --freq day)

MODEL_ORDER  = ["pto", "zoc", "kkt", "hybrid", "kkt_v3", "hybrid_v3"]
MODEL_LABELS = {"pto": "PTO", "zoc": "ZOC", "kkt": "KKT", "hybrid": "Hybrid",
                "kkt_v3": "KKT", "hybrid_v3": "Hybrid"}
# Same 4 maximally-distinct hues used for the MTA plot (blue/red/yellow/green --
# no two series share a color family).
COLORS = {
    "pto":    "#2a78d6",  # blue
    "zoc":    "#e34948",  # red
    "kkt":    "#eda100",  # yellow
    "hybrid": "#008300",  # green
    "kkt_v3":    "#eda100",  # yellow (v3 checkpoints, same hue as kkt)
    "hybrid_v3": "#008300",  # green  (v3 checkpoints, same hue as hybrid)
}
INK_SECONDARY = "#52514e"
GRID_COLOR    = "#e1e0d9"
AXIS_COLOR    = "#c3c2b7"


def sem(x):
    """Standard error of the mean across seeds, sample standard deviation (ddof=1)."""
    n = len(x)
    return x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def build_daily(df):
    """Per-day means with a 7-day centered rolling mean applied per (model, seed) series."""
    per_seed_day = (
        df.groupby(["model", "seed", "date"])["defender_utility"]
        .mean()
        .reset_index()
        .sort_values(["model", "seed", "date"])
    )
    per_seed_day["value"] = (
        per_seed_day.groupby(["model", "seed"])["defender_utility"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, center=True, min_periods=1).mean())
    )
    agg = (
        per_seed_day.groupby(["model", "date"])["value"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    ylabel = (f"Defender utility per day, {ROLL_WINDOW}-day centered rolling mean\n"
              "(mean ± SEM across seeds)")
    return agg, ylabel


def build_monthly(df):
    """Non-overlapping calendar-month bins: mean per (model, seed, month), then mean and
    SEM across seeds."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    per_seed_month = (
        df.groupby(["model", "seed", "month"])["defender_utility"]
        .mean()
        .reset_index()
    )
    agg = (
        per_seed_month.groupby(["model", "month"])["defender_utility"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
        .rename(columns={"month": "date"})
    )
    ylabel = "Defender utility, monthly mean (mean ± SEM across seeds)"
    return agg, ylabel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--in_csv", type=str, default=str(DEFAULT_IN_CSV),
        help="Per-window CSV to read (columns time_period, defender_utility, model, seed). "
             f"Default: {DEFAULT_IN_CSV}",
    )
    parser.add_argument(
        "--agg_csv", type=str, default=None,
        help="Where to write the plotted aggregate. Default: next to the input, "
             "per_day_utility_<models>[_monthly].csv.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help=f"Comma-separated subset of {MODEL_ORDER} (default: all).",
    )
    parser.add_argument(
        "--freq", type=str, default="month", choices=["day", "month"],
        help="'day': 7-day rolling mean. 'month': non-overlapping calendar-month "
             "bins (statistically cleaner SEM). Default: month.",
    )
    parser.add_argument("--title", type=str, default=None,
                        help="Figure title; empty string for none (default: the built-in title).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: figs/<auto name>.png next to this script).")
    parser.add_argument("--figsize", type=str, default="13,6.2",
                        help="Figure size in inches, W,H (default 13,6.2).")
    parser.add_argument("--dpi", type=int, default=170, help="Output dpi (default 170).")
    parser.add_argument("--end_labels", type=int, default=1,
                        help="1: label each line at its right end (default); 0: legend only.")
    parser.add_argument("--ylabel", type=str, default=None, help="Y-axis label override.")
    parser.add_argument("--legend_pos", type=str, default="below", choices=["below", "above"],
                        help="Legend below the axes (default) or above them.")
    parser.add_argument("--tick_every", type=int, default=1, help="Label every N-th month (default 1).")
    parser.add_argument("--legend_fontsize", type=float, default=None, help="Legend font size.")
    args = parser.parse_args()
    selected = args.models.split(",") if args.models else MODEL_ORDER

    IN_CSV = Path(args.in_csv)
    df = pd.read_csv(IN_CSV, parse_dates=["time_period"])
    df["date"] = df["time_period"].dt.normalize()

    present_models = [m for m in MODEL_ORDER if m in selected and m in df["model"].unique()]
    tag = "_vs_".join(present_models) + ("_monthly" if args.freq == "month" else "")
    OUT_CSV = Path(args.agg_csv) if args.agg_csv else IN_CSV.parent / f"per_day_utility_{tag}.csv"
    OUT_PNG = Path(args.out) if args.out else ROOT / "figs" / f"gsg_per_day_utility_{tag}.png"

    agg, ylabel = build_monthly(df) if args.freq == "month" else build_daily(df)
    if args.ylabel is not None:
        ylabel = args.ylabel
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)
    print(f"Saved {args.freq}-level aggregate -> {OUT_CSV}")

    fig, ax = plt.subplots(figsize=tuple(float(v) for v in args.figsize.split(",")))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    is_monthly = args.freq == "month"
    for model in present_models:
        sub = agg[agg["model"] == model].sort_values("date")
        dates = sub["date"]
        mean  = sub["mean"].values
        sem_v = sub["sem"].values
        color = COLORS[model]

        ax.fill_between(dates, mean - sem_v, mean + sem_v, color=color, alpha=0.15, linewidth=0)
        ax.plot(dates, mean, color=color, linewidth=2, solid_capstyle="round",
                marker="o" if is_monthly else None, markersize=6 if is_monthly else None)

        if args.end_labels:
            last_date, last_val = dates.iloc[-1], mean[-1]
            ax.plot([last_date], [last_val], marker="o", markersize=4.5, color=color,
                    markeredgewidth=0)
            ax.annotate(
                f"  {MODEL_LABELS[model]}",
                xy=(last_date, last_val), xycoords="data",
                xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color=INK_SECONDARY,
            )

    ax.set_xlabel("Date (Etosha test period)", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    _title = args.title if args.title is not None else "Strategic game performance over the test period — GSG Etosha"
    ax.set_title(
        _title,
        color="#0b0b0b", fontsize=13, loc="left",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=args.tick_every))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.set_xlim(agg["date"].min(), agg["date"].max() + pd.Timedelta(days=25))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    ax.legend(
        [plt.Line2D([0], [0], color=COLORS[m], linewidth=2) for m in present_models],
        [MODEL_LABELS[m] for m in present_models],
        loc="lower center", bbox_to_anchor=((0.5, 1.0) if args.legend_pos == "above" else (0.5, -0.22)), ncol=len(present_models), frameon=False,
        labelcolor=INK_SECONDARY, fontsize=args.legend_fontsize,
    )

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved plot -> {OUT_PNG}")

    print("\nOverall per-model mean of aggregated means:")
    for model in present_models:
        sub = agg[agg["model"] == model]
        print(f"  {model:8s} = {sub['mean'].mean():.4f}")


if __name__ == "__main__":
    main()
