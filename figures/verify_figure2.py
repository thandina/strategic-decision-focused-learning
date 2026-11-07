"""
verify_figure2.py -- recompute every number behind Figure 2 and Table 1 of the paper from the
shipped per-window files, so the figure can be checked independently of the plotting scripts.

    python verify_figure2.py            # uses the shipped files under ../results/paper/

Inputs (defaults relative to this script; override with the flags of the same name)
  --mta_per_window  ../results/paper/infrastructure/per_window_utility_reconstructed.csv
                    (1,098 windows x 5 seeds x 4 models; made by reloading each checkpoint of the
                    training run, regenerating test predictions and re-solving the game per window)
  --mta_monthly     ../results/paper/infrastructure/per_day_utility_pto_vs_zoc_vs_kkt_v3_vs_hybrid_v3_monthly.csv
                    (the monthly values plotted in Figure 2(b), written by plot_utility_over_time_mta.py)
  --gsg_per_window  ../results/paper/conservation/per_window_utility_from_preds_gsg.csv
                    (396 test days x 5 seeds x 4 models; the game re-solved per day from the saved
                    test predictions of the training run behind Table 1)
  --gsg_monthly     ../results/paper/conservation/per_day_utility_pto_vs_zoc_vs_kkt_vs_hybrid_monthly.csv
                    (the monthly values plotted in Figure 2(a), written by plot_utility_over_time_gsg.py)

For every model it prints the per-seed mean utility averaged over seeds (= Table 1's mean), the
population and sample standard deviations across seeds, the values printed in Table 1, and the
window-weighted average of the plotted monthly curve, which must equal the Table 1 mean.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "results" / "paper"
TABLE = {  # Table 1 of the paper (mean +- std across 5 runs)
    "mta": {"pto": (-13.611, 0.210), "zoc": (-13.160, 0.355), "kkt": (-12.829, 0.465), "hybrid": (-12.578, 0.383)},
    "gsg": {"pto": (-0.831, 0.005), "zoc": (-0.785, 0.009), "kkt": (-0.404, 0.483), "hybrid": (-0.173, 0.013)},
}
NAMES = {
    "mta": {"pto": "pto", "zoc": "zoc", "kkt_v3": "kkt", "hybrid_v3": "hybrid"},
    "gsg": {"pto": "pto", "zoc": "zoc", "kkt": "kkt", "hybrid": "hybrid"},
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mta_per_window", default=str(PAPER / "infrastructure" / "per_window_utility_reconstructed.csv"))
    p.add_argument("--mta_monthly", default=str(PAPER / "infrastructure" / "per_day_utility_pto_vs_zoc_vs_kkt_v3_vs_hybrid_v3_monthly.csv"))
    p.add_argument("--gsg_per_window", default=str(PAPER / "conservation" / "per_window_utility_from_preds_gsg.csv"))
    p.add_argument("--gsg_monthly", default=str(PAPER / "conservation" / "per_day_utility_pto_vs_zoc_vs_kkt_vs_hybrid_monthly.csv"))
    args = p.parse_args()

    cases = {
        "mta": {"per_window": Path(args.mta_per_window), "monthly": Path(args.mta_monthly), "names": NAMES["mta"]},
        "gsg": {"per_window": Path(args.gsg_per_window), "monthly": Path(args.gsg_monthly), "names": NAMES["gsg"]},
    }

    for game, c in cases.items():
        pw = pd.read_csv(c["per_window"], parse_dates=["time_period"])
        pw = pw[pw["model"].isin(c["names"])].copy()
        pw["model"] = pw["model"].map(c["names"])
        pw["month"] = pw["time_period"].dt.to_period("M")
        mo = pd.read_csv(c["monthly"], parse_dates=["date"])
        mo["model"] = mo["model"].map(c["names"])
        mo["month"] = mo["date"].dt.to_period("M")
        windows_per_month = pw[pw["model"] == "pto"].groupby("month")["seed"].size() / pw[pw["model"] == "pto"]["seed"].nunique()
        print(f"\n=== {game.upper()} ===  {c['per_window']}")
        print(f"{'model':7s} {'seeds':>5s} {'windows':>7s} | {'mean':>8s} {'std(pop)':>8s} {'std(smp)':>8s} | {'table':>8s} {'table sd':>8s} | {'plot avg (weighted)':>19s} {'plain avg of points':>19s}")
        for m in ["pto", "zoc", "kkt", "hybrid"]:
            sub = pw[pw["model"] == m]
            per_seed = sub.groupby("seed")["defender_utility"].mean()
            n_win = sub.groupby("seed").size().iloc[0]
            pts = mo[mo["model"] == m].set_index("month")["mean"]
            w = windows_per_month.reindex(pts.index)
            weighted = float((pts * w).sum() / w.sum())
            t_mean, t_sd = TABLE[game][m]
            print(f"{m:7s} {len(per_seed):5d} {n_win:7d} | {per_seed.mean():8.4f} {per_seed.std(ddof=0):8.4f} {per_seed.std(ddof=1):8.4f} | "
                  f"{t_mean:8.3f} {t_sd:8.3f} | {weighted:19.4f} {pts.mean():19.4f}")
        print(f"  months: {len(windows_per_month)}; windows per month: " + ", ".join(f"{k}:{int(v)}" for k, v in windows_per_month.items()))
    print("\nConvention: every +- in the paper is the sample standard deviation across the 5 runs"
          " (ddof=1, divide by n-1), and the Figure 2 bands are the standard error with the same ddof=1."
          " 'plot avg (weighted)' weights each monthly point by its number of test windows and must equal 'mean';"
          " the plain average of the 12/14 points differs where months have unequal numbers of windows"
          " (the Etosha test period starts on 24 Feb 2013, so its first month has 5 days).")


if __name__ == "__main__":
    main()
