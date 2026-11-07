"""
build_etosha_cells.py
=====================
Rebuilds ``data_ETOSHA_CELL_9.csv``, the conservation-game training file used in
"Strategic Decision Focused Learning" (NeurIPS 2026), from the raw Movebank export of the
study "African elephants in Etosha National Park (data from Tsalyuk et al. 2018)".

Raw input
---------
The study CSV downloaded from the Movebank Data Repository (see ``README.md`` in this
folder). Only four columns are read:
``timestamp, location-long, location-lat, individual-local-identifier``.
The download used for the paper had 2,930,263 GPS fixes for 15 collared elephants
(LA1 ... LA27), 2008-10-28 to 2014-03-28.

Procedure (verified to reproduce the paper's file byte for byte)
----------------------------------------------------------------
1. Parse ``timestamp`` (UTC, as exported by Movebank); drop rows with an unparseable
   timestamp or a missing coordinate (there are none in the paper's download).
2. All elephants and all dates are used: no identifier filter and no date cut-off.
3. ``time_period`` = calendar day of the fix (timestamp floored to midnight UTC).
   For every (day, elephant) pair with at least one fix, take the **mean** latitude and
   longitude of that elephant's fixes on that day ("mean position per day").
4. Assign every mean position to one of ``n_cells`` = k x k grid cells:
   ``lat_idx = digitize(lat, lat_edges) - 1``, clipped to ``[0, k-1]``, the same for
   longitude, and ``cell_id = lat_idx * k + lon_idx`` (cell 0 is the south-west cell,
   cell_id increases eastwards then northwards). Positions outside the outer edges are
   clipped into the nearest border cell, so every (day, elephant) is counted exactly once.
5. ``elephant_count`` = number of elephants whose mean position fell in the cell that day.
   The table is expanded to every (day with at least one fix) x (cell) pair and missing
   pairs are filled with 0. Days on which no elephant has a fix are **not** in the file
   (2 calendar days in the paper's download), which is why it has 1,976 days rather than
   the 1,978 calendar days spanned.
6. ``day_of_week`` (0 = Monday), ``month`` and ISO ``week_of_year`` are derived from
   ``time_period``; rows are ordered by (``time_period``, ``cell_id``) and written with
   pandas defaults (``index=False``; dates as ``YYYY-MM-DD``).

The grid edges
--------------
The paper's 3 x 3 grid is defined by the latitude and longitude edges stored in
``PAPER_LAT_EDGES`` / ``PAPER_LON_EDGES`` below (identical to the
``data_ETOSHA_CELL_9_grid.csv`` file that sat next to the training file). They are used
verbatim, which is what makes the rebuild byte-identical. They are **not** an even split
of the bounding box, nor the tertiles of the daily positions: the original one-off script
that chose them was not preserved, and the rule behind them could not be recovered
(see ``README.md``). With ``--grid`` you can pass your own edge file in the same format;
with ``--edges quantile`` or ``--edges uniform`` the script instead derives edges from the
data (equal-count per axis, or an even split of the bounding box). Those options give a
*different* file from the paper's and exist only for new experiments or other grid sizes.

Usage
-----
    python build_etosha_cells.py --raw "data/raw/African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv" \
        --out data/data_ETOSHA_CELL_9.csv [--n-cells 9] [--grid path/to/grid.csv] [--edges paper|quantile|uniform]

Expected output for the paper's download (``--n-cells 9``, default edges):
    shape (17784, 6); 1,976 days, 2008-10-28 to 2014-03-28; md5 ae25418fa8538e26a838594e1f0bad17
A grid file ``<out stem>_grid.csv`` (cell bounds) is written next to the output, unless
``--no-write-grid`` is given.
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

# Edges of the paper's 3 x 3 grid (degrees). Four latitude edges (south -> north) and four
# longitude edges (west -> east). Interior values are the cell boundaries; the outer values
# are the grid bounds (positions outside them are clipped into the border cells).
# The literals below are copied character for character from the paper's grid file: two of
# them (-18.598374684936726 and 14.755893161065167) are one floating-point ulp away from what
# pandas' default CSV parser reads back, which does not change any cell assignment (no daily
# position lies within 1e-6 degrees of an edge) but is needed to rewrite the grid file exactly.
PAPER_LAT_EDGES = [-19.370557318813876, -19.19102029953592, -19.08003857465603, -18.598374684936726]
PAPER_LON_EDGES = [14.755893161065167, 15.653931497414366, 15.977430239457314, 16.728916244327888]

EXPECTED = {
    "rows": 17784,
    "days": 1976,
    "first": "2008-10-28",
    "last": "2014-03-28",
    "md5": "ae25418fa8538e26a838594e1f0bad17",
}

RAW_COLS = ["timestamp", "location-long", "location-lat", "individual-local-identifier"]


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_daily_positions(raw_csv):
    """Mean position of every elephant on every day it has at least one fix."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw Movebank file not found: {raw_csv}")
    print("[1/4] Reading raw GPS fixes (4 columns only)...")
    df = pd.read_csv(raw_csv, usecols=RAW_COLS)
    df = df.rename(columns={"individual-local-identifier": "elephant_id",
                            "location-lat": "latitude", "location-long": "longitude"})
    print(f"      {len(df):,} fixes, {df['elephant_id'].nunique()} elephants")

    print("[2/4] Parsing timestamps and averaging positions per (day, elephant)...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    if len(df) != before:
        print(f"      dropped {before - len(df):,} rows with an invalid timestamp or coordinate")
    df["time_period"] = df["timestamp"].dt.floor("D")
    daily = (
        df.groupby(["time_period", "elephant_id"], as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    )
    print(f"      {len(daily):,} (day, elephant) positions on {daily['time_period'].nunique():,} days, "
          f"{daily['time_period'].min().date()} to {daily['time_period'].max().date()}")
    return daily


def edges_from_grid_file(path, k):
    """Recover the k+1 latitude and longitude edges from a cell-bounds CSV."""
    g = pd.read_csv(path)
    lat = sorted(set(g["lat_min"]).union(g["lat_max"]))
    lon = sorted(set(g["lon_min"]).union(g["lon_max"]))
    if len(lat) != k + 1 or len(lon) != k + 1:
        raise ValueError(f"{path}: expected {k + 1} distinct latitude and longitude edges, "
                         f"found {len(lat)} and {len(lon)}")
    return np.array(lat, dtype=float), np.array(lon, dtype=float)


def derive_edges(daily, k, how):
    """Edges computed from the data itself. NOT the paper's grid; see the module docstring."""
    if how == "quantile":
        q = np.linspace(0.0, 1.0, k + 1)
        return np.quantile(daily["latitude"], q), np.quantile(daily["longitude"], q)
    if how == "uniform":
        eps = 1e-6
        return (np.linspace(daily["latitude"].min() - eps, daily["latitude"].max() + eps, k + 1),
                np.linspace(daily["longitude"].min() - eps, daily["longitude"].max() + eps, k + 1))
    raise ValueError(how)


def build_cell_counts(daily, lat_edges, lon_edges, k):
    """Count elephants per (day, cell) and expand to the full day x cell table."""
    d = daily.copy()
    d["lat_idx"] = (np.digitize(d["latitude"].values, lat_edges) - 1).clip(0, k - 1)
    d["lon_idx"] = (np.digitize(d["longitude"].values, lon_edges) - 1).clip(0, k - 1)
    d["cell_id"] = d["lat_idx"] * k + d["lon_idx"]

    obs = d.groupby(["time_period", "cell_id"]).size().reset_index(name="elephant_count")

    all_periods = sorted(d["time_period"].unique())
    full_idx = pd.MultiIndex.from_product([all_periods, list(range(k * k))],
                                          names=["time_period", "cell_id"])
    full = pd.DataFrame(index=full_idx).reset_index()
    full = full.merge(obs, on=["time_period", "cell_id"], how="left")
    full["elephant_count"] = full["elephant_count"].fillna(0).astype(int)

    tp = pd.to_datetime(full["time_period"])
    full["day_of_week"] = tp.dt.dayofweek
    full["month"] = tp.dt.month
    full["week_of_year"] = tp.dt.isocalendar().week.astype(int)
    return full


def grid_table(lat_edges, lon_edges, k):
    rows = []
    for cell in range(k * k):
        i, j = divmod(cell, k)
        rows.append({"cell_id": cell, "lat_min": lat_edges[i], "lat_max": lat_edges[i + 1],
                     "lon_min": lon_edges[j], "lon_max": lon_edges[j + 1]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="Raw Movebank study CSV.")
    p.add_argument("--out", required=True, help="Where to write data_ETOSHA_CELL_<n>.csv.")
    p.add_argument("--n-cells", type=int, default=9, help="Number of cells, a perfect square (default 9).")
    p.add_argument("--grid", default=None,
                   help="Optional cell-bounds CSV (columns cell_id,lat_min,lat_max,lon_min,lon_max) "
                        "whose edges to use instead of the built-in paper edges.")
    p.add_argument("--edges", choices=["paper", "quantile", "uniform"], default="paper",
                   help="Where the edges come from when --grid is not given: the paper's "
                        "built-in 3x3 edges (default; requires --n-cells 9), or edges derived "
                        "from the data (NOT the paper's grid).")
    p.add_argument("--no-write-grid", action="store_true", help="Do not write <out stem>_grid.csv.")
    args = p.parse_args()

    k = int(round(np.sqrt(args.n_cells)))
    if k * k != args.n_cells:
        raise SystemExit(f"--n-cells must be a perfect square (got {args.n_cells})")

    daily = load_daily_positions(args.raw)

    print("[3/4] Choosing grid edges...")
    if args.grid:
        lat_edges, lon_edges = edges_from_grid_file(args.grid, k)
        print(f"      edges read from {args.grid}")
    elif args.edges == "paper":
        if k != 3:
            raise SystemExit("The built-in paper edges define a 3x3 grid; for another size pass "
                             "--grid or choose --edges quantile|uniform.")
        lat_edges, lon_edges = np.array(PAPER_LAT_EDGES), np.array(PAPER_LON_EDGES)
        print("      using the paper's built-in 3x3 edges")
    else:
        lat_edges, lon_edges = derive_edges(daily, k, args.edges)
        print(f"      edges derived from the data ({args.edges}); this is NOT the paper's grid")
    print(f"      lat edges: {[float(v) for v in lat_edges]}")
    print(f"      lon edges: {[float(v) for v in lon_edges]}")
    n_out = int(((daily["latitude"] < lat_edges[0]) | (daily["latitude"] > lat_edges[-1]) |
                 (daily["longitude"] < lon_edges[0]) | (daily["longitude"] > lon_edges[-1])).sum())
    print(f"      {n_out:,} positions lie outside the outer edges and are clipped into border cells")

    print("[4/4] Counting elephants per (day, cell) and saving...")
    full = build_cell_counts(daily, lat_edges, lon_edges, k)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    full.to_csv(args.out, index=False)
    if not args.no_write_grid:
        stem, _ = os.path.splitext(args.out)
        grid_table(lat_edges, lon_edges, k).to_csv(f"{stem}_grid.csv", index=False)
        print(f"      grid bounds written to {stem}_grid.csv")

    digest = md5_of(args.out)
    print(f"\n[Done] {args.out}")
    print(f"       shape     : {full.shape}")
    print(f"       days      : {full['time_period'].nunique():,}  "
          f"({full['time_period'].min().date()} to {full['time_period'].max().date()})")
    print(f"       max count : {full['elephant_count'].max()}")
    print(f"       md5       : {digest}")
    if args.n_cells == 9 and not args.grid and args.edges == "paper":
        ok = (digest == EXPECTED["md5"])
        print(f"       expected  : shape ({EXPECTED['rows']}, 6), {EXPECTED['days']:,} days, "
              f"{EXPECTED['first']} to {EXPECTED['last']}, md5 {EXPECTED['md5']}")
        print(f"       byte-identical to the paper's file: {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
