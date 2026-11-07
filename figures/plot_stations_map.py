#!/usr/bin/env python
"""
plot_stations_map.py -- Appendix Figure 5(b): map of the 123 Manhattan subway
stations of the Infrastructure Security Game.

Every station complex that appears in the processed training file is drawn as
a red dot at its latitude/longitude on an OpenStreetMap basemap of Manhattan.

Inputs
    --raw PATH   the raw "MTA Subway Hourly Ridership" CSV from data.ny.gov
                 (dataset wujg-7c2s; see data/README.md). Only the columns
                 station_complex_id, latitude and longitude are read, one
                 row per station complex. The file is several GB and is not
                 distributed with this repository, so this argument is required.
    --data PATH  the processed data_MTA_8H_Manhattan.csv, which defines the
                 123 stations (default ../data/data_MTA_8H_Manhattan.csv)
    --out DIR    output folder (default ./out/); writes mta_all_stations_map.png

Optional dependency: `contextily` (pip install contextily) fetches the
OpenStreetMap tiles; it is NOT in requirements.txt. Without it, or offline,
the stations are drawn on a plain light-blue background instead.

Run from anywhere, e.g. from the repository root:
    python figures/plot_stations_map.py --raw /path/to/MTA_Subway_Hourly_Ridership.csv
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import contextily as ctx          # optional: OpenStreetMap basemap tiles
except ImportError:                   # fall back to a plain background below
    ctx = None

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "out")
OUT_NAME    = "mta_all_stations_map.png"
DPI         = 300

DEFAULT_PROCESSED_CSV = os.path.join(REPO_ROOT, "data", "data_MTA_8H_Manhattan.csv")


def load_station_coords(raw_csv, processed_csv):
    """
    One (station_complex_id, latitude, longitude) row per station in the
    processed file, taken from the first occurrence of that id in the raw file.
    """
    processed = pd.read_csv(processed_csv)
    our_ids   = set(processed["station_complex_id"].astype(str).unique())

    raw = pd.read_csv(raw_csv,
                      usecols=["station_complex_id", "latitude", "longitude"],
                      low_memory=False)
    raw["station_complex_id"] = raw["station_complex_id"].astype(str)
    coords = (raw.drop_duplicates("station_complex_id")
                 .query("station_complex_id in @our_ids")
                 [["station_complex_id", "latitude", "longitude"]])
    coords["latitude"]  = coords["latitude"].astype(float)
    coords["longitude"] = coords["longitude"].astype(float)
    return coords


def make_map(coords, out_png):
    """Scatter the stations (padded bounding box, equal aspect, no axes) over the basemap and save."""
    lats = coords["latitude"].values
    lons = coords["longitude"].values

    pad_lat = 0.012
    pad_lon = 0.018

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.set_xlim(lons.min() - pad_lon, lons.max() + pad_lon)
    ax.set_ylim(lats.min() - pad_lat, lats.max() + pad_lat)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.scatter(
        lons, lats,
        s=28, color="#e63946",
        edgecolors="white", linewidths=0.5,
        zorder=5,
    )

    if ctx is None:
        print("  [!] contextily not installed — drawing without map tiles "
              "(pip install contextily for the OpenStreetMap basemap)")
        ax.set_facecolor("#d4e6f1")
    else:
        try:
            ctx.add_basemap(
                ax, crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom="auto", attribution_size=5,
            )
        except Exception as exc:
            print(f"  [!] Map tiles failed: {exc}")
            ax.set_facecolor("#d4e6f1")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[Saved] {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Appendix Figure 5(b): map of the 123 MTA stations.")
    ap.add_argument("--raw", type=str, required=True,
                    help="Raw MTA Subway Hourly Ridership CSV from data.ny.gov "
                         "(columns station_complex_id, latitude, longitude).")
    ap.add_argument("--data", type=str, default=DEFAULT_PROCESSED_CSV,
                    help="Processed data_MTA_8H_Manhattan.csv defining the stations "
                         "(default ../data/data_MTA_8H_Manhattan.csv).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT,
                    help=f"Output folder (default ./out/); writes {OUT_NAME}.")
    args = ap.parse_args()

    print("Loading station coordinates...")
    coords = load_station_coords(args.raw, args.data)
    print(f"  {len(coords)} stations loaded")
    print("Rendering map...")
    make_map(coords, os.path.join(args.out, OUT_NAME))
