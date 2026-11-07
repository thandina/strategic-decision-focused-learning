"""
build_mta_manhattan.py
======================
Builds ``data_MTA_8H_Manhattan.csv``, the infrastructure-game training file used in
"Strategic Decision Focused Learning" (NeurIPS 2026), from the raw MTA ridership export.

Raw input
---------
The NY State open-data dataset "MTA Subway Hourly Ridership: 2020-2024" (id ``wujg-7c2s``),
exported as CSV. Only these columns are read:
``transit_timestamp, station_complex_id, borough, ridership``.
See ``README.md`` in this folder for where to download it.

Transformation (kept identical to the script that produced the paper's file)
----------------------------------------------------------------------------
1. ``ridership``: strip thousands separators, coerce to numeric, drop unparseable rows,
   divide by 1,000 (the file stores ridership in **thousands of riders**).
2. ``transit_timestamp``: parse with ``%m/%d/%Y %I:%M:%S %p``; rows that fail fall back
   to pandas' generic parser; rows that still fail are dropped.
3. Keep rows with ``borough == "Manhattan"`` (123 station complexes) and
   ``transit_timestamp >= 2022-01-01``.
4. Floor every timestamp to an 8-hour bin (00:00, 08:00, 16:00) and sum ridership per
   (bin, station). Only (bin, station) pairs that occur in the raw data are kept; no
   zero-filling is done.
5. Add ``hour`` (bin start hour), ``day_of_week`` (0 = Monday) and ``month``; sort by
   (``station_complex_id`` as a string, ``time_period``); write with ``index=False``.

Output columns
--------------
``time_period, station_complex_id, ridership, hour, day_of_week, month``

Expected output (the file behind Table 1 / Figure 2 of the paper)
------------------------------------------------------------------
* 123 stations, 403,469 rows, 6 columns
* ``time_period`` from 2022-01-01 16:00:00 to 2024-12-31 16:00:00
* ridership in thousands, max 83.469
* md5 ``c5b182823062e9f469ea82858fca1fe5``

Note: the raw dataset is a living Socrata export. A download made on a later date can
contain revised rows, so an exact md5 match is not guaranteed; use ``--verify-against``
to quantify any difference against a trusted copy.

Usage
-----
    python build_mta_manhattan.py --raw data/raw/MTA_Subway_Hourly_Ridership_2020-2024.csv \
        --out data/data_MTA_8H_Manhattan.csv [--verify-against path/to/trusted.csv]
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

# Date range of the training file: 2022-01-01 onwards (2020-2021 are excluded).
START_DATE = "2022-01-01"

EXPECTED = {
    "stations": 123,
    "rows": 403469,
    "first": "2022-01-01 16:00:00",
    "last": "2024-12-31 16:00:00",
    "md5": "c5b182823062e9f469ea82858fca1fe5",
}


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build(raw_csv, out_csv):
    """Build the 8-hour Manhattan ridership file from the raw hourly export."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw MTA file not found: {raw_csv}")

    print("[1/4] Reading raw MTA data...")
    # Read only the columns we need; the raw file is large (several GB).
    cols = ["transit_timestamp", "station_complex_id", "borough", "ridership"]
    df = pd.read_csv(raw_csv, usecols=cols, low_memory=False)
    print(f"      Loaded {len(df):,} rows")

    print("[2/4] Cleaning and scaling ridership...")
    # Ridership can contain comma-formatted strings (e.g. "1,234"); strip and convert.
    df["ridership"] = (
        df["ridership"].astype(str).str.replace(",", "", regex=False)
    )
    df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce")
    df = df.dropna(subset=["ridership"])
    # Scale to thousands of riders.
    df["ridership"] = df["ridership"] / 1000.0

    print("[3/4] Parsing timestamps and filtering Manhattan stations...")
    # Try the known MTA format first; fall back to pandas' generic parser for any mismatches.
    ts = pd.to_datetime(df["transit_timestamp"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    fallback = ts.isna()
    if fallback.any():
        ts.loc[fallback] = pd.to_datetime(df.loc[fallback, "transit_timestamp"], errors="coerce")
    df["transit_timestamp"] = ts
    df = df.dropna(subset=["transit_timestamp"])

    # Filter to the Manhattan borough and the desired date range.
    df["station_complex_id"] = df["station_complex_id"].astype(str)
    df = df[df["borough"] == "Manhattan"].copy()
    df = df[df["transit_timestamp"] >= pd.Timestamp(START_DATE)].copy()

    n_stations = df["station_complex_id"].nunique()
    print(f"      Manhattan stations found: {n_stations}")
    print(f"      Date range after filter: {df['transit_timestamp'].min()} -> {df['transit_timestamp'].max()}")

    print("[4/4] Bucketing into 8-hour periods and saving...")
    # Floor every timestamp to the nearest 8-hour boundary (00:00, 08:00, 16:00).
    df["time_period"] = df["transit_timestamp"].dt.floor("8h")

    # Sum ridership within each (time_period, station) bucket.
    df_8h = (
        df.groupby(["time_period", "station_complex_id"], as_index=False)["ridership"]
        .sum()
        .sort_values(["station_complex_id", "time_period"])
        .reset_index(drop=True)
    )

    # Calendar features used by the models.
    df_8h["hour"]        = df_8h["time_period"].dt.hour
    df_8h["day_of_week"] = df_8h["time_period"].dt.dayofweek
    df_8h["month"]       = df_8h["time_period"].dt.month

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir, exist_ok=True)
    df_8h.to_csv(out_csv, index=False)
    print(f"\n[Done] Saved to {out_csv}")
    print(f"       Stations  : {df_8h['station_complex_id'].nunique()}  (expected {EXPECTED['stations']})")
    print(f"       Rows      : {len(df_8h):,}  (expected {EXPECTED['rows']:,})")
    print(f"       Date range: {df_8h['time_period'].min()} -> {df_8h['time_period'].max()}"
          f"  (expected {EXPECTED['first']} -> {EXPECTED['last']})")
    print(f"       md5       : {md5_of(out_csv)}  (expected {EXPECTED['md5']})")
    return df_8h


def _load_for_compare(path):
    # station_complex_id mixes numeric-looking ids ("10") with alphanumeric ones
    # ("TRAM1"); force str so both files are compared on the same footing.
    d = pd.read_csv(path, dtype={"station_complex_id": str})
    d["time_period"] = pd.to_datetime(d["time_period"])
    return d


def verify(out_csv, ref_csv):
    """Compare the freshly built file with a trusted copy; print what matches."""
    print(f"\n[Verify] {out_csv}  vs  {ref_csv}")
    same_md5 = md5_of(out_csv) == md5_of(ref_csv)
    print(f"  byte-identical (md5): {same_md5}")
    a = _load_for_compare(out_csv)
    b = _load_for_compare(ref_csv)
    print(f"  shape:   built {a.shape}  reference {b.shape}  match={a.shape == b.shape}")
    print(f"  columns: match={list(a.columns) == list(b.columns)}  ({list(a.columns)})")
    print(f"  stations: built {a['station_complex_id'].nunique()}  reference {b['station_complex_id'].nunique()}")
    key = ["time_period", "station_complex_id"]
    m = a.merge(b, on=key, how="outer", suffixes=("_built", "_ref"), indicator=True)
    only_built = int((m["_merge"] == "left_only").sum())
    only_ref = int((m["_merge"] == "right_only").sum())
    print(f"  (time_period, station) pairs only in built: {only_built}; only in reference: {only_ref}")
    both = m[m["_merge"] == "both"]
    if len(both):
        max_abs = float(np.nanmax(np.abs(both["ridership_built"] - both["ridership_ref"])))
        print(f"  max abs ridership difference on common pairs: {max_abs:.6g}")
        for c in ["hour", "day_of_week", "month"]:
            n_bad = int((both[f"{c}_built"] != both[f"{c}_ref"]).sum())
            print(f"  {c}: {n_bad} mismatching rows")
    ok = same_md5 or (a.shape == b.shape and only_built == 0 and only_ref == 0 and max_abs == 0.0)
    print(f"  RESULT: {'MATCH' if ok else 'DIFFERENT'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="Raw MTA hourly ridership CSV (Socrata export).")
    p.add_argument("--out", required=True, help="Where to write data_MTA_8H_Manhattan.csv.")
    p.add_argument("--verify-against", default=None,
                   help="Optional trusted copy of the output to compare against.")
    args = p.parse_args()
    build(args.raw, args.out)
    if args.verify_against:
        verify(args.out, args.verify_against)


if __name__ == "__main__":
    main()
