"""
data_pipeline.py -- data loading and the train / validation / test split of the
Infrastructure Security Game (Table 1 and Figure 2(b) of "Strategic Decision
Focused Learning"). Originally the loader of ml_predict_vmta_v2.py; that file's
run_pipeline driver and stats/summary writers were an older experiment driver,
not used by the paper run, and are not included.

Input file (default ../data/data_MTA_8H_Manhattan.csv, built from the public
"MTA Subway Hourly Ridership 2020-2024" table): columns time_period (start of
an 8-hour window: 00:00, 08:00 or 16:00), station_complex_id (123 Manhattan
station complexes) and ridership (thousands of riders in that window),
2022-01-01 16:00 .. 2024-12-31 16:00.

load_and_prep_mta builds one row per (window, station) with
    features  month, day_of_week, hour, ridership_lag1..6 (the station's six
              previous windows; rows without a full lag history are dropped)
              and 123 one-hot station columns stn_<id>        -> 132 columns
    target    ridership
and splits by window:
    test        time_period >= 2024-01-01                       (1,098 windows)
    validation  15% of the pre-2024 windows, drawn once with
                np.random.seed(42) + np.random.shuffle (identical for every
                model and seed)
    train       the remaining pre-2024 windows
Feature scaling (StandardScaler) is done inside each model on its own
training rows.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 6   # lagged windows per station used as features

# ../data/data_MTA_8H_Manhattan.csv, resolved relative to this file.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data_MTA_8H_Manhattan.csv"

# ==========================================
# 1. Load Dataset
# ==========================================

def load_and_prep_mta(data_path=DEFAULT_DATA_PATH):
    """
    Load the ridership table and build the split described in the module docstring.

    Inputs
        data_path : CSV path (str or Path). Exits with status 1 if it is missing.
    Returns
        train_set, val_set, test_set : DataFrames (one row per window x station)
            with time_period, station_complex_id, ridership, hour, day_of_week,
            month, ridership_lag1..6 and the one-hot station columns
        station_ids  : sorted list of the 123 station ids (strings)
        feature_cols : the 132 feature column names, in model input order
    Side effects: prints the split sizes; calls np.random.seed(42) on the
    global numpy RNG to draw the validation windows.
    """
    data_path = str(data_path)
    if not os.path.exists(data_path):
        print(f"[Error] File {data_path} not found."); sys.exit(1)

    print(f"\n[System] Loading MTA Data ...")

    # Two accepted layouts. The shipped 8-hour file has a time_period column
    # and takes the else-branch. A raw hourly export (transit_timestamp column)
    # would be summed into 4-hour bins here, which is NOT the paper's 8-hour
    # setting; the branch is kept verbatim from the original code.
    header_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
    if 'transit_timestamp' in header_cols:
        cols = ['transit_timestamp', 'station_complex_id', 'ridership']
        df = pd.read_csv(data_path, usecols=cols, dtype={'station_complex_id': str})
        df['transit_timestamp'] = pd.to_datetime(df['transit_timestamp'])
        df['time_period'] = df['transit_timestamp'].dt.floor('4h')
        df_binned = df.groupby(['time_period', 'station_complex_id'])['ridership'].sum().reset_index()
    else:
        cols = ['time_period', 'station_complex_id', 'ridership']
        df_binned = pd.read_csv(data_path, usecols=cols, low_memory=False,
                                dtype={'station_complex_id': str})
        df_binned['time_period'] = pd.to_datetime(df_binned['time_period'])

    df_binned['station_complex_id'] = df_binned['station_complex_id'].astype(str)

    # Calendar features of the window.
    df_binned['hour']        = df_binned['time_period'].dt.hour
    df_binned['day_of_week'] = df_binned['time_period'].dt.dayofweek
    df_binned['month']       = df_binned['time_period'].dt.month

    # Lag features: the station's ridership in its previous 1..6 windows
    # (shift within station, in time order). The first 6 windows of each
    # station have no full history and are dropped by dropna().
    df_binned = df_binned.sort_values(['station_complex_id', 'time_period'])
    lag_cols = []
    for i in range(1, WINDOW_SIZE + 1):
        name = f'ridership_lag{i}'
        df_binned[name] = df_binned.groupby('station_complex_id')['ridership'].shift(i)
        lag_cols.append(name)

    df_final = df_binned.dropna().reset_index(drop=True)

    # --- One-hot encode station identity ---
    # Each station gets its own binary column. The MLP can then learn
    # different weights for different stations via the shared weight matrix.
    ohe = pd.get_dummies(df_final['station_complex_id'], prefix='stn').astype(int)
    ohe_cols = ohe.columns.tolist()
    df_final = pd.concat([df_final, ohe], axis=1)

    station_ids  = sorted(df_final['station_complex_id'].unique().tolist(), key=str)
    feature_cols = ['month', 'day_of_week', 'hour'] + lag_cols + ohe_cols

    # Test = calendar year 2024. Validation = 15% of the earlier windows,
    # chosen by a fixed shuffle (seed 42) so every model sees the same split.
    test_start = '2024-01-01'
    test_set  = df_final[df_final['time_period'] >= pd.Timestamp(test_start)].copy()
    pre_2024  = df_final[df_final['time_period'] <  pd.Timestamp(test_start)].copy()

    unique_times = pre_2024['time_period'].unique()
    val_size     = int(len(unique_times) * 0.15)
    np.random.seed(42)
    np.random.shuffle(unique_times)
    val_times   = unique_times[:val_size]
    train_times = unique_times[val_size:]

    val_set   = pre_2024[pre_2024['time_period'].isin(val_times)].copy()
    train_set = pre_2024[pre_2024['time_period'].isin(train_times)].copy()

    print(f"[System] Train/Val/Test Split Complete.")
    print(f"   - Training Rows:   {len(train_set):,}")
    print(f"   - Validation Rows: {len(val_set):,}")
    print(f"   - Testing Rows:    {len(test_set):,}")
    print(f"   - Feature Cols:    {len(feature_cols)} ({len(ohe_cols)} station OHE)")
    return train_set, val_set, test_set, station_ids, feature_cols

# ==========================================
# 2. Metrics
# ==========================================

def calculate_mta_metrics(pred_df, actual_df, k=10):
    """
    MAE and top-k recall between a forecast and the truth.

    Inputs
        pred_df, actual_df : DataFrames with ridership and station_complex_id;
            the MAE subtracts the two ridership columns aligned on the
            DataFrame index (predictions from a model's generate() keep the
            index of the frame they were made from).
        k : size of the top-k set (by ridership) used for the recall.
    Returns (mae, top_k_recall).
    Not called by train.py, which computes the Table 1 MAE after a merge on
    (time_period, station_complex_id); kept as the loader's companion metric.
    """
    mae = np.mean(np.abs(pred_df['ridership'] - actual_df['ridership']))
    actual_top_k = set(actual_df.nlargest(k, 'ridership')['station_complex_id'])
    pred_top_k   = set(pred_df.nlargest(k,   'ridership')['station_complex_id'])
    top_k_recall = len(actual_top_k & pred_top_k) / k
    return mae, top_k_recall
