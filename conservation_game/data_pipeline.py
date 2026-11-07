"""
Data loading, features, temporal split and the MAE metric for the conservation
game (Etosha National Park elephants, 9 spatial cells, daily windows).

Serves Table 1 (conservation game; both the utility and the MAE column use the
test set built here) and Figure 2. Original file: ml_predict_vgsg2.py, of which
only the loader and the metric are kept; its old driver (`run_pipeline`) was
not used for the paper.

Input file (data_ETOSHA_CELL_9.csv, not shipped with the code): one row per
(day, cell) with columns time_period, cell_id, elephant_count, day_of_week,
month, week_of_year. elephant_count is the number of tracked elephants whose
mean position that day fell in the cell.
"""
import os
import numpy as np
import pandas as pd

WINDOW_SIZE = 6             # lags of the target per cell used as features
TEST_START = "2013-02-24"   # first test day; the test set is every day from here on


def load_and_prep_gsg2(data_path, test_start=TEST_START):
    """
    Load the cell-count CSV, add lag features and split it into train /
    validation / test exactly as in the paper's run.

    Features: month, day_of_week, week_of_year and elephant_count_lag1..6
    (the previous 6 days' counts of the same cell); rows whose lags are not
    all available (the first 6 days of each cell) are dropped.
    Split: test = every day >= test_start. Of the remaining ("pre-test") unique
    dates, the first 15% in order of first appearance go to validation and the
    rest to training. The frame is sorted by cell then date, so "first 15%" is
    the earliest 15% of pre-test days: a chronological validation block, not a
    random one. This is deliberate: it is the split that produced Table 1.
    Scaling is NOT done here; each model fits its own StandardScaler on the
    rows it backpropagates on.

    Inputs:  data_path: path to the CSV. test_start: ISO date string.
    Returns: (train_set, val_set, test_set, cell_ids, feature_cols)
             three DataFrames (original index kept), the sorted list of cell
             ids (strings) and the list of 9 feature column names.
    Side effects: prints split statistics. Raises FileNotFoundError if the
    CSV is missing.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"\n[System] Loading GSG2 Cell Data ({data_path}) ...")

    df_binned = pd.read_csv(data_path)
    df_binned['time_period'] = pd.to_datetime(df_binned['time_period'])
    df_binned['cell_id'] = df_binned['cell_id'].astype(str)

    # Lag features: the cell's own count on each of the previous WINDOW_SIZE days.
    df_binned = df_binned.sort_values(['cell_id', 'time_period'])
    lag_cols = []
    for i in range(1, WINDOW_SIZE + 1):
        name = f'elephant_count_lag{i}'
        df_binned[name] = df_binned.groupby('cell_id')['elephant_count'].shift(i)
        lag_cols.append(name)

    df_final = df_binned.dropna().reset_index(drop=True)

    cell_ids = sorted(df_final['cell_id'].unique().tolist())
    feature_cols = ['month', 'day_of_week', 'week_of_year'] + lag_cols

    # Temporal split: test is everything from test_start on.
    test_set = df_final[df_final['time_period'] >= pd.Timestamp(test_start)].copy()

    pre_test = df_final[df_final['time_period'] < pd.Timestamp(test_start)].copy()

    unique_times = pre_test['time_period'].unique()
    val_size = int(len(unique_times) * 0.15)

    # No shuffle here, on purpose: the paper's validation set is the first 15%
    # of pre-test dates in file order (the earliest days). A seeded shuffle of
    # unique_times would give a different train/val split from Table 1.
    val_times = unique_times[:val_size]
    train_times = unique_times[val_size:]

    val_set = pre_test[pre_test['time_period'].isin(val_times)].copy()
    train_set = pre_test[pre_test['time_period'].isin(train_times)].copy()

    def _print_subset_stats(name, subset):
        print(f"\n{name}")
        print("-" * len(name))
        print(f"Rows: {len(subset):,}")
        print(f"Unique cells: {subset['cell_id'].nunique()}")
        if len(subset) > 0:
            print(
                "Time span: "
                f"{subset['time_period'].min()} -> {subset['time_period'].max()}"
            )
            print(f"Dates covered (unique days): {subset['time_period'].dt.date.nunique()}")
        else:
            print("Time span: N/A")
            print("Dates covered (unique days): 0")

    print(f"[System] Train/Val/Test Split Successful (first 15% of pre-test days = Val).")
    print(f"   - Training Rows:   {len(train_set):,}")
    print(f"   - Validation Rows: {len(val_set):,}")
    print(f"   - Testing Rows:    {len(test_set):,}")
    _print_subset_stats("Train", train_set)
    _print_subset_stats("Validation", val_set)
    _print_subset_stats("Test", test_set)
    return train_set, val_set, test_set, cell_ids, feature_cols


def calculate_gsg2_metrics(pred_df, actual_df, k=10):
    """
    MAE of the predicted per-cell counts, plus a top-k cell recall.

    Inputs:  pred_df: output of a model's generate() (same index as actual_df).
             actual_df: the matching true rows. k: size of the top-k sets.
    Returns: (mae, top_k_recall). mae is the mean absolute error over all rows
             (this is the MAE column of Table 1). top_k_recall is the overlap
             between the k largest predicted and the k largest true rows'
             cell ids divided by k; train.py discards it.
    """
    mae = np.mean(np.abs(pred_df['elephant_count'] - actual_df['elephant_count']))

    actual_top_k = set(actual_df.nlargest(k, 'elephant_count')['cell_id'])
    pred_top_k = set(pred_df.nlargest(k, 'elephant_count')['cell_id'])

    overlap = len(actual_top_k.intersection(pred_top_k))
    top_k_recall = overlap / k

    return mae, top_k_recall
