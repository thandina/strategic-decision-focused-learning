"""
Conservation Security Game: payoff matrices and the strategic evaluator.

Serves the "utility" column of Table 1 (conservation game) and, through
reconstruct_per_window.py, the per-day series behind Figure 2. It is also the
accept-or-revert / checkpoint-selection score used inside models/zoc.py,
models/kkt.py and models/hybrid.py (always on their held-out D_eval days,
never on the test set). Original file: game_run_vgsg2.py.

Game (one window = one day, N = 9 spatial cells, V = elephant count per cell):
  * the defender commits to a mixed strategy p over the N cells (one patrol);
  * the attacker (poacher) then hits one cell;
  * payoffs: defender D[d, a] = +V[a] if d == a else -V[a];
             attacker A[d, a] = -10   if d == a else +V[a]   (caught: -10).
  * the defender's p is the strong Stackelberg equilibrium of the game built
    from the PREDICTED counts; the attacker best-responds to that p under the
    TRUE counts; the defender's utility is p . D_true[:, t*].
Utility is non-positive in practice: 0 is the ceiling and is reached when the
attacker is deterred onto a cell whose true count is 0.
"""
import numpy as np
import sys


def gen_gsg2_utility_matrices(elephant_count_vector):
    """
    Build the (N x N) defender and attacker payoff matrices from one window's
    per-cell counts.

    Input:   elephant_count_vector: pandas Series of length N (true or
             predicted counts), already sorted by cell_id.
    Returns: (D_t, A_t) numpy arrays with
             D_t[d, a] = +V[a] if d == a else -V[a]
             A_t[d, a] = -10.0 if d == a else  V[a]
    Rows index the defender's covered cell d, columns the attacked cell a.
    """
    N = len(elephant_count_vector)
    A_t = np.zeros((N, N))
    D_t = np.zeros((N, N))
    V = elephant_count_vector.values

    for d in range(N):
        for a in range(N):
            if d == a:
                # Attacker caught: fixed penalty; defender saves the cell's value.
                A_t[d, a] = -10.0
                D_t[d, a] = V[a]
            else:
                # Attacker takes the cell's value; defender loses it.
                A_t[d, a] = V[a]
                D_t[d, a] = -V[a]
    return D_t, A_t


def evaluate_game_performance(pred_df, actual_df, num_eval_points=10, collect_snapshots=False):
    """
    Total defender utility over every window in `actual_df`.

    For each unique time_period (sorted): build the payoff matrices from the
    predicted counts and from the true counts; solve the SSE LP on the
    predicted game to get the defender mix p; let the attacker best-respond to
    p under the true game (defender-favoured ties, see solvers/nfg_sse_lp.py);
    add p . D_true[:, t*] to the running total.

    Inputs:  pred_df, actual_df: DataFrames with columns time_period, cell_id,
             elephant_count covering the same windows and cells (pred_df is the
             output of a model's generate()).
             num_eval_points: unused, kept for call compatibility.
             collect_snapshots: if True also return the per-window utilities.
    Returns: total utility (float), or (total, [{time_period, defender_utility}, ...])
             when collect_snapshots is True. Callers divide the total by the
             number of windows to get the per-window mean reported in Table 1.
    Side effects: a solver failure prints "Snapshot Error" and exits the process.
    """
    actual_f = actual_df
    pred_f = pred_df

    all_unique_times = sorted(actual_f['time_period'].unique())

    eval_times = all_unique_times
    snapshot_records = []

    total_defender_utility = 0
    valid_snapshots = 0

    for t in eval_times:
        # Sort by cell_id so that row/column i of both games is the same cell.
        t_actual = actual_f[actual_f['time_period'] == t].sort_values('cell_id')
        t_pred = pred_f[pred_f['time_period'] == t].sort_values('cell_id')

        if len(t_actual) == 0 or len(t_pred) == 0:
            continue

        D_gen, A_gen = gen_gsg2_utility_matrices(t_pred['elephant_count'])
        D_real, A_real = gen_gsg2_utility_matrices(t_actual['elephant_count'])

        try:
            from solvers.nfg_sse_lp import solve_general_sum_normal_form_mta, calculate_actual_attacker_strat_mta
            # Defender commits to the SSE mix of the PREDICTED game ...
            _, best_defender_strategy, _ = solve_general_sum_normal_form_mta(D_gen, A_gen)

            # ... and is scored against the attacker's best response in the TRUE game.
            a_star, defender_utility = calculate_actual_attacker_strat_mta(
                best_defender_strategy,
                D_real,
                A_real
            )

            total_defender_utility += defender_utility
            valid_snapshots += 1
            if collect_snapshots:
                snapshot_records.append({'time_period': t, 'defender_utility': float(defender_utility)})


        except Exception as e:
            print(f"Snapshot Error: {e}")
            sys.exit()
            continue

    if collect_snapshots:
        return total_defender_utility, snapshot_records
    return total_defender_utility
