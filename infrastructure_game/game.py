"""
game.py -- the Infrastructure Security Game and its test-time evaluation
(Table 1 "Utility" column and the per-window utilities behind Figure 2(b) of
"Strategic Decision Focused Learning"). Originally game_run_vmta_final.py.

One game is played per 8-hour window. The defender (one patrol unit) commits
to a mixed strategy p over the N = 123 Manhattan stations; the attacker
observes p and attacks one station. With V the ridership vector of the window
(thousands of riders), rows = patrolled station d, columns = attacked station a:

    attacker  A[d, a] = V[a]   if d != a   (attack succeeds)
              A[a, a] = -10              (attacker caught: fixed penalty)
    defender  D[d, a] = -V[a]  if d != a   (station lost)
              D[a, a] = +V[a]            (station protected)

Evaluation of a forecast V_hat (evaluate_game_performance): the Strong
Stackelberg Equilibrium strategy p is computed on the payoffs built from V_hat
(solvers/nfg_sse_lp.py), the attacker then best-responds to p under the TRUE
payoffs, and the defender's true expected utility p . D_true[:, t*] is
recorded. Utilities are <= 0 (0 only if the attacker is deterred onto an empty
station). Table 1 reports the mean over the 1,098 test windows of 2024;
train.py divides the total returned here by the number of test windows.
"""
import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. UTILITY MATRIX GENERATOR
# ==========================================
def gen_mta_utility_matrices(ridership_vector, use_gpu=False):
    """
    Build the (defender, attacker) payoff matrices of one window.

    Inputs
        ridership_vector : length-N array or Series of ridership values, in the
            row order chosen by the caller (the evaluator sorts by station id).
        use_gpu : if True and CUDA is available, build the matrices with torch
            in float32 on the GPU and copy them back. The paper run used the
            numpy float64 path below (MTA_EVAL_USE_GPU unset).
    Returns
        (D, A): two N x N arrays, conventions as in the module docstring.
    """
    values = ridership_vector.values if hasattr(ridership_vector, "values") else ridership_vector
    V = np.asarray(values, dtype=float)
    N = len(V)

    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                v = torch.tensor(V, device="cuda", dtype=torch.float32)
                A_t = v.unsqueeze(0).repeat(N, 1)   # A[d, a] = V[a] for every row d ...
                D_t = -A_t.clone()                  # D[d, a] = -V[a] ...
                idx = torch.arange(N, device="cuda")
                A_t[idx, idx] = -10.0               # ... diagonal: attacker caught
                D_t[idx, idx] = v                   # ... diagonal: station protected
                return D_t.cpu().numpy(), A_t.cpu().numpy()
        except Exception:
            pass

    # numpy path: every row is a copy of V, then the diagonal is overwritten.
    A_t = np.tile(V, (N, 1))
    D_t = -A_t.copy()
    np.fill_diagonal(A_t, -10.0)
    np.fill_diagonal(D_t, V)
    return D_t, A_t

# ==========================================
# 2. SOLVER WORKER
# ==========================================
def _solve_snapshot_utility(snapshot_payload):
    """
    Evaluate one window. snapshot_payload = (D_gen, A_gen, D_real, A_real),
    the payoff matrices built from the forecast (gen) and from the truth (real).

    1. Solve the SSE LP on the forecast payoffs -> defender strategy p.
    2. The attacker best-responds to p under the TRUE payoffs, ties (within
       1e-6) broken in the defender's favour -> target t*.
    3. Return the defender's true expected utility p . D_real[:, t*] (float).

    The solver import is inside the function so that process-pool workers,
    which start fresh on macOS/Windows, resolve it themselves; the package
    root must be on sys.path (train.py and reconstruct_per_window.py add it).
    """
    D_gen, A_gen, D_real, A_real = snapshot_payload
    from solvers.nfg_sse_lp import solve_general_sum_normal_form_mta, calculate_actual_attacker_strat_mta
    _, best_defender_strategy, _ = solve_general_sum_normal_form_mta(D_gen, A_gen)
    _, defender_utility = calculate_actual_attacker_strat_mta(best_defender_strategy, D_real, A_real)
    return float(defender_utility)

# ==========================================
# 3. STRATEGIC EVALUATOR
# ==========================================
def evaluate_game_performance(pred_df, actual_df, num_eval_points=10,
                               max_workers=None, use_gpu=False,
                               collect_snapshots=False):
    """
    Total defender utility of a forecast over every window of actual_df.

    Inputs
        pred_df, actual_df : DataFrames with columns time_period,
            station_complex_id and ridership (forecast / truth). Every window
            present in actual_df is evaluated; num_eval_points is unused and
            kept only for signature compatibility.
        max_workers : process-pool size. None -> the MTA_EVAL_WORKERS
            environment variable, else os.cpu_count(); 0 or 1 -> solve in the
            calling process. Changes speed only (the LP solver itself is
            deterministic for a given window).
        use_gpu : build the payoff matrices on CUDA (also enabled by
            MTA_EVAL_USE_GPU=1); unset in the paper run.
        collect_snapshots : if True, also return one record per window.

    Returns
        float: the sum of per-window utilities, or
        (total, [{'time_period': t, 'defender_utility': u}, ...]) when
        collect_snapshots is True. 0.0 (or (0.0, [])) if no window could be
        paired. Any exception raised while solving prints "Snapshot Error"
        and exits the process (sys.exit), as in the original code.

    Used at test time by train.py and reconstruct_per_window.py, and during
    training by models/zoc.py, kkt.py and hybrid.py on their inner evaluation
    split (accept-or-revert and best-checkpoint decisions).
    """
    actual_f = actual_df
    pred_f   = pred_df

    all_unique_times = sorted(actual_f['time_period'].unique())
    eval_times = all_unique_times

    if max_workers is None:
        env_workers = os.getenv("MTA_EVAL_WORKERS")
        max_workers = int(env_workers) if env_workers else (os.cpu_count() or 32)
    if not use_gpu:
        use_gpu = os.getenv("MTA_EVAL_USE_GPU", "0") == "1"

    # One payload per window. Both frames are restricted to the window and
    # sorted by station id, so row i of the forecast and of the truth is the
    # same station. Windows absent from either frame are skipped.
    payloads_with_t = []
    for t in eval_times:
        t_actual = actual_f[actual_f['time_period'] == t].sort_values('station_complex_id')
        t_pred   = pred_f[pred_f['time_period'] == t].sort_values('station_complex_id')
        if len(t_actual) == 0 or len(t_pred) == 0:
            continue
        D_gen, A_gen   = gen_mta_utility_matrices(t_pred['ridership'],   use_gpu=use_gpu)
        D_real, A_real = gen_mta_utility_matrices(t_actual['ridership'], use_gpu=use_gpu)
        payloads_with_t.append((t, D_gen, A_gen, D_real, A_real))

    if len(payloads_with_t) == 0:
        return (0.0, []) if collect_snapshots else 0.0

    try:
        times_ordered = [t for t, *_ in payloads_with_t]
        payloads      = [(D_gen, A_gen, D_real, A_real)
                         for _, D_gen, A_gen, D_real, A_real in payloads_with_t]

        # Windows are independent: solve them in a process pool (ex.map keeps
        # the window order). Each window costs one LP enumeration (N LPs with
        # N variables each, HiGHS) plus one best-response computation.
        if max_workers and max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                utils = list(ex.map(_solve_snapshot_utility, payloads))
        else:
            utils = [_solve_snapshot_utility(p) for p in payloads]

        total = float(np.sum(utils))
        if collect_snapshots:
            snapshot_records = [{'time_period': t, 'defender_utility': u}
                                 for t, u in zip(times_ordered, utils)]
            return total, snapshot_records
        return total

    except Exception as e:
        print(f"Snapshot Error: {e}")
        sys.exit()
