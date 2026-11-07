"""
solvers/nfg_sse_lp.py -- Strong Stackelberg Equilibrium (SSE) of one window of
the Infrastructure Security Game by LP enumeration, and the attacker's best
response to a committed defender strategy. Used by game.py for the test-time
evaluation (Table 1, Figure 2(b)) and by the ZOC / KKT / Hybrid models through
game.evaluate_game_performance on their inner evaluation split. The KKT and
Hybrid training step (models/kkt.py, models/hybrid.py) solves the same
single-target LP with scipy directly so that the enumeration can run in threads.

Payoff conventions (see game.py): N x N matrices, rows = patrolled station,
columns = attacked station. The defender's strategy is a probability vector p
over the rows.
"""
import numpy as np
from scipy.optimize import linprog
import sys


def solve_general_sum_normal_form_mta(defender_matrix, attacker_matrix):
    """
    Strong Stackelberg Equilibrium by enumeration of the attacker's target.

    For every target t the LP
        maximise   p . D[:, t]
        subject to (A[:, t'] - A[:, t]) . p <= 0   for all t' != t   (t is a best response)
                   sum p = 1,  0 <= p <= 1
    is solved with scipy.optimize.linprog (HiGHS); the target with the largest
    optimal defender utility wins (a candidate must beat the incumbent by more
    than 1e-9; exact ties keep the first feasible target). Making the attacker
    indifferent and breaking ties in the defender's favour is the SSE
    convention.

    Inputs
        defender_matrix, attacker_matrix : N x N arrays (N >= 2).
    Returns
        (best_response_t, best_defender_strategy, best_defender_utility):
        the attacker's SSE target, the defender's mixed strategy (length-N
        array) and its predicted utility.
    Side effects: prints a diagnostic and exits the process (sys.exit(1)) if
    the best-response constraint set is empty (N < 2) or no target has a
    feasible LP, as in the original code.
    """
    m, n = defender_matrix.shape
    tie_break_tol = 1e-9
    best_defender_utility = -np.inf
    best_defender_strategy = None
    best_response_t = -1

    feasible_found = False

    for t in range(n):
        # linprog minimises, so the objective is -D[:, t].
        c = -defender_matrix[:, t]

        # IC Constraints: Attacker's utility at t >= utility at any t'
        A_ub = []
        for t_prime in range(n):
            if t_prime == t:
                continue
            # (A_s,t' - A_s,t) * p_s <= 0
            A_ub.append(attacker_matrix[:, t_prime] - attacker_matrix[:, t])

        # Strict input check: this should not be empty in your intended setup.
        if len(A_ub) == 0:
            print(f"\n{'!'*60}")
            print("CRITICAL INPUT ERROR: Empty IC constraint set for linprog.")
            print("This usually means an invalid/degenerate snapshot was passed in.")
            print(f"Loop target t: {t}")
            print(f"Matrix dimensions (m x n): {m} x {n}")
            print(f"Defender matrix shape: {defender_matrix.shape}")
            print(f"Attacker matrix shape: {attacker_matrix.shape}")
            print("Expected at least 2 attacker actions (n >= 2).")
            print(f"{'!'*60}")
            sys.exit(1)

        A_ub = np.array(A_ub)
        b_ub = np.zeros(len(A_ub))

        # Probability constraints
        A_eq = [np.ones(m)]
        b_eq = [1]
        bounds = [(0, 1) for _ in range(m)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            feasible_found = True
            current_defender_utility = -res.fun
            if current_defender_utility > best_defender_utility + tie_break_tol:
                best_defender_utility = current_defender_utility
                best_defender_strategy = res.x
                best_response_t = t
            elif abs(current_defender_utility - best_defender_utility) <= tie_break_tol:
                # If numerically tied on defender utility, keep first feasible
                # candidate (any tied attacker response is defender-equivalent).
                pass

    # --- CRITICAL FAILURE CHECK ---
    if not feasible_found:
        print(f"\n{'!'*60}")
        print("CRITICAL SOLVER FAILURE: No feasible SSE strategy found.")
        print(f"Matrix Dimensions: {m}x{n}")
        print(f"Defender Matrix Max/Min: {defender_matrix.max():.2f}/{defender_matrix.min():.2f}")
        print(f"Attacker Matrix Max/Min: {attacker_matrix.max():.2f}/{attacker_matrix.min():.2f}")
        print("Possible causes: NaN values in ridership or degenerate payoff structures.")
        print(f"{'!'*60}")
        sys.exit(1) # Pause everything and exit


    return best_response_t, best_defender_strategy, best_defender_utility

def calculate_actual_attacker_strat_mta(defender_prob_vector, D_real, A_real):
    """
    Attacker's best response to a committed defender strategy, and the
    defender's resulting utility, under the TRUE payoffs.

    Inputs
        defender_prob_vector : length-N strategy p (from the LP solved on the
            forecast payoffs).
        D_real, A_real : N x N payoff matrices built from the true ridership.
    Returns
        (t_star, defender_utility) with defender_utility = p . D_real[:, t_star].

    The attacker's expected utility of target t is (p @ A_real)[t]. Every
    target within 1e-6 of the maximum counts as a best response (at the SSE
    the attacker is typically exactly indifferent and floating-point dot
    products of theoretically equal values differ by ~1e-7, so 1e-9 would miss
    genuine ties). Among tied targets the one with the largest defender
    utility is chosen: the defender-favoured (Strong Stackelberg) convention,
    consistent with how solve_general_sum_normal_form_mta found p.
    """
    attacker_expected_utilities = defender_prob_vector @ A_real

    # Find all attacker best responses (within numerical tolerance)
    max_atk_util = float(np.max(attacker_expected_utilities))
    # 1e-6 tolerance: floating point dot-products that are theoretically equal
    # can differ by ~1e-7, so 1e-9 is too tight and misses genuine ties.
    tie_tol = 1e-6
    tied_targets = np.where(
        np.abs(attacker_expected_utilities - max_atk_util) <= tie_tol
    )[0]

    if len(tied_targets) == 1:
        t_star = int(tied_targets[0])
    else:
        # Defender-favored tie-breaking: pick the tied target that maximises
        # the defender's expected utility under the current patrol strategy p.
        def_utils = np.array(
            [float(defender_prob_vector @ D_real[:, t]) for t in tied_targets]
        )
        t_star = int(tied_targets[np.argmax(def_utils)])

    actual_defender_utility = float(defender_prob_vector @ D_real[:, t_star])
    return t_star, actual_defender_utility
