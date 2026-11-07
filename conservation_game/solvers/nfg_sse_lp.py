"""
Strong Stackelberg equilibrium (SSE) of a general-sum normal-form game by LP
enumeration, and the attacker's best response to a committed defender mix.

Used by game.py (evaluation of every test window, Table 1 and Figure 2) and,
inside training, by models/kkt.py and models/hybrid.py to find the attacker
target before the differentiable re-solve. Original file: solvers/nfg_sse_lp.py.

Conventions. The defender picks a mixed strategy p over the N targets (rows),
the attacker then picks one target t (columns). defender_matrix[d, t] and
attacker_matrix[d, t] are the two players' payoffs when the defender covers d
and the attacker hits t. The `_mta` suffix on the two functions is historical
(they were first written for the infrastructure game); the same solver serves
both games in the paper.
"""

import numpy as np
from scipy.optimize import linprog
import sys


def solve_general_sum_normal_form_mta(defender_matrix, attacker_matrix):
    """
    Strong Stackelberg equilibrium by enumerating the attacker's target.

    For each target t solve the LP
        max_p  p . defender_matrix[:, t]
        s.t.   p . (attacker_matrix[:, t'] - attacker_matrix[:, t]) <= 0  for all t' != t
               sum(p) = 1,  0 <= p <= 1
    (scipy linprog, HiGHS). The constraints make t a best response of the
    attacker to p; ties in the attacker's payoff are allowed, which is the
    "strong" (defender-favoured) convention. The t with the largest LP value
    wins; a later t replaces the incumbent only if it is better by more than
    1e-9.

    Inputs:  defender_matrix, attacker_matrix: (m x n) numpy arrays, n >= 2.
    Returns: (best_response_t, best_defender_strategy, best_defender_utility)
             best_defender_strategy is the full probability vector (length m).
    Side effects: prints a diagnostic and calls sys.exit(1) if n < 2 or if no
    target gives a feasible LP (never happens for the payoff matrices built in
    game.py, where every target is attackable).
    """
    m, n = defender_matrix.shape
    tie_break_tol = 1e-9
    best_defender_utility = -np.inf
    best_defender_strategy = None
    best_response_t = -1

    feasible_found = False

    for t in range(n):
        # Objective: linprog minimises, so minimise -p . D[:, t].
        c = -defender_matrix[:, t]

        # Incentive-compatibility rows: (A[:, t'] - A[:, t]) . p <= 0 for t' != t,
        # i.e. the attacker gains at least as much from t as from any other t'.
        A_ub = []
        for t_prime in range(n):
            if t_prime == t:
                continue
            A_ub.append(attacker_matrix[:, t_prime] - attacker_matrix[:, t])

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

        # Simplex constraints on p.
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
                # Numerically tied on defender utility: keep the first feasible
                # candidate (any tied attacker response is defender-equivalent).
                pass

    if not feasible_found:
        print(f"\n{'!'*60}")
        print("CRITICAL SOLVER FAILURE: No feasible SSE strategy found.")
        print(f"Matrix Dimensions: {m}x{n}")
        print(f"Defender Matrix Max/Min: {defender_matrix.max():.2f}/{defender_matrix.min():.2f}")
        print(f"Attacker Matrix Max/Min: {attacker_matrix.max():.2f}/{attacker_matrix.min():.2f}")
        print("Possible causes: NaN values in the targets or degenerate payoff structures.")
        print(f"{'!'*60}")
        sys.exit(1)


    return best_response_t, best_defender_strategy, best_defender_utility


def calculate_actual_attacker_strat_mta(defender_prob_vector, D_real, A_real):
    """
    Attacker best response to a committed defender mix, and the defender's
    resulting expected utility, under the given (true) payoff matrices.

    Inputs:  defender_prob_vector: length-m probability vector p.
             D_real, A_real: (m x n) true defender / attacker payoff matrices.
    Returns: (t_star, defender_utility) with t_star the attacked target and
             defender_utility = p . D_real[:, t_star].

    Tie-breaking is defender-favoured (strong Stackelberg convention). At an
    SSE the attacker is typically exactly indifferent between several targets;
    among the targets whose expected attacker payoff is within 1e-6 of the
    maximum, the one with the highest defender payoff is chosen. A raw argmax
    would pick the lowest index and can give a utility far below the SSE value.
    The 1e-6 tolerance (rather than 1e-9) absorbs floating-point differences of
    order 1e-7 between dot products that are equal in exact arithmetic.
    """
    attacker_expected_utilities = defender_prob_vector @ A_real

    max_atk_util = float(np.max(attacker_expected_utilities))
    tie_tol = 1e-6
    tied_targets = np.where(
        np.abs(attacker_expected_utilities - max_atk_util) <= tie_tol
    )[0]

    if len(tied_targets) == 1:
        t_star = int(tied_targets[0])
    else:
        # Among the attacker's tied best responses, take the one that is best
        # for the defender under the committed strategy p.
        def_utils = np.array(
            [float(defender_prob_vector @ D_real[:, t]) for t in tied_targets]
        )
        t_star = int(tied_targets[np.argmax(def_utils)])

    actual_defender_utility = float(defender_prob_vector @ D_real[:, t_star])
    return t_star, actual_defender_utility
