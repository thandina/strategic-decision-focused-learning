"""
models/hybrid.py -- the Hybrid model of Table 1 of "Strategic Decision Focused
Learning" ("Hybrid" in the paper; model key `hybrid_v3` in train.py;
originally regression_hybrid_mta_v3.py).

Training is cut into blocks ("regimes") of n_store = 50 epochs that alternate
between two objectives, with the accept-or-revert rule of ZOC after each block:

  ODD  blocks (Huber; block 1 is the warm-up): supervised Huber loss on all
      123 station predictions, in mini-batches of batch_time_periods = 8
      windows drawn from the max_periods_per_epoch = 32 windows sampled each
      epoch, Gaussian input noise (std noise_std = 0.01), gradient norm
      clipped at 1.0.
  EVEN blocks (KKT): the KKT step of models/kkt.py -- the top_k = 20 stations
      by predicted ridership (detached selection), the attacker target found
      with scipy/HiGHS on detached predictions, the LP re-solved through
      cvxpylayers (SCS, eps = solver_eps = 1e-2) so the strategy carries
      gradients, loss = -mean utility + huber_weight * mean Huber over all
      123 stations, gradient norm clipped at 1.0.

  huber_weight anneals linearly from 5.0 to 3.0 over 300 epochs and then
  holds, but its clock starts after the warm-up block: 5.0 at epoch 51 (the
  first KKT epoch), 3.0 at epoch 350 and afterwards (epochs 1..50 are Huber
  only, so the weight is unused there). Values set by train.py.

  Accept-or-revert: every block starts from the best weights so far; at its
  end the game utility of the current weights on the inner evaluation split
  D_eval (15% of the training windows, split with RandomState(random_seed))
  is compared with the best so far: strictly better -> ACCEPT (new best),
  otherwise -> REVERT. The returned model is the best accepted block. With
  500 epochs: blocks 1-50 Huber, 51-100 KKT, 101-150 Huber, ..., 451-500 KKT.

The optimiser (AdamW lr 5e-4, weight_decay 0.01) and the LR scheduler
(ReduceLROnPlateau on the validation Huber loss) persist across blocks; only
the weights are reverted. All 123 stations enter training; no station is
pre-filtered by variance.

cvxpy and cvxpylayers are required to instantiate the class (the import
failure is deferred to __init__ so that this module imports without them).

Execution order:
  get_runtime_hyperparams()                 called by train.py to get defaults
  REGRESSION_HYBRID_MTA_V3.__init__()       builds MLP, scaler, LP layer slot
  REGRESSION_HYBRID_MTA_V3.train()          runs the regime-alternating loop
  REGRESSION_HYBRID_MTA_V3.generate()       predictions on a held-out set
  REGRESSION_HYBRID_MTA_V3.get_loss()       Huber validation loss (used inside train)
"""

import os
import sys
import importlib
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler

# cvxpy and cvxpylayers power the differentiable LP in even (KKT) regimes.
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


# =============================================================================
# Payoff matrix + LP helper functions
# (Identical to models/kkt.py)
# =============================================================================

def _build_payoff_tensors(V_hat: torch.Tensor):
    """
    Build the N×N defender (D) and attacker (A) payoff matrices from a
    ridership vector V_hat.

    The game is a Stackelberg Security Game (SSG):
      D[i, j] = +V_hat[j]  if i == j   (defender patrols j, attacker hits j → wins V[j])
      D[i, j] = -V_hat[j]  if i != j   (defender misses → loses V[j])
      A[i, j] = +V_hat[j]  if i != j   (attacker succeeds → gains V[j])
      A[i, j] = -10.0      if i == j   (attacker is caught → fixed penalty)

    Parameters
    ----------
    V_hat : torch.Tensor, shape (N,)
        Predicted ridership for N stations. May carry gradients.

    Returns
    -------
    D : torch.Tensor, shape (N, N)  — defender payoff matrix
    A : torch.Tensor, shape (N, N)  — attacker payoff matrix
    """
    # Read station count and device from the input tensor.
    N   = V_hat.shape[0]
    eye = torch.eye(N, device=V_hat.device)

    # sgn[i,j] = 2·I[i,j] - 1  →  +1 if i==j,  -1 if i≠j
    # D[i,j] = V[j] · sgn[i,j]  =  +V[j] if i==j (catch),  -V[j] if i≠j (miss)
    D   = V_hat.unsqueeze(0) * (2.0 * eye - 1.0)

    # (1-I)[i,j] = 1 if i≠j,  0 if i==j
    # A[i,j] = V[j]·(1-I[i,j]) - 10·I[i,j]
    #        = +V[j] if i≠j (attack succeeds),  -10 if i==j (attacker caught)
    A   = V_hat.unsqueeze(0) * (1.0 - eye) - 10.0 * eye

    return D, A


def _build_lp_params(V_hat: torch.Tensor, target_t: int):
    """
    Extract the LP objective vector and constraint matrix for a fixed attacker
    target t, ready to pass into the cvxpylayers Stackelberg layer.

    The Stackelberg LP for fixed target t:
        maximise   D[:, t] · p
        subject to (A[:, s] - A[:, t]) · p <= 0   for all s != t
                   sum(p) == 1,  p >= 0

    Parameters
    ----------
    V_hat    : torch.Tensor, shape (N,)  — predicted ridership (carries gradients)
    target_t : int                        — attacker's candidate pure strategy

    Returns
    -------
    d_col   : torch.Tensor, shape (N,)       — LP objective coefficients
    br_rows : torch.Tensor, shape (N-1, N)   — best-response constraint matrix
    """
    # Build full payoff matrices from the current (gradient-carrying) V_hat.
    D, A = _build_payoff_tensors(V_hat)
    N    = V_hat.shape[0]

    # d_col[i] = D[i, t]  =  +V[t] if i==t,  -V[t] if i≠t
    # LP objective:  max_p  d_col · p  =  max_p  Σ_i p[i]·D[i,t]
    d_col = D[:, target_t]

    # br_rows[k] = A[:,s_k] - A[:,t]   for s_k ≠ t
    # Constraint:  Σ_i p[i]·A[i,s_k] ≤ Σ_i p[i]·A[i,t]  (t is attacker BR to p)
    br_rows = torch.stack([A[:, tp] - A[:, target_t] for tp in range(N) if tp != target_t])

    return d_col, br_rows


def _make_stackelberg_layer(N: int) -> "CvxpyLayer":
    """
    Build a cvxpylayers-differentiable LP layer for the Stackelberg subproblem.

    LP formulation (fixed attacker target t):
        maximise   d_col · p
        subject to br_matrix · p <= 0     (attacker BR constraints)
                   sum(p) == 1            (simplex constraint)
                   p >= 0                 (non-negativity, via nonneg=True)

    Parameters
    ----------
    N : int — number of stations

    Returns
    -------
    CvxpyLayer — differentiable LP layer accepting (d_col, br_matrix) and
                 returning the optimal defender mixed strategy p*.
    """
    # Defender mixed strategy variable: N-dimensional probability vector.
    p = cp.Variable(N, nonneg=True)

    # Declare CVXPY parameters that will be filled with torch tensors at solve time.
    d_col_p  = cp.Parameter(N,        name="d_col")     # objective vector
    br_mat_p = cp.Parameter((N-1, N), name="br_matrix") # constraint matrix

    # LP:  max_{p ≥ 0}  d_col · p
    #      s.t.  br_matrix · p ≤ 0     (N-1 attacker best-response constraints)
    #            1ᵀp = 1               (probability simplex)
    #            p ≥ 0                 (via nonneg=True)
    prob = cp.Problem(cp.Maximize(d_col_p @ p), [br_mat_p @ p <= 0, cp.sum(p) == 1])

    # Wrap in CvxpyLayer for PyTorch autograd integration.
    return CvxpyLayer(prob, parameters=[d_col_p, br_mat_p], variables=[p])


# =============================================================================
# Neural network architecture
# =============================================================================

class MTARidershipMLP(nn.Module):
    """
    3-layer MLP predicting ridership for a single station from its feature vector.

    Architecture:
        Linear(input_dim → hidden_dim) → LayerNorm → SiLU → Dropout(0.2)
        → Linear(hidden_dim → hidden_dim//2) → SiLU
        → Linear(hidden_dim//2 → 1) → ReLU

    Final ReLU enforces non-negative predictions.
    squeeze(-1) in forward() returns shape (B,) instead of (B, 1) — convenient
    for the topk selection in the training loop (the other models return (B, 1)).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Parameters
        ----------
        input_dim  : int — number of input features per row
        hidden_dim : int — width of the first hidden layer
        """
        super().__init__()
        # Full feedforward stack in a single Sequential for clean forward().
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),        # expand input to hidden space
            nn.LayerNorm(hidden_dim),                # stabilise activations (important for noisy LP gradients)
            nn.SiLU(),                               # smooth non-linearity
            nn.Dropout(0.2),                         # regularisation
            nn.Linear(hidden_dim, hidden_dim // 2),  # halve width
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),           # scalar ridership output
            nn.ReLU(),                               # enforce non-negativity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim) or (B, window, input_dim)

        Returns
        -------
        torch.Tensor, shape (B,)  — predicted ridership per row (squeezed).
        """
        # Flatten any windowed dimensions before passing through the network.
        x = x.reshape(x.size(0), -1)
        # squeeze(-1) converts (B, 1) → (B,) for direct use in topk and scatter ops.
        return self.network(x).squeeze(-1)


# =============================================================================
# Hyperparameters
# =============================================================================

def get_runtime_hyperparams() -> dict:
    """
    Return the default hyperparameter dictionary for REGRESSION_HYBRID_MTA_V3.

    Called by train.py before instantiation. train.py overrides, for the paper
    run: epochs 500, random_seed (--seed), top_k 20 (--top_k),
    max_periods_per_epoch 32, lp_workers (--lp_workers; 6 in the launch
    script), huber_weight 5.0, huber_weight_end 3.0 (--huber_weight_end),
    huber_anneal_epochs 300. Unchanged defaults that ran: hidden_dim 256,
    lr 5e-4, noise_std 0.01, n_store 50, batch_time_periods 8,
    split_ratio 0.85, solver_eps 1e-2.

    Returns
    -------
    dict with keys:
        hidden_dim           : int   — MLP hidden width
        epochs               : int   — total training epochs across all regimes
        lr                   : float — AdamW learning rate
        noise_std            : float — std of Gaussian noise injected in Huber regimes
        n_store              : int   — epochs per regime (both Huber and KKT)
        batch_time_periods   : int   — periods per gradient step (KKT phase)
        max_periods_per_epoch: int   — cap on periods per KKT epoch
        split_ratio          : float — fraction of periods used for gradient updates
        random_seed          : int   — reproducibility seed
        solver_eps           : float — SCS tolerance in differentiable LP solves
        top_k                : int   — N stations entering the LP each step
        lp_workers           : int   — ThreadPoolExecutor workers for Pass 1 (0 = auto)
    """
    return {
        'hidden_dim': 256,
        'epochs': 300,
        'lr': 5e-4,
        'noise_std': 0.01,       # small noise injected during Huber phase for regularisation
        'n_store': 50,           # length of each regime in epochs (regime 1: epochs 1-50, regime 2: 51-100, ...)
        'batch_time_periods': 8,
        'max_periods_per_epoch': 128,
        'split_ratio': 0.85,
        'random_seed': 42,
        'solver_eps': 1e-2,
        'top_k': 20,
        'lp_workers': 0,
        'huber_weight': 5.0,
        'huber_weight_end': 1.0,
        'huber_anneal_epochs': 300,
    }


# =============================================================================
# Main model class
# =============================================================================

class REGRESSION_HYBRID_MTA_V3:
    """
    Hybrid end-to-end model ("Hybrid" in Table 1): alternating Huber + KKT
    regimes with ZOC-style accept-or-revert checkpointing (MTA domain).

    Training alternates between:
      - Odd regimes  (Huber): supervised prediction loss with noise injection.
      - Even regimes (KKT):   differentiable LP strategic loss.

    At the end of each regime, the model is evaluated on a held-out eval split.
    The weights are accepted only if strategic utility improved; otherwise the
    model is reverted to the best prior checkpoint. This prevents any single
    bad regime from degrading the final model.
    """

    def __init__(self, params: dict, feature_cols: list, window_size: int):
        """
        Initialise the model, scaler, MLP, and LP layer slot.

        Parameters
        ----------
        params       : dict  — hyperparameters from get_runtime_hyperparams()
        feature_cols : list  — input feature column names
        window_size  : int   — lookback window (stored but MLP flattens inputs)

        Side effects
        ------------
        self.model     : MTARidershipMLP on best available device
        self.scaler    : sklearn StandardScaler (fitted during train())
        self._lp_layer : None until first KKT regime call
        self._lp_N     : None until first KKT regime call
        """
        # Guard: KKT regimes require the differentiable LP library.
        if not _CVXPY_AVAILABLE:
            raise ImportError("cvxpy and cvxpylayers are required for REGRESSION_HYBRID_MTA.")

        # Store config for use throughout training and inference.
        self.params       = params
        self.feature_cols = feature_cols
        self.window_size  = window_size

        # Scaler is fitted on the backprop split during train(); reused in generate().
        self.scaler = StandardScaler()

        # Detect best compute device (CUDA > MPS > CPU).
        from utils import best_device
        self.device = best_device()

        # Build the MLP on the selected device.
        hidden_dim     = int(params.get('hidden_dim', 256))
        self.model     = MTARidershipMLP(len(feature_cols), hidden_dim).to(self.device)

        # LP layer and its cached size are built lazily in _ensure_lp_layer().
        self._lp_layer = None
        self._lp_N     = None

        self.n_lp_calls    = 0
        self.n_lp_failures = 0

    # -------------------------------------------------------------------------

    def _ensure_lp_layer(self, N: int):
        """
        Build (or rebuild) the Stackelberg LP layer if N has changed.

        Building the cvxpylayers layer compiles the CVXPY problem and is
        expensive; we cache it and only rebuild when the game size changes.

        Parameters
        ----------
        N : int — number of stations entering the LP

        Side effects
        ------------
        self._lp_layer : CvxpyLayer for size N
        self._lp_N     : N (cached for size comparison)
        """
        # Only rebuild if never built or if game size has changed.
        if self._lp_layer is None or self._lp_N != N:
            self._lp_layer = _make_stackelberg_layer(N)
            self._lp_N     = N

    # -------------------------------------------------------------------------

    def _get_eval_fn(self):
        """
        Import and return the strategic game evaluator function.

        Loads evaluate_game_performance from game.py at call time (lazy import;
        the package root must be on sys.path, which train.py ensures).

        Returns
        -------
        callable — game.evaluate_game_performance

        Raises
        ------
        SystemExit if game cannot be imported (critical failure).
        """
        try:
            return importlib.import_module("game").evaluate_game_performance
        except ImportError:
            print("CRITICAL ERROR: game module not found")
            sys.exit(1)

    # -------------------------------------------------------------------------

    def _utility_for_period(self, V_hat: torch.Tensor, V_actual: torch.Tensor, lp_workers: int) -> torch.Tensor:
        """
        Compute the differentiable defender utility for one time period.

        Used only during the KKT (even) training regime. Performs two LP solves:

        Pass 1 (non-differentiable, scipy/HiGHS): enumerate all N attacker targets
        in parallel to find t* — the one that maximises defender utility under
        current V_hat.

        Pass 2 (differentiable, cvxpylayers/SCS): re-solve the LP for t* so
        gradients can flow back through p* into V_hat and into the MLP weights.

        Then evaluate p* against ground-truth ridership V_actual to compute the
        defender's actual (not predicted) utility.

        Parameters
        ----------
        V_hat   : torch.Tensor, shape (N,)  — predicted ridership for top-K stations.
                  Must carry gradients.
        V_actual: torch.Tensor, shape (N,)  — ground-truth ridership for the same
                  stations. Used for utility evaluation only (no gradients).
        lp_workers: int — number of ThreadPoolExecutor workers for Pass 1.

        Returns
        -------
        torch.Tensor, scalar
            Defender utility under p* vs. true ridership.
            Carries gradients into V_hat. Returns 0.0 (no grad) on LP failure.
        """
        # Read station count and ensure the LP layer exists for this N.
        N = V_hat.shape[0]
        self._ensure_lp_layer(N)

        # ── Pass 1: parallel LP enumeration to find t* ───────────────────────
        with torch.no_grad():
            D_det, A_det = _build_payoff_tensors(V_hat.detach())
            D_np = D_det.cpu().numpy()
            A_np = A_det.cpu().numpy()

        def _solve_t(t):
            d_col_t = D_np[:, t]
            br_t    = np.array([A_np[:, tp] - A_np[:, t]
                                for tp in range(N) if tp != t])
            res = linprog(
                -d_col_t,
                A_ub=br_t, b_ub=np.zeros(N - 1),
                A_eq=[np.ones(N)], b_eq=[1.0],
                bounds=[(0, 1)] * N, method='highs',
            )
            return t, res

        best_util = -1e18
        best_t    = 0

        with ThreadPoolExecutor(max_workers=lp_workers) as pool:
            for t, res in pool.map(_solve_t, range(N)):
                if res.success and -res.fun > best_util:
                    best_util = -res.fun
                    best_t    = t
        # best_t = t* = SSE attacker target under current predictions.

        # ── Pass 2: differentiable solve for t* (cvxpylayers) ─────────────────
        # Build LP parameters using the live (gradient-carrying) V_hat for best_t.
        d_col, br_matrix = _build_lp_params(V_hat, best_t)

        # Solve via SCS through cvxpylayers. The KKT residuals from this solve
        # are stored internally, enabling gradient computation in backward().
        # Tolerance eps = params['solver_eps'] = 1e-2 in the paper run (unlike
        # models/kkt.py, which hard-codes 1e-4). A failed solve counts as a
        # failure and contributes a constant 0 (no gradient) for this window.
        self.n_lp_calls += 1
        try:
            (p_star,) = self._lp_layer(
                d_col, br_matrix,
                solver_args={"solve_method": "SCS", "eps": self.params.get('solver_eps', 1e-2)}
            )
        except Exception:
            self.n_lp_failures += 1
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        # Cast to float32 — cvxpylayers can return float64 on some backends.
        p_star = p_star.float()

        # ── Evaluate utility against ground-truth ridership ───────────────────
        # Build payoff matrices from actual (not predicted) ridership.
        D_actual, A_actual = _build_payoff_tensors(V_actual)

        # Attacker best response to p* under TRUE payoffs:
        #   t_actual = argmax_t  (p* @ A_actual)_t  =  argmax_t  Σ_i p*[i]·A_actual[i,t]
        # Detached — selecting t is a decision, not part of the gradient path.
        with torch.no_grad():
            t_actual = int((p_star.detach() @ A_actual).argmax().item())

        # Defender utility:
        #   u = p* · D_actual[:, t_actual]  =  Σ_i p*[i] · D_actual[i, t_actual]
        # Carries gradients: u → p* → LP KKT conditions → V_hat → MLP.
        return p_star @ D_actual[:, t_actual]

    # -------------------------------------------------------------------------

    def train(self, train_set, val_set):
        """
        Run the full alternating-regime Hybrid training loop.

        High-level steps:
          1. Split training time periods into a backprop set (85%) and an eval set (15%)
             with RandomState(random_seed); fit the scaler on the backprop rows.
          2. Pre-build per-period tensors (scaled features, true ridership clipped
             at 0, rows sorted by station id), shared by both phases.
          3. Loop over regimes in blocks of n_store epochs; each epoch samples
             max_periods_per_epoch windows and visits them in batches of
             batch_time_periods:
               Odd  regime (Huber): Huber loss per batch of time periods with
                   noise injection (4 steps of 8 windows with the paper settings).
               Even regime (KKT):  strategic LP loss per batch of time periods,
                   with dynamic top-K station selection.
          4. At the end of each regime, evaluate on the eval split.
             If strategic utility improved → ACCEPT new weights.
             Otherwise → REVERT to prior best weights.
          5. Periodically checkpoint the best weights for post-hoc analysis.
          6. At the end, load the globally best weights.

        Parameters
        ----------
        train_set : pd.DataFrame
            Training data. Must have 'time_period', 'station_complex_id',
            'ridership', plus all columns in self.feature_cols.
        val_set   : pd.DataFrame
            Validation data. Used for Huber loss monitoring and LR scheduling.

        Returns
        -------
        None. Updates self.model in-place; sets self.epoch_checkpoints.
        """
        # ── Extract scalar hyperparameters ────────────────────────────────────
        split_ratio = float(self.params.get('split_ratio', 0.85))
        seed        = int(self.params.get('random_seed', 42))
        top_k       = int(self.params.get('top_k', 20))

        # LP worker count — 0 means use all available cores.
        lp_workers_cfg = int(self.params.get('lp_workers', 0))
        lp_workers = lp_workers_cfg if lp_workers_cfg > 0 else (os.cpu_count() or 4)

        # Seeded RNG for reproducible period shuffling each epoch.
        rng = np.random.RandomState(seed)

        # ── Split training data into backprop and eval splits ─────────────────
        # All stations are included (no variance-based pre-filtering here).
        train_sub = train_set.copy()

        # Get unique time periods and shuffle with the seeded RNG.
        unique_times = train_sub['time_period'].drop_duplicates().to_numpy().copy()
        rng.shuffle(unique_times)

        # Partition into backprop (used for gradient updates) and eval (accept/revert).
        split_idx    = int(split_ratio * len(unique_times))
        backprop_set = train_sub[train_sub['time_period'].isin(set(unique_times[:split_idx]))].copy()
        eval_set     = train_sub[train_sub['time_period'].isin(set(unique_times[split_idx:]))].copy()

        # Fit the feature scaler on the backprop split only (avoids val leakage).
        self.scaler.fit(backprop_set[self.feature_cols])

        # N is the number of stations entering the LP per training step.
        N = top_k

        # ── Pre-build per-period tensors for the KKT phase ───────────────────
        # For each time period, we store the scaled feature matrix and the
        # full ground-truth ridership vector. During KKT training, the top-K
        # stations are selected dynamically from these tensors.
        period_data = {}
        for tp, grp in backprop_set.groupby('time_period'):
            # Sort by station ID for consistent row ordering; reset index for positional access.
            grp  = grp.sort_values('station_complex_id').reset_index(drop=True)

            # Scale features and move to device.
            X_tp = torch.tensor(
                self.scaler.transform(grp[self.feature_cols]), dtype=torch.float32
            ).to(self.device)

            # Ground-truth ridership clipped to 0 (negative ridership is meaningless).
            V_actual_full = torch.tensor(
                grp['ridership'].clip(lower=0).values, dtype=torch.float32
            ).to(self.device)

            period_data[tp] = (X_tp, V_actual_full)

        # period_data is reused by both the KKT and Huber phases for mini-batch
        # iteration — no separate full-batch tensor needed.

        # ── Optimiser, scheduler, and criterion ──────────────────────────────
        all_periods = list(period_data.keys())
        optimizer   = optim.AdamW(self.model.parameters(),
                                  lr=self.params.get('lr', 5e-4), weight_decay=0.01)

        # Halve lr when Huber validation loss plateaus for 5 epochs.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.HuberLoss()

        # ── Read remaining hyperparameters ────────────────────────────────────
        num_epochs    = int(self.params.get('epochs', 300))
        n_store       = int(self.params.get('n_store', 50))           # epochs per regime
        noise_std     = float(self.params.get('noise_std', 0.01))     # Huber phase noise
        batch_t       = int(self.params.get('batch_time_periods', 8)) # KKT batch size
        max_per_epoch = int(self.params.get('max_periods_per_epoch', len(all_periods)))
        hw_start      = float(self.params.get('huber_weight', 5.0))
        hw_end        = float(self.params.get('huber_weight_end', 1.0))
        anneal_epochs = int(self.params.get('huber_anneal_epochs', 300))
        eval_fn       = self._get_eval_fn()   # lazy-imported game evaluator

        print(
            f"  [MTA-Hybrid-V3] top_k={N} | {len(all_periods)} periods | "
            f"n_store={n_store} | lp_workers={lp_workers} | device: {self.device}"
        )
        print(
            f"  [MTA-Hybrid-V3] huber_weight: {hw_start:.2f} → {hw_end:.2f} "
            f"over {anneal_epochs} epochs (KKT regimes only)"
        )
        print(
            f"  [MTA-Hybrid-V3] Regime schedule: Huber(warm-up) → KKT → Huber → KKT ..."
        )

        # ── Checkpoint and regime tracking ────────────────────────────────────
        # best_state always holds the best model weights seen so far.
        # best_eval_score tracks the corresponding strategic utility on eval_set.
        best_state      = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_eval_score = -np.inf
        regime_num      = 0        # increments by 1 each n_store-epoch block
        self.epoch_checkpoints = {}

        # ── Outer loop: iterate over regimes in blocks of n_store epochs ─────
        for start_epoch in range(1, num_epochs + 1, n_store):
            # Compute the last epoch of this regime (may be less than n_store at the end).
            end_epoch  = min(start_epoch + n_store - 1, num_epochs)
            regime_num += 1

            # Regime 1 is always Huber (warm-up / pure prediction, no LP).
            # KKT starts at regime 2.  Pattern: Huber → KKT → Huber → KKT ...
            # Note: the "hw a → b" range in the banner below is computed
            # without the warm-up offset and is informational only; the weight
            # actually applied is huber_weight, computed per epoch further down.
            use_kkt = (regime_num % 2 == 0)
            phase   = "KKT  " if use_kkt else "Huber"
            print(f"\n  [Regime {regime_num}] {phase} | epochs {start_epoch}–{end_epoch}"
                  + (f" | hw {hw_start + (hw_end - hw_start) * min((start_epoch-1)/max(anneal_epochs-1,1),1.0):.3f}"
                     f" → {hw_start + (hw_end - hw_start) * min((end_epoch-1)/max(anneal_epochs-1,1),1.0):.3f}"
                     if use_kkt else " | prediction warm-up"))

            # Revert to the best known weights at the start of each regime.
            # This ensures a bad previous regime can't contaminate the new one.
            self.model.load_state_dict(best_state)

            # ── Inner loop: epochs within this regime ─────────────────────────
            for epoch in range(start_epoch, end_epoch + 1):
                # Training mode enables Dropout.
                self.model.train()

                # Shuffle periods each epoch for stochastic mini-batching.
                rng.shuffle(all_periods)
                epoch_periods = all_periods[:max_per_epoch]  # apply per-epoch budget cap

                epoch_loss = 0.0
                n_steps    = 0

                # Anneal huber weight linearly from hw_start → hw_end over anneal_epochs.
                # Offset by n_store so hw=hw_start at the first KKT epoch (epoch n_store+1),
                # not burned away during the Huber warm-up regime. With the paper
                # settings (n_store 50, 5.0 → 3.0 over 300): 5.0 at epoch 51,
                # 3.0 at epoch 350 and every later epoch (t_frac capped at 1).
                t_frac       = min(max(epoch - n_store - 1, 0) / max(anneal_epochs - 1, 1), 1.0)
                huber_weight = hw_start + (hw_end - hw_start) * t_frac

                if use_kkt:
                    # ── KKT regime: strategic LP loss + huber regularisation ──
                    for start in range(0, len(epoch_periods), batch_t):
                        # Slice out the next batch of time period keys.
                        batch_periods = epoch_periods[start: start + batch_t]
                        optimizer.zero_grad()

                        # Accumulate strategic utility and huber loss over the batch.
                        total_util  = torch.tensor(0.0, device=self.device)
                        total_huber = torch.tensor(0.0, device=self.device)

                        for tp in batch_periods:
                            # Retrieve pre-built feature and ridership tensors for this period.
                            X_tp, V_actual_full = period_data[tp]

                            # Forward pass: predict ridership for all stations in this period.
                            # Output shape is (num_stations,) due to squeeze(-1) in forward().
                            raw_preds = self.model(X_tp)

                            # ── Dynamic top-K selection ───────────────────────
                            # Detach before topk so station selection has no gradient.
                            # topk(N).indices gives positions of the N largest predictions.
                            topk_indices  = raw_preds.detach().topk(N).indices

                            # Slice to top-K: V_hat keeps gradients, V_actual_topk is ground truth.
                            V_hat         = raw_preds[topk_indices]
                            V_actual_topk = V_actual_full[topk_indices]

                            # Compute differentiable utility for these N stations.
                            total_util  = total_util  + self._utility_for_period(V_hat, V_actual_topk, lp_workers)
                            # Huber over all stations (same as KKT-V3 — keeps predictions grounded).
                            total_huber = total_huber + nn.HuberLoss()(raw_preds, V_actual_full)

                        B = max(len(batch_periods), 1)
                        # L = -(1/B) Σ u_b  +  λ(t) · (1/B) Σ H(ŷ, y)
                        # λ anneals from hw_start → hw_end, balancing prediction quality vs strategy.
                        loss = -total_util / B + huber_weight * total_huber / B

                        # Skip this batch if no LP returned a gradient (all failed).
                        if not loss.requires_grad:
                            n_steps += 1
                            continue

                        # Backpropagate strategic gradient into MLP weights.
                        loss.backward()
                        # g ← g · min(1, 1/‖g‖₂)  so ‖g‖₂ ≤ 1  (LP gradients can spike)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_steps    += 1
                else:
                    # ── Huber regime: mini-batch supervised prediction loss ────
                    # Iterate over the same time-period batches as the KKT phase
                    # so the number of gradient steps per epoch matches it (4 steps
                    # of 8 windows with the paper settings; PTO/ZOC instead take one
                    # full-batch step per epoch over all training rows).
                    for start in range(0, len(epoch_periods), batch_t):
                        batch_periods = epoch_periods[start: start + batch_t]
                        optimizer.zero_grad()

                        # Stack all station rows from this batch of time periods.
                        X_batch = torch.cat([period_data[tp][0] for tp in batch_periods], dim=0)
                        Y_batch = torch.cat([period_data[tp][1] for tp in batch_periods], dim=0)

                        # Perturb inputs:  X̃ = X_batch + ε,   ε ~ N(0, σ²·I),   σ = noise_std
                        noisy_X = X_batch + torch.randn_like(X_batch) * noise_std

                        # L_Huber = H(f_θ(X̃), Y_batch)
                        # H(ŷ,y) = ½(ŷ-y)²    if |ŷ-y| ≤ 1
                        #         = |ŷ-y| - ½  if |ŷ-y| > 1
                        loss = criterion(self.model(noisy_X), Y_batch)

                        loss.backward()
                        # g ← g · min(1, 1/‖g‖₂)  so ‖g‖₂ ≤ 1
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_steps    += 1

                # Periodic progress logging within each regime.
                if epoch == start_epoch or epoch % 10 == 0 or epoch == end_epoch:
                    val_loss = self.get_loss(val_set)
                    scheduler.step(val_loss)
                    viol_pct = (100.0 * self.n_lp_failures / max(self.n_lp_calls, 1))
                    lp_str   = (f" | LP fail: {self.n_lp_failures}/{self.n_lp_calls} ({viol_pct:.1f}%)"
                                if use_kkt else "")
                    print(
                        f"    > [{phase}] Epoch {epoch:3d}/{num_epochs} | "
                        f"loss: {epoch_loss/max(n_steps,1):.5f} | Val Huber: {val_loss:.5f} | "
                        f"hw: {huber_weight:.3f}{lp_str}"
                    )

            # ── End-of-regime accept-or-revert ────────────────────────────────
            # Capture the current weights as a candidate checkpoint.
            candidate_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            if len(eval_set) > 0:
                # Accept-or-revert rule:
                #   score = U(θ_candidate, eval_set)   [strategic utility on eval split]
                #
                #   if score > best_eval_score:
                #       best_state      ← θ_candidate   (ACCEPT — improvement found)
                #       best_eval_score ← score
                #   else:
                #       best_state unchanged              (REVERT — next regime reloads best at loop top)
                #
                # This prevents any single bad regime from permanently degrading the model.
                score = eval_fn(self.generate(eval_set), eval_set)
                print(
                    f"    [Regime {regime_num} | {phase}] "
                    f"eval util: {score:.5f} | best: {best_eval_score:.5f} | "
                    + ("ACCEPTED" if score > best_eval_score else "REVERTED")
                )
                if score > best_eval_score:
                    best_eval_score = score
                    best_state      = candidate_state
            else:
                # No eval set available — always accept (fallback for small datasets).
                best_state = candidate_state

            # Periodically save a frozen copy of the best state for post-hoc analysis.
            if end_epoch % 50 == 0 or end_epoch == num_epochs:
                self.epoch_checkpoints[end_epoch] = {k: v.clone() for k, v in best_state.items()}

        # ── Restore the globally best checkpoint ─────────────────────────────
        self.model.load_state_dict(best_state)
        print(f"  [MTA-Hybrid-V3] Done. Best eval util: {best_eval_score:.5f}")

    # -------------------------------------------------------------------------

    def get_loss(self, df) -> float:
        """
        Compute Huber loss on a DataFrame using current model weights.

        Used inside train() for LR scheduling and regime-progress logging.

        Parameters
        ----------
        df : pd.DataFrame — must have self.feature_cols and 'ridership' columns.

        Returns
        -------
        float — mean Huber loss. Returns 0.0 for an empty DataFrame.
        """
        # Eval mode disables Dropout for a deterministic output.
        self.model.eval()
        if df.empty:
            return 0.0

        # Scale features and move to device.
        X = torch.tensor(self.scaler.transform(df[self.feature_cols]), dtype=torch.float32).to(self.device)

        # Ground-truth ridership as a 1D tensor (squeeze(-1) in forward gives (B,) output).
        Y = torch.tensor(df['ridership'].values, dtype=torch.float32).to(self.device)

        # No gradient tracking needed for a pure evaluation call.
        with torch.no_grad():
            return nn.HuberLoss()(self.model(X), Y).item()

    # -------------------------------------------------------------------------

    def generate(self, test_df) -> "pd.DataFrame":
        """
        Generate ridership predictions for all rows in test_df.

        Called at test time by train.py and reconstruct_per_window.py, and
        internally at the end of each training regime for the accept-or-revert
        evaluation.

        Parameters
        ----------
        test_df : pd.DataFrame
            Must have self.feature_cols, 'time_period', and 'station_complex_id'.

        Returns
        -------
        pd.DataFrame with columns ['time_period', 'station_complex_id', 'ridership'].
            Ridership values are clipped to >= 0 (np.maximum) as a safety guard,
            though the final ReLU in the MLP should already enforce this.
        """
        # Eval mode: disable Dropout for deterministic inference.
        self.model.eval()

        # Scale test features with the scaler fitted on the training backprop split.
        X_test = torch.tensor(
            self.scaler.transform(test_df[self.feature_cols]), dtype=torch.float32
        ).to(self.device)

        # Run inference without building a computation graph.
        with torch.no_grad():
            preds = test_df[['time_period', 'station_complex_id']].copy()
            # Clip to 0 as a belt-and-suspenders guard against tiny negative outputs.
            preds['ridership'] = np.maximum(
                self.model(X_test).cpu().numpy(), 0.0
            )
        return preds
