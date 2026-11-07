"""
models/kkt.py -- the KKT model of Table 1 of "Strategic Decision Focused
Learning" ("KKT" in the paper; model key `kkt_v3` in train.py; originally
regression_e2e_kkt_mta_v3.py).

Decision-focused training through the Stackelberg LP. Each epoch samples at
most `max_periods_per_epoch` = 32 training windows (reshuffled every epoch)
and visits them in mini-batches of `batch_time_periods` = 8. For each window:

  1. the MLP predicts ridership for all 123 stations;
  2. the `top_k` = 20 stations with the largest PREDICTED ridership are
     selected (the selection is detached: no gradient flows through it);
  3. Pass 1: the attacker's SSE target t* among those 20 stations is found by
     enumerating the 20 single-target LPs with scipy/HiGHS on detached
     predictions, in a thread pool of `lp_workers` threads;
  4. Pass 2: the LP for t* is re-solved through cvxpylayers (SCS, eps 1e-4) so
     that the defender strategy p* carries gradients w.r.t. the predictions;
  5. the training utility is p* . D_true[:, t_true], where t_true is the
     attacker's best response to p* under the TRUE ridership of the 20
     stations (a detached argmax); an LP failure contributes 0;
  6. loss = -(1/B) sum utility + huber_weight * (1/B) sum HuberLoss(all 123
     stations), huber_weight annealed linearly from 5.0 (epoch 1) to 3.0
     (epoch 300) and held at 3.0 for epochs 300..500 (values set by train.py).
  Gradient norm clipped at 1.0; AdamW lr 5e-4, weight_decay 0.01.

Model selection: at epoch 1, every 5th epoch and the last epoch, the game
utility of the current weights on the inner evaluation split D_eval (15% of
the training windows, split with RandomState(random_seed)) is computed with
game.evaluate_game_performance; the best state seen is restored at the end
of training.

cvxpy and cvxpylayers are required to instantiate the class (the import
failure is deferred to __init__ so that this module imports without them).
"""

import os
import importlib
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler

# The differentiable LP needs cvxpy + cvxpylayers. Only record availability
# here; the model constructor raises if they are missing.
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


# =============================================================================
# Payoff / LP helpers
# =============================================================================

def _build_payoff_tensors(V_hat):
    """
    Defender and attacker payoff matrices (N x N torch tensors) from a
    ridership vector V_hat of length N, differentiable in V_hat:
        D[d, a] = +V[a] if d == a else -V[a]
        A[d, a] = -10   if d == a else  V[a]
    Rows = patrolled station, columns = attacked station (same game as game.py).
    """
    N   = V_hat.shape[0]
    dev = V_hat.device
    eye = torch.eye(N, device=dev)
    sgn = 2.0 * eye - 1.0                               # +1 on the diagonal, -1 off it
    D   = V_hat.unsqueeze(0) * sgn
    A   = V_hat.unsqueeze(0) * (1.0 - eye) - 10.0 * eye
    return D, A


def _build_lp_params(V_hat, target_t):
    """
    LP data for a fixed attacker target t, both differentiable in V_hat:
        d_col   = D[:, t]                          objective, length N
        br_rows = [A[:, t'] - A[:, t]] for t' != t   (N-1) x N best-response rows
    The LP is  max_p d_col . p  s.t.  br_rows @ p <= 0, sum p = 1, p >= 0.
    """
    D, A  = _build_payoff_tensors(V_hat)
    N     = V_hat.shape[0]
    d_col = D[:, target_t]
    br_rows = torch.stack([A[:, tp] - A[:, target_t]
                           for tp in range(N) if tp != target_t])
    return d_col, br_rows


def _make_stackelberg_layer(N):
    """
    cvxpylayers layer for the single-target Stackelberg LP of size N:
    parameters (d_col, br_matrix) -> optimal defender strategy p (length N).
    Compiling the problem is expensive, so the model caches one layer per N.
    """
    p        = cp.Variable(N, nonneg=True)
    d_col_p  = cp.Parameter(N,        name="d_col")
    br_mat_p = cp.Parameter((N-1, N), name="br_matrix")
    prob     = cp.Problem(
        cp.Maximize(d_col_p @ p),
        [br_mat_p @ p <= 0, cp.sum(p) == 1]
    )
    return CvxpyLayer(prob, parameters=[d_col_p, br_mat_p], variables=[p])


# =============================================================================
# MLP
# =============================================================================

class MTARidershipMLP(nn.Module):
    """
    Per-row ridership regressor shared by all four models:
        Linear(132 -> 256) -> LayerNorm -> SiLU -> Dropout(0.2)
        -> Linear(256 -> 128) -> SiLU -> Linear(128 -> 1) -> ReLU
    Input (B, 132) scaled features; output (B, 1), non-negative.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.network(x)


# =============================================================================
# Hyperparameters
# =============================================================================

def get_runtime_hyperparams():
    """
    Defaults. train.py overrides, for the paper run: epochs 500, random_seed
    (--seed), top_k 20 (--top_k), max_periods_per_epoch 32, lp_workers
    (--lp_workers; 6 in the launch script), huber_weight 5.0,
    huber_weight_end 3.0 (--huber_weight_end), huber_anneal_epochs 300.
    Unchanged defaults that ran: hidden_dim 256, lr 5e-4, batch_time_periods 8,
    split_ratio 0.85. `solver_eps` is NOT read by this file: the Pass-2 SCS
    tolerance is the literal 1e-4 in _utility_for_period.
    """
    return {
        'hidden_dim':            256,
        'epochs':                300,
        'lr':                    5e-4,
        'batch_time_periods':    8,
        'max_periods_per_epoch': 128,
        'split_ratio':           0.85,
        'random_seed':           42,
        'solver_eps':            1e-2,
        'top_k':                 9,
        # Huber annealing: starts high to stabilise early predictions,
        # decays to a lower value so the strategic LP signal can dominate.
        'huber_weight':          5.0,
        'huber_weight_end':      1.0,
        'huber_anneal_epochs':   300,
        # 0 = use os.cpu_count(); set explicitly to cap parallelism if needed.
        'lp_workers':            0,
    }


# =============================================================================
# Model
# =============================================================================

class REGRESSION_E2E_KKT_MTA_V3:
    """
    KKT model: MLP trained on -utility (through the differentiable LP) plus an
    annealed Huber term, with best-D_eval-utility checkpoint selection.

    params       : dict from get_runtime_hyperparams() (+ train.py overrides)
    feature_cols : the 132 feature columns from data_pipeline.load_and_prep_mta
    window_size  : stored only
    Attributes after train(): self.model (best D_eval state), self.scaler
    (fitted on the backprop rows), self.epoch_checkpoints (copies of the best
    state every 50 epochs; informational), self.n_lp_calls / self.n_lp_failures
    (Pass-2 solve attempts and SCS exceptions).
    Raises ImportError at construction if cvxpy / cvxpylayers are missing.
    """

    def __init__(self, params, feature_cols, window_size):
        if not _CVXPY_AVAILABLE:
            raise ImportError("cvxpy and cvxpylayers are required.")

        self.params       = params
        self.feature_cols = feature_cols
        self.window_size  = window_size
        self.scaler       = StandardScaler()

        from utils import best_device
        self.device = best_device()

        input_dim  = len(self.feature_cols)
        hidden_dim = int(params.get('hidden_dim', 256))
        self.model = MTARidershipMLP(input_dim, hidden_dim).to(self.device)
        self._lp_layer = None

        # KKT gradient validity tracking — incremented in _utility_for_period.
        # n_lp_calls  : total Pass-2 (cvxpylayers) solve attempts
        # n_lp_failures: attempts where SCS raised an exception (gradient zeroed)
        self.n_lp_calls    = 0
        self.n_lp_failures = 0

    # -------------------------------------------------------------------------

    def _ensure_lp_layer(self, N):
        """Build the cvxpylayers LP layer for N stations once; rebuild only if N changes."""
        if self._lp_layer is None or getattr(self, '_lp_N', None) != N:
            self._lp_layer = _make_stackelberg_layer(N)
            self._lp_N     = N

    # -------------------------------------------------------------------------

    def _utility_for_period(self, V_hat, V_actual, lp_workers):
        """
        Differentiable defender utility of one window (the KKT step).

        Inputs
            V_hat    : (N,) predicted ridership of the N = top_k selected
                       stations; carries gradients.
            V_actual : (N,) true ridership of the same stations.
            lp_workers : threads for the Pass-1 enumeration.
        Returns a scalar tensor p* . D_true[:, t_true] that backpropagates
        into V_hat through the LP solution map, or a constant 0 (no gradient)
        if the cvxpylayers solve raises; n_lp_calls / n_lp_failures are updated.
        """
        N = V_hat.shape[0]
        self._ensure_lp_layer(N)

        # ── Pass 1: parallel LP enumeration to find t* ───────────────────────
        # Payoffs from DETACHED predictions; one scipy/HiGHS LP per candidate
        # target t (max p.D[:,t] s.t. (A[:,t']-A[:,t]).p <= 0, sum p = 1,
        # 0 <= p <= 1); t* is the target with the largest optimum, first one on
        # exact ties (best_t = 0 if every LP fails).
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

        # Threads only change wall-clock time; pool.map returns in target order.
        with ThreadPoolExecutor(max_workers=lp_workers) as pool:
            for t, res in pool.map(_solve_t, range(N)):
                if res.success and -res.fun > best_util:
                    best_util = -res.fun
                    best_t    = t

        # ── Pass 2: differentiable re-solve for t* ────────────────────────────
        # Same LP, now with the gradient-carrying V_hat, solved by SCS inside
        # cvxpylayers (tolerance 1e-4, hard-coded) so that p* is a function of
        # V_hat with gradients through the LP's KKT conditions.
        d_col, br_matrix = _build_lp_params(V_hat, best_t)
        self.n_lp_calls += 1
        try:
            (p_star,) = self._lp_layer(
                d_col, br_matrix,
                solver_args={"solve_method": "SCS", "eps": 1e-4},
            )
        except Exception:
            self.n_lp_failures += 1
            return torch.tensor(0.0, device=self.device)

        # Attacker's best response to p* under the TRUE payoffs (plain argmax,
        # detached) and the defender's true utility, differentiable in p*.
        p_star   = p_star.float()
        D_actual, A_actual = _build_payoff_tensors(V_actual)
        with torch.no_grad():
            t_actual = int((p_star.detach() @ A_actual).argmax().item())
        return p_star @ D_actual[:, t_actual]

    # -------------------------------------------------------------------------

    def train(self, train_set, val_set):
        """
        KKT training loop (see module docstring).

        train_set : training rows, split 85/15 by window with
            RandomState(random_seed) into backprop rows (gradient steps; scaler
            fitted on them) and the inner evaluation split D_eval (checkpoint
            selection). val_set : rows used only for the validation Huber loss
            that drives the LR scheduler.
        Updates self.model in place; restores the best D_eval state at the end.
        """
        split_ratio = float(self.params.get('split_ratio', 0.85))
        seed        = int(self.params.get('random_seed', 42))
        top_k       = int(self.params.get('top_k', 9))

        # LP worker count — 0 means use all available cores.
        lp_workers_cfg = int(self.params.get('lp_workers', 0))
        lp_workers     = lp_workers_cfg if lp_workers_cfg > 0 else (os.cpu_count() or 4)

        # Huber annealing schedule parameters.
        hw_start       = float(self.params.get('huber_weight',         5.0))
        hw_end         = float(self.params.get('huber_weight_end',     1.0))
        anneal_epochs  = int(self.params.get('huber_anneal_epochs',   300))

        # One seeded RNG for the inner split and the per-epoch window sampling.
        rng = np.random.RandomState(seed)

        train_sub    = train_set.copy()
        unique_times = train_sub['time_period'].drop_duplicates().to_numpy().copy()
        rng.shuffle(unique_times)

        split_idx    = int(split_ratio * len(unique_times))
        backprop_set = train_sub[train_sub['time_period'].isin(set(unique_times[:split_idx]))].copy()
        eval_set     = train_sub[train_sub['time_period'].isin(set(unique_times[split_idx:]))].copy()
        eval_set_full = val_set

        self.scaler.fit(backprop_set[self.feature_cols])
        N = top_k

        # Pre-build per-window tensors: scaled features and the true ridership
        # (clipped at 0) of all 123 stations, rows sorted by station id.
        period_data = {}
        for tp, grp in backprop_set.groupby('time_period'):
            grp = grp.sort_values('station_complex_id').reset_index(drop=True)
            X_tp = torch.tensor(
                self.scaler.transform(grp[self.feature_cols]), dtype=torch.float32
            ).to(self.device)
            V_actual_full = torch.tensor(
                grp['ridership'].clip(lower=0).values, dtype=torch.float32
            ).to(self.device)
            period_data[tp] = (X_tp, V_actual_full)

        all_periods = list(period_data.keys())
        optimizer   = optim.AdamW(
            self.model.parameters(),
            lr=self.params.get('lr', 5e-4),
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        num_epochs    = int(self.params.get('epochs', 200))
        batch_t       = int(self.params.get('batch_time_periods', 8))
        max_per_epoch = int(self.params.get('max_periods_per_epoch', len(all_periods)))

        print(
            f"  [MTA-KKT-V3] top_k={N} | {len(all_periods)} periods | "
            f"batch={batch_t} | lp_workers={lp_workers} | device: {self.device}"
        )
        print(
            f"  [MTA-KKT-V3] huber_weight: {hw_start:.2f} → {hw_end:.2f} "
            f"over {anneal_epochs} epochs"
        )

        best_val_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_val_util  = -np.inf
        self.epoch_checkpoints = {}
        ckpt_interval  = int(self.params.get('checkpoint_interval', 50))

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            # Reshuffle all backprop windows and keep the first max_per_epoch.
            rng.shuffle(all_periods)
            epoch_periods = all_periods[:max_per_epoch]

            # Linear anneal: epoch 1 → hw_start, epoch anneal_epochs → hw_end,
            # then held (t_frac capped at 1). With 5.0 → 3.0 over 300 epochs:
            # 5.0 at epoch 1, 3.0 at epoch 300 and for every later epoch.
            t_frac       = min((epoch - 1) / max(anneal_epochs - 1, 1), 1.0)
            huber_weight = hw_start + (hw_end - hw_start) * t_frac

            epoch_loss = 0.0
            n_steps    = 0

            for start in range(0, len(epoch_periods), batch_t):
                batch_periods = epoch_periods[start: start + batch_t]
                optimizer.zero_grad()

                total_util  = torch.tensor(0.0, device=self.device)
                total_huber = torch.tensor(0.0, device=self.device)

                for tp in batch_periods:
                    X_tp, V_actual_full = period_data[tp]
                    raw_preds     = self.model(X_tp).squeeze(-1)
                    # top_k selection on DETACHED predictions: which stations
                    # enter the LP is a decision, not part of the gradient path.
                    topk_indices  = raw_preds.detach().topk(N).indices
                    V_hat         = raw_preds[topk_indices]
                    V_actual_topk = V_actual_full[topk_indices]

                    u            = self._utility_for_period(V_hat, V_actual_topk, lp_workers)
                    total_util   = total_util  + u
                    # Huber over ALL 123 stations of the window.
                    total_huber  = total_huber + nn.HuberLoss()(raw_preds, V_actual_full)

                # loss = -(1/B) sum utility + huber_weight * (1/B) sum Huber
                loss = (
                    -total_util  / max(len(batch_periods), 1)
                    + huber_weight * total_huber / max(len(batch_periods), 1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps    += 1

            avg_loss = epoch_loss / max(n_steps, 1)

            # Evaluation points: epoch 1, every 5th epoch, last epoch. The LR
            # scheduler steps on the validation Huber loss; the game utility on
            # D_eval (process pool of MTA_EVAL_WORKERS) selects the checkpoint.
            # Any exception in that evaluation is swallowed (val_util = nan).
            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                val_loss = self.get_loss(eval_set_full)
                scheduler.step(val_loss)

                if len(eval_set) > 0:
                    try:
                        game_mod  = importlib.import_module("game")
                        eval_pred = self.generate(eval_set)
                        val_util  = game_mod.evaluate_game_performance(eval_pred, eval_set)
                        if val_util > best_val_util:
                            best_val_util  = val_util
                            best_val_state = {k: v.detach().clone()
                                              for k, v in self.model.state_dict().items()}
                    except Exception:
                        val_util = float('nan')
                else:
                    val_util = float('nan')

                viol_pct = (100.0 * self.n_lp_failures / max(self.n_lp_calls, 1))
                print(
                    f"    > Epoch {epoch:3d}/{num_epochs} | "
                    f"Strategic loss: {avg_loss:.5f} | Val Huber: {val_loss:.5f} | "
                    f"Eval util: {val_util:.4f} | hw: {huber_weight:.3f} | "
                    f"LP fail: {self.n_lp_failures}/{self.n_lp_calls} ({viol_pct:.1f}%)"
                )

            if epoch % ckpt_interval == 0 or epoch == num_epochs:
                self.epoch_checkpoints[epoch] = {k: v.clone() for k, v in best_val_state.items()}

        # Checkpoint selection: restore the best D_eval utility seen (if any
        # evaluation succeeded); otherwise the final weights are kept.
        if best_val_util > -np.inf:
            self.model.load_state_dict(best_val_state)
            print(f"  [MTA-KKT-V3] Restored best checkpoint (eval util={best_val_util:.5f})")

    # -------------------------------------------------------------------------

    def get_loss(self, df):
        """Huber loss of the current weights on df (eval mode, no grad); 0.0 if df is empty."""
        self.model.eval()
        if df.empty:
            return 0.0
        X = torch.tensor(
            self.scaler.transform(df[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        Y = torch.tensor(df['ridership'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            return nn.HuberLoss()(self.model(X), Y).item()

    # -------------------------------------------------------------------------

    def generate(self, test_df):
        """
        Predict ridership for every row of test_df (eval mode: dropout off).
        Returns a DataFrame with time_period, station_complex_id, ridership
        (the prediction), keeping test_df's index.
        """
        self.model.eval()
        X_test = torch.tensor(
            self.scaler.transform(test_df[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            preds = self.model(X_test).cpu().numpy().flatten()
        results = test_df[['time_period', 'station_complex_id']].copy()
        results['ridership'] = preds
        return results
