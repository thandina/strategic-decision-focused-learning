"""
KKT: differentiable Stackelberg LP (cvxpylayers) for the conservation game.

Serves Table 1, row KKT (conservation game), and Figure 2 through the files
written by train.py. Original file: models/regression_e2e_kkt_gsg2.py (class
name kept: REGRESSION_E2E_KKT_GSG2). Requires cvxpy and cvxpylayers; the
module imports without them, but constructing the class raises ImportError.

What it does: the MLP's predictions V_hat for one day are turned into payoff
matrices, the defender's SSE mix p*(V_hat) is computed by an LP that is
differentiated through its KKT conditions (cvxpylayers, SCS), and the
training loss is minus the defender's utility of p* against the TRUE counts.
Per window:
    t*  = attacker target of the SSE of the predicted game, found with the
          scipy LP solver on the detached V_hat (no gradient);
    p*  = argmax_p D_hat[:, t*] . p  s.t.  (A_hat[:, t'] - A_hat[:, t*]) . p <= 0,
          sum p = 1, p >= 0, re-solved through cvxpylayers so that p* carries
          gradients w.r.t. V_hat;
    u   = p* . D_true[:, t_true], t_true = argmax_t p* . A_true[:, t].
The gradient flows only through the LP for the selected t*; the choice of t*
itself is piecewise constant in V_hat, which is the volatility discussed in
the paper. There is no Huber term in this loss.

Checkpoint selection: at epoch 1, every 5th epoch and the last epoch the
total defender utility on the held-out D_eval days is computed; the weights
with the best D_eval utility are restored at the end of training.

Paper hyperparameters (epochs overridden to 500, random_seed to the run
seed by train.py): hidden_dim 256, lr 5e-4, AdamW weight_decay 0.01,
gradient-norm clipping 1.0, batch_time_periods 8, max_periods_per_epoch 128,
split_ratio 0.85 with RandomState(random_seed), solver_eps 1e-2 (SCS),
ReduceLROnPlateau(min, patience 5, factor 0.5) on the validation Huber loss.
"""

import sys
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Payoff tensor construction  (fully differentiable in V_hat)
# ---------------------------------------------------------------------------

def _build_payoff_tensors(V_hat: torch.Tensor):
    """
    Torch version of game.gen_gsg2_utility_matrices, differentiable in V_hat.
    Returns (D, A), both (N, N):
        D[d, a] = +V_hat[a] if d == a else -V_hat[a]
        A[d, a] = -10       if d == a else  V_hat[a]
    """
    N   = V_hat.shape[0]
    dev = V_hat.device
    eye = torch.eye(N, device=dev)                   # (N,N) float

    # D[d,a] = V_hat[a] * sgn  where sgn[d,a] = +1 if d==a else -1
    sgn = 2.0 * eye - 1.0
    D   = V_hat.unsqueeze(0) * sgn                  # broadcast: (1,N) * (N,N)

    # A[d,a] = V_hat[a] * (1-eye) - 10 * eye
    A   = V_hat.unsqueeze(0) * (1.0 - eye) - 10.0 * eye

    return D, A                                      # both (N,N)


def _build_lp_params(V_hat: torch.Tensor, target_t: int):
    """
    Parameters of the Stackelberg LP for a fixed attacker target.

    Returns:
        d_col     : (N,)   defender payoff column D[:, target_t] (the objective)
        br_matrix : (N-1, N) rows A[:, t'] - A[:, target_t] for t' != target_t
                    (the attacker's incentive-compatibility constraints)
    Both carry gradients w.r.t. V_hat.
    """
    D, A = _build_payoff_tensors(V_hat)
    N = V_hat.shape[0]

    d_col = D[:, target_t]                           # (N,)

    t_prime_list = [tp for tp in range(N) if tp != target_t]
    br_rows = torch.stack([A[:, tp] - A[:, target_t] for tp in t_prime_list])
    # br_rows shape: (N-1, N)

    return d_col, br_rows


# ---------------------------------------------------------------------------
# Build ONE CvxpyLayer for a given N  (reused across all forward passes)
# ---------------------------------------------------------------------------

def _make_stackelberg_layer(N: int) -> "CvxpyLayer":
    """
    Differentiable LP layer (built once per N, reused for every window):
        max  d_col . p
        s.t. br_matrix @ p <= 0   (attacker best-response constraints)
             sum(p) = 1, p >= 0
    Parameters: d_col (N,), br_matrix (N-1, N). Variable: p (N,).
    """
    p         = cp.Variable(N, nonneg=True)
    d_col_p   = cp.Parameter(N, name="d_col")
    br_mat_p  = cp.Parameter((N - 1, N), name="br_matrix")

    objective   = cp.Maximize(d_col_p @ p)
    constraints = [br_mat_p @ p <= 0, cp.sum(p) == 1]
    prob        = cp.Problem(objective, constraints)

    return CvxpyLayer(prob, parameters=[d_col_p, br_mat_p], variables=[p])


# ---------------------------------------------------------------------------
# Architecture  (same as the other models)
# ---------------------------------------------------------------------------

class CellCountMLP(nn.Module):
    """Same architecture as models/pto.py: 9 -> hidden -> hidden/2 -> 1 (LayerNorm, SiLU, Dropout 0.2)."""
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
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.network(x).squeeze(-1)


def get_runtime_hyperparams():
    """Default hyperparameters; train.py overrides 'epochs' (500) and 'random_seed' (run seed)."""
    return {
        'hidden_dim': 256,
        'epochs': 300,
        'lr': 5e-4,
        'batch_time_periods': 8,
        'max_periods_per_epoch': 128,
        'split_ratio': 0.85,
        'random_seed': 42,
        'solver_eps': 1e-2,
    }


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class REGRESSION_E2E_KKT_GSG2:
    """
    KKT model. Construct with (params, feature_cols, window_size); call
    train(train_set, val_set) then generate(test_set). window_size is stored
    but unused. Raises ImportError at construction if cvxpy/cvxpylayers are
    missing.
    """

    def __init__(self, params, feature_cols, window_size):
        if not _CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy and cvxpylayers are required for REGRESSION_E2E_KKT_GSG2. "
                "Install via: pip install cvxpy cvxpylayers"
            )
        self.params = params
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.scaler = StandardScaler()
        from utils import best_device; self.device = best_device()

        input_dim  = len(self.feature_cols)
        hidden_dim = int(params.get('hidden_dim', 256))
        self.model = CellCountMLP(input_dim, hidden_dim).to(self.device)
        self._lp_layer = None   # built lazily once N is known

    def _ensure_lp_layer(self, N: int):
        """Build the cvxpylayers LP for N targets on first use."""
        if self._lp_layer is None:
            self._lp_layer = _make_stackelberg_layer(N)

    def _utility_for_period(
        self,
        V_hat: torch.Tensor,       # (N,) with grad — predicted cell counts
        V_actual: torch.Tensor,    # (N,) no grad  — true cell counts
    ) -> torch.Tensor:
        """
        Defender utility of one window as a differentiable scalar.

        Step 1: attacker target t* of the SSE of the PREDICTED game, found with
                the scipy LP solver on the detached V_hat (no gradient).
        Step 2: re-solve the LP for that fixed t* through cvxpylayers (SCS,
                eps = solver_eps) so that p* is differentiable in V_hat.
        Step 3: attacker best response t_true to p* under the TRUE payoffs by
                plain argmax (lowest index on exact ties; the evaluation code
                in game.py breaks ties in the defender's favour instead), and
                utility p* . D_true[:, t_true].
        Returns a scalar tensor with gradient w.r.t. V_hat, or a constant 0
        tensor (no gradient) if the cvxpylayers solve fails.
        """
        N = V_hat.shape[0]
        self._ensure_lp_layer(N)

        # Step 1: find the best-response target t* with the scipy SSE solver on
        # the detached prediction (one call enumerates all N targets).
        with torch.no_grad():
            from solvers.nfg_sse_lp import solve_general_sum_normal_form_mta
            V_np  = V_hat.detach().cpu().numpy()
            D_np  = np.array([[V_np[a] if d == a else -V_np[a] for a in range(N)]
                               for d in range(N)], dtype=float)
            A_np  = np.array([[-10.0 if d == a else V_np[a] for a in range(N)]
                               for d in range(N)], dtype=float)
            best_t, _, _ = solve_general_sum_normal_form_mta(D_np, A_np)

        # Step 2: re-solve for best_t WITH gradient through cvxpylayers
        d_col, br_matrix = _build_lp_params(V_hat, best_t)

        solver_eps = self.params.get('solver_eps', 1e-3)
        try:
            (p_star,) = self._lp_layer(
                d_col, br_matrix,
                solver_args={"solve_method": "SCS", "eps": solver_eps}
            )
        except Exception as e:
            # LP infeasible / solver failure for this prediction: contribute a
            # constant zero with no gradient.
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        p_star = p_star.float()  # cvxpylayers may output float64

        # Step 3: compute ACTUAL utility against true state V_actual
        D_actual, A_actual = _build_payoff_tensors(V_actual)
        with torch.no_grad():
            atk_utils = (p_star.detach() @ A_actual)
            t_actual  = int(atk_utils.argmax().item())
        # Defender utility: gradient flows through p_star -> LP -> V_hat -> model parameters
        utility = p_star @ D_actual[:, t_actual]
        return utility

    def train(self, train_set, val_set):
        """
        Mini-batch training on minus the defender utility, with best-D_eval
        checkpoint restoration.

        1. Shuffle the unique training dates with RandomState(random_seed)
           and take the first split_ratio (85%) as backprop dates, the rest
           as D_eval. Fit the scaler on the backprop rows only; precompute
           per-day tensors (features, true counts, cell index).
        2. Each epoch: reshuffle the backprop days with the same RandomState,
           take the first max_periods_per_epoch (128) of them, and step the
           optimizer once per batch of batch_time_periods (8) days on
           loss = -(mean utility over the batch), with gradient-norm clipping
           at 1.0. A batch whose every window failed in the LP has no
           gradient and is skipped. The per-day prediction V_hat is the
           model's output per cell (one row per cell, aggregated by cell
           index), clipped at 0.
        3. At epoch 1, every 5th epoch and the last epoch: validation Huber
           loss (scheduler step) and total defender utility on D_eval; the
           weights with the best D_eval utility are remembered and restored
           at the end.
        Side effects: updates self.model / self.scaler; self.epoch_checkpoints
        {epoch: best-D_eval weights so far} every 50 epochs and at the last
        epoch (per-epoch MAE file of train.py).
        """
        split_ratio = float(self.params.get('split_ratio', 0.85))
        seed        = int(self.params.get('random_seed', 42))
        rng = np.random.RandomState(seed)

        # Inner split: backprop days vs held-out strategic days D_eval.
        unique_times = train_set['time_period'].drop_duplicates().to_numpy().copy()
        rng.shuffle(unique_times)
        split_idx    = int(split_ratio * len(unique_times))
        backprop_times = set(unique_times[:split_idx])
        eval_times     = set(unique_times[split_idx:])

        backprop_set = train_set[train_set['time_period'].isin(backprop_times)].copy()
        eval_set     = train_set[train_set['time_period'].isin(eval_times)].copy()

        self.scaler.fit(backprop_set[self.feature_cols])

        # Pre-build lookup: time_period -> (X tensor, Y_actual tensor)
        cell_ids = sorted(backprop_set['cell_id'].unique())
        N = len(cell_ids)
        cell_to_idx = {c: i for i, c in enumerate(cell_ids)}

        # Precompute everything per period so the training loop does zero DataFrame work.
        # period_data[tp] = (X_tensor, V_actual_tensor, cell_index_tensor)
        # cell_index_tensor maps each row of X_tensor to its cell index in [0, N).
        period_data = {}
        for tp, grp in backprop_set.groupby('time_period'):
            grp = grp.sort_values('cell_id')
            X_tp = torch.tensor(
                self.scaler.transform(grp[self.feature_cols]), dtype=torch.float32
            ).to(self.device)
            V_actual = torch.tensor(
                grp.set_index('cell_id').reindex(cell_ids)['elephant_count']
                   .fillna(0.0).values,
                dtype=torch.float32, device=self.device
            )
            # Integer index into [0, N) for each row, used for scatter-add aggregation
            cell_idx = torch.tensor(
                [cell_to_idx[c] for c in grp['cell_id']], dtype=torch.long, device=self.device
            )
            period_data[tp] = (X_tp, V_actual, cell_idx)

        all_periods = list(period_data.keys())
        optimizer   = optim.AdamW(
            self.model.parameters(),
            lr=self.params.get('lr', 5e-4),
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        num_epochs       = int(self.params.get('epochs', 200))
        batch_t          = int(self.params.get('batch_time_periods', 8))
        max_per_epoch    = int(self.params.get('max_periods_per_epoch', len(all_periods)))

        print(
            f"  [GSG2-KKT] N={N} | {len(all_periods)} training periods | "
            f"batch={batch_t} periods/step | device: {self.device}"
        )

        best_val_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_val_util  = -np.inf
        self.epoch_checkpoints = {}
        ckpt_interval = int(self.params.get('checkpoint_interval', 50))

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            # The same RandomState that made the split picks the epoch's days.
            rng.shuffle(all_periods)
            # Subsample periods per epoch: like SGD, each period is still seen
            # about (max_per_epoch/total * num_epochs) times over full training.
            epoch_periods = all_periods[:max_per_epoch]
            epoch_loss = 0.0
            n_steps    = 0

            for start in range(0, len(epoch_periods), batch_t):
                batch_periods = epoch_periods[start: start + batch_t]
                optimizer.zero_grad()
                total_util = torch.tensor(0.0, device=self.device)

                for tp in batch_periods:
                    X_tp, V_actual, cell_idx = period_data[tp]
                    raw_preds = self.model(X_tp)

                    # Per-cell prediction vector: scatter the row predictions into
                    # cell slots and average (one row per cell, so this is a reorder),
                    # then clip at 0 as generate() does.
                    V_hat  = torch.zeros(N, device=self.device).scatter_add(0, cell_idx, raw_preds)
                    counts = torch.zeros(N, device=self.device).scatter_add(
                        0, cell_idx, torch.ones_like(raw_preds)
                    )
                    V_hat = torch.clamp(V_hat / counts.clamp(min=1), min=0.0)

                    u = self._utility_for_period(V_hat, V_actual)
                    total_util = total_util + u

                loss = -total_util / max(len(batch_periods), 1)
                if not loss.requires_grad:
                    # Every window of the batch hit the LP failure path: nothing to backprop.
                    n_steps += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_steps    += 1

            avg_loss = epoch_loss / max(n_steps, 1)

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                val_loss = self.get_loss(val_set)
                scheduler.step(val_loss)
                # Strategic utility on D_eval decides the checkpoint that is restored at the end.
                if len(eval_set) > 0:
                    try:
                        game_mod = importlib.import_module("game")
                        eval_preds = self.generate(eval_set)
                        val_util   = game_mod.evaluate_game_performance(eval_preds, eval_set)
                        if val_util > best_val_util:
                            best_val_util  = val_util
                            best_val_state = {k: v.detach().clone()
                                              for k, v in self.model.state_dict().items()}
                    except Exception:
                        val_util = float('nan')
                else:
                    val_util = float('nan')
                print(
                    f"    > Epoch {epoch:3d}/{num_epochs} | "
                    f"Strategic loss: {avg_loss:.5f} | Val Huber: {val_loss:.5f} | "
                    f"Eval util: {val_util:.4f}"
                )

            # The epoch checkpoint records the best-D_eval weights seen so far.
            if epoch % ckpt_interval == 0 or epoch == num_epochs:
                self.epoch_checkpoints[epoch] = {k: v.clone() for k, v in best_val_state.items()}

        # Restore best checkpoint by strategic utility
        if best_val_util > -np.inf:
            self.model.load_state_dict(best_val_state)
            print(f"  [GSG2-KKT] Restored best checkpoint (eval util={best_val_util:.5f})")

    def get_loss(self, df):
        """Huber loss of the current model on df (eval mode); 0.0 if df is empty."""
        self.model.eval()
        if df.empty:
            return 0.0
        X = torch.tensor(
            self.scaler.transform(df[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        Y = torch.tensor(df['elephant_count'].values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return nn.HuberLoss()(self.model(X), Y).item()

    def generate(self, test_df):
        """
        Predict counts for every row of test_df (eval mode). Returns a DataFrame
        [time_period, cell_id, elephant_count] with test_df's index; predictions
        are clipped at 0.
        """
        self.model.eval()
        X_test = torch.tensor(
            self.scaler.transform(test_df[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            preds = self.model(X_test).cpu().numpy()
        results = test_df[['time_period', 'cell_id']].copy()
        results['elephant_count'] = np.maximum(preds, 0.0)
        return results
