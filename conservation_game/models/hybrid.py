"""
Hybrid (Algorithm 1 of the paper): alternating Huber and KKT blocks with
accept-or-revert, for the conservation game.

Serves Table 1, row Hybrid (conservation game), and Figure 2 through the
files written by train.py. Original file: models/regression_hybrid_gsg2.py
(class name kept: REGRESSION_HYBRID_GSG2). Requires cvxpy and cvxpylayers;
the module imports without them, but constructing the class raises
ImportError.

What it does: training is cut into blocks ("regimes") of n_store = 50
epochs, as in ZOC. Odd blocks (1, 3, 5, ...) train the full-batch Huber loss
with input noise; even blocks (2, 4, 6, ...) train the KKT loss (minus the
defender utility obtained by differentiating through the Stackelberg LP, as
in models/kkt.py). So block 1 is always a Huber warm-up. Every block starts
from the best weights so far; at the end of the block the total defender
utility on the held-out D_eval days is compared with the best so far and the
block is accepted or reverted. The final model is the best accepted block.
Only the weights are reverted; the AdamW moments and the LR schedule carry
on across blocks. With 500 epochs there are 10 blocks: 5 Huber, 5 KKT.

Paper hyperparameters (epochs overridden to 500, random_seed to the run seed
by train.py): hidden_dim 256, lr 5e-4, noise_std 0.01 (Huber blocks),
n_store 50, batch_time_periods 8, max_periods_per_epoch 128 (KKT blocks),
split_ratio 0.85 with RandomState(random_seed), solver_eps 1e-2 (SCS), AdamW
weight_decay 0.01, gradient-norm clipping 1.0 in both block types,
ReduceLROnPlateau(min, patience 5, factor 0.5) on the validation Huber loss.
No Huber term inside the KKT loss and no annealing.
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
# Payoff + LP helpers  (identical to models/kkt.py)
# ---------------------------------------------------------------------------

def _build_payoff_tensors(V_hat: torch.Tensor):
    """Differentiable (D, A) payoff matrices from a count vector; see models/kkt.py."""
    N   = V_hat.shape[0]
    eye = torch.eye(N, device=V_hat.device)
    D   = V_hat.unsqueeze(0) * (2.0 * eye - 1.0)
    A   = V_hat.unsqueeze(0) * (1.0 - eye) - 10.0 * eye
    return D, A


def _build_lp_params(V_hat: torch.Tensor, target_t: int):
    """LP objective column D[:, t*] and the (N-1, N) best-response constraint rows; see models/kkt.py."""
    D, A  = _build_payoff_tensors(V_hat)
    N     = V_hat.shape[0]
    d_col = D[:, target_t]
    br_rows = torch.stack([A[:, tp] - A[:, target_t] for tp in range(N) if tp != target_t])
    return d_col, br_rows


def _make_stackelberg_layer(N: int):
    """cvxpylayers LP: max d_col . p s.t. br_matrix @ p <= 0, sum p = 1, p >= 0; see models/kkt.py."""
    p        = cp.Variable(N, nonneg=True)
    d_col_p  = cp.Parameter(N, name="d_col")
    br_mat_p = cp.Parameter((N - 1, N), name="br_matrix")
    prob     = cp.Problem(cp.Maximize(d_col_p @ p), [br_mat_p @ p <= 0, cp.sum(p) == 1])
    return CvxpyLayer(prob, parameters=[d_col_p, br_mat_p], variables=[p])


# ---------------------------------------------------------------------------
# Architecture
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
        'noise_std': 0.01,
        'n_store': 50,
        'batch_time_periods': 8,
        'max_periods_per_epoch': 128,
        'split_ratio': 0.85,
        'random_seed': 42,
        'solver_eps': 1e-2,
    }


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class REGRESSION_HYBRID_GSG2:
    """
    Hybrid model (Algorithm 1). Construct with (params, feature_cols,
    window_size); call train(train_set, val_set) then generate(test_set).
    window_size is stored but unused. Raises ImportError at construction if
    cvxpy/cvxpylayers are missing.
    """

    def __init__(self, params, feature_cols, window_size):
        if not _CVXPY_AVAILABLE:
            raise ImportError("cvxpy and cvxpylayers are required for REGRESSION_HYBRID_GSG2.")
        self.params       = params
        self.feature_cols = feature_cols
        self.window_size  = window_size
        self.scaler       = StandardScaler()
        from utils import best_device; self.device = best_device()

        hidden_dim  = int(params.get('hidden_dim', 256))
        self.model  = CellCountMLP(len(feature_cols), hidden_dim).to(self.device)
        self._lp_layer = None

    def _ensure_lp_layer(self, N: int):
        """Build the cvxpylayers LP for N targets on first use."""
        if self._lp_layer is None:
            self._lp_layer = _make_stackelberg_layer(N)

    def _get_eval_fn(self):
        """Return game.evaluate_game_performance (the strategic score used for accept-or-revert)."""
        try:
            return importlib.import_module("game").evaluate_game_performance
        except ImportError:
            print("CRITICAL ERROR: game module not found"); sys.exit(1)

    def _utility_for_period(self, V_hat: torch.Tensor, V_actual: torch.Tensor, N: int):
        """
        Defender utility of one window as a differentiable scalar (same three
        steps as models/kkt.py): attacker target t* from the scipy SSE solver
        on the detached prediction; p* from the cvxpylayers re-solve at that
        t* (SCS, eps = solver_eps); utility p* . D_true[:, t_true] with t_true
        the plain-argmax attacker response to p* under the true payoffs.
        Returns a constant 0 tensor (no gradient) if the cvxpylayers solve fails.
        """
        self._ensure_lp_layer(N)
        # Step 1: attacker target of the SSE of the predicted game, no gradient.
        with torch.no_grad():
            from solvers.nfg_sse_lp import solve_general_sum_normal_form_mta
            V_np = V_hat.detach().cpu().numpy()
            D_np = np.array([[V_np[a] if d == a else -V_np[a] for a in range(N)]
                              for d in range(N)], dtype=float)
            A_np = np.array([[-10.0 if d == a else V_np[a] for a in range(N)]
                              for d in range(N)], dtype=float)
            best_t, _, _ = solve_general_sum_normal_form_mta(D_np, A_np)

        # Step 2: differentiable re-solve of the LP at the fixed target best_t.
        d_col, br_matrix = _build_lp_params(V_hat, best_t)
        try:
            (p_star,) = self._lp_layer(
                d_col, br_matrix,
                solver_args={"solve_method": "SCS", "eps": self.params.get('solver_eps', 1e-2)}
            )
        except Exception:
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        # Step 3: utility against the true counts; gradient flows through p_star only.
        p_star = p_star.float()
        D_actual, A_actual = _build_payoff_tensors(V_actual)
        with torch.no_grad():
            t_actual = int((p_star.detach() @ A_actual).argmax().item())
        return p_star @ D_actual[:, t_actual]

    def train(self, train_set, val_set):
        """
        Alternating Huber / KKT blocks with accept-or-revert on D_eval.

        1. Shuffle the unique training dates with RandomState(random_seed);
           first split_ratio (85%) = backprop dates, rest = D_eval. Fit the
           scaler on the backprop rows only. Precompute per-day tensors (KKT
           blocks) and the full feature/target tensors (Huber blocks).
        2. Blocks of n_store epochs, numbered from 1: odd = Huber (full
           batch, input noise), even = KKT (batches of batch_time_periods
           days out of max_periods_per_epoch resampled days, loss = minus the
           mean utility; a batch with no gradient is skipped). Both block
           types clip the gradient norm at 1.0. The day order is reshuffled
           with the same RandomState at EVERY epoch, Huber epochs included,
           so the KKT mini-batches depend on the whole epoch history. Each
           block starts from the best weights so far. Validation Huber loss
           and scheduler step at the first epoch of the block, every 10th
           epoch and the last epoch of the block.
        3. After each block: total defender utility on D_eval; accepted if
           strictly better than the best so far (the first block always beats
           -inf), else reverted. If D_eval were empty every block would be
           accepted. At the end the best accepted weights are loaded.
        Side effects: updates self.model / self.scaler; self.epoch_checkpoints
        {end_epoch: best accepted weights so far} at every block end that is
        a multiple of 50 and at the last epoch (per-epoch MAE file of train.py).
        """
        split_ratio = float(self.params.get('split_ratio', 0.85))
        seed        = int(self.params.get('random_seed', 42))
        rng = np.random.RandomState(seed)

        # Inner split: backprop days vs held-out strategic days D_eval.
        unique_times = train_set['time_period'].drop_duplicates().to_numpy().copy()
        rng.shuffle(unique_times)
        split_idx      = int(split_ratio * len(unique_times))
        backprop_set   = train_set[train_set['time_period'].isin(set(unique_times[:split_idx]))].copy()
        eval_set       = train_set[train_set['time_period'].isin(set(unique_times[split_idx:]))].copy()

        self.scaler.fit(backprop_set[self.feature_cols])
        cell_ids    = sorted(backprop_set['cell_id'].unique())
        N           = len(cell_ids)
        cell_to_idx = {c: i for i, c in enumerate(cell_ids)}

        # Per-period tensors for the KKT blocks
        period_data = {}
        for tp, grp in backprop_set.groupby('time_period'):
            grp = grp.sort_values('cell_id')
            X_tp = torch.tensor(
                self.scaler.transform(grp[self.feature_cols]), dtype=torch.float32
            ).to(self.device)
            V_actual = torch.tensor(
                grp.set_index('cell_id').reindex(cell_ids)['elephant_count'].fillna(0.0).values,
                dtype=torch.float32, device=self.device
            )
            cell_idx = torch.tensor(
                [cell_to_idx[c] for c in grp['cell_id']], dtype=torch.long, device=self.device
            )
            period_data[tp] = (X_tp, V_actual, cell_idx)

        # Full tensors for the Huber blocks
        X_train = torch.tensor(
            self.scaler.transform(backprop_set[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        Y_train = torch.tensor(backprop_set['elephant_count'].values, dtype=torch.float32).to(self.device)

        all_periods   = list(period_data.keys())
        optimizer     = optim.AdamW(self.model.parameters(),
                                    lr=self.params.get('lr', 5e-4), weight_decay=0.01)
        scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion     = nn.HuberLoss()

        num_epochs    = int(self.params.get('epochs', 300))
        n_store       = int(self.params.get('n_store', 50))
        noise_std     = float(self.params.get('noise_std', 0.01))
        batch_t       = int(self.params.get('batch_time_periods', 8))
        max_per_epoch = int(self.params.get('max_periods_per_epoch', len(all_periods)))
        eval_fn       = self._get_eval_fn()

        print(
            f"  [GSG2-Hybrid] N={N} | {len(all_periods)} periods | "
            f"n_store={n_store} | device: {self.device}"
        )

        # best_state starts as the random initialisation with score -inf, so the
        # first (Huber) block is always accepted.
        best_state      = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_eval_score = -np.inf
        regime_num      = 0
        self.epoch_checkpoints = {}

        for start_epoch in range(1, num_epochs + 1, n_store):
            end_epoch  = min(start_epoch + n_store - 1, num_epochs)
            regime_num += 1
            use_kkt    = (regime_num % 2 == 0)   # odd=Huber, even=KKT
            phase      = "KKT  " if use_kkt else "Huber"

            # Revert to best at the start of every regime (optimizer state is kept)
            self.model.load_state_dict(best_state)

            for epoch in range(start_epoch, end_epoch + 1):
                self.model.train()
                # Reshuffled every epoch, in Huber blocks too (advances the RandomState).
                rng.shuffle(all_periods)
                epoch_periods = all_periods[:max_per_epoch]
                epoch_loss = 0.0
                n_steps    = 0

                if use_kkt:
                    # KKT block: minus the mean differentiable utility per batch of days.
                    for start in range(0, len(epoch_periods), batch_t):
                        batch_periods = epoch_periods[start: start + batch_t]
                        optimizer.zero_grad()
                        total_util = torch.tensor(0.0, device=self.device)

                        for tp in batch_periods:
                            X_tp, V_actual, cell_idx = period_data[tp]
                            raw_preds = self.model(X_tp)
                            # Per-cell prediction vector (one row per cell), clipped at 0.
                            V_hat = torch.zeros(N, device=self.device).scatter_add(0, cell_idx, raw_preds)
                            counts = torch.zeros(N, device=self.device).scatter_add(
                                0, cell_idx, torch.ones_like(raw_preds)
                            )
                            V_hat = torch.clamp(V_hat / counts.clamp(min=1), min=0.0)
                            total_util = total_util + self._utility_for_period(V_hat, V_actual, N)

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
                else:
                    # Huber block: one full-batch step on noisy inputs.
                    optimizer.zero_grad()
                    noisy_X = X_train + torch.randn_like(X_train) * noise_std
                    loss    = criterion(self.model(noisy_X), Y_train)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_steps     = 1

                if epoch == start_epoch or epoch % 10 == 0 or epoch == end_epoch:
                    val_loss = self.get_loss(val_set)
                    scheduler.step(val_loss)
                    print(
                        f"    > [{phase}] Epoch {epoch:3d}/{num_epochs} | "
                        f"loss: {epoch_loss/max(n_steps,1):.5f} | Val Huber: {val_loss:.5f}"
                    )

            # Strategic evaluation on D_eval: accept the block or revert.
            candidate_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if len(eval_set) > 0:
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
                best_state = candidate_state

            # The epoch checkpoint records the best accepted weights at this point.
            if end_epoch % 50 == 0 or end_epoch == num_epochs:
                self.epoch_checkpoints[end_epoch] = {k: v.clone() for k, v in best_state.items()}

        self.model.load_state_dict(best_state)
        print(f"  [GSG2-Hybrid] Done. Best eval util: {best_eval_score:.5f}")

    def get_loss(self, df):
        """Huber loss of the current model on df (eval mode); 0.0 if df is empty."""
        self.model.eval()
        if df.empty:
            return 0.0
        X = torch.tensor(self.scaler.transform(df[self.feature_cols]), dtype=torch.float32).to(self.device)
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
