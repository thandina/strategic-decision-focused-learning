"""
ZOC (zeroth-order checkpointing) for the conservation game.

Serves Table 1, row ZOC (conservation game), and Figure 2 through the files
written by train.py. Original file: models/regression_zo3_model_gsg2.py
(class name kept: REGRESSION_ZO3_MODEL_GSG2; the output files written by
train.py keep the prefix "zo3").

What it does: the same MLP and Huber training as PTO, but training is cut
into blocks ("regimes") of n_store = 50 epochs and the game decides which
block to keep. The training dates are split once into a backprop part (85%)
and a held-out strategic part D_eval (15%). Every block starts from the best
weights so far; at the end of the block the model's total defender utility on
D_eval (game.evaluate_game_performance) is compared with the best so far and
the block is accepted (becomes the new best) or reverted. The final model is
the best accepted block. Only the model weights are reverted; the AdamW
moments and the LR schedule carry on across blocks.

Quirk, kept on purpose: the inner backprop / D_eval split uses a fixed
np.random.RandomState(42), not the run seed (KKT and Hybrid use the seed).
This is what produced the paper's numbers. Nothing else in this model is
seeded: initial weights, dropout and input noise come from torch's default
generator, which the runner leaves unseeded.

Paper hyperparameters (epochs overridden to 500 by train.py): hidden_dim 256,
lr 1e-3, noise_std 0.01, n_store 50 (10 blocks), split_ratio 0.85, AdamW
weight_decay 0.01, ReduceLROnPlateau(min, patience 5, factor 0.5) on the
validation Huber loss, full batch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import importlib


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
    """
    Default hyperparameters; train.py overrides 'epochs' (500) and
    'random_seed' (the run seed). 'random_seed' is not read by this model:
    the inner split uses RandomState(42) regardless (see module docstring).
    """
    return {
        'hidden_dim': 256,
        'epochs': 100,
        'lr': 0.001,
        'noise_std': 0.01,
        'n_store': 50,
        'random_seed': 42,
        'split_ratio': 0.85,
    }


class REGRESSION_ZO3_MODEL_GSG2:
    """
    ZOC model. Construct with (params, feature_cols, window_size); call
    train(train_set, val_set) then generate(test_set). window_size is stored
    but unused.
    """
    def __init__(self, params, feature_cols, window_size):
        self.params = params
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.scaler = StandardScaler()
        from utils import best_device; self.device = best_device()

        input_dim = len(self.feature_cols)
        hidden_dim = int(params.get('hidden_dim', 256))
        self.model = CellCountMLP(input_dim, hidden_dim).to(self.device)

    def _get_eval_fn(self):
        """Return game.evaluate_game_performance (the strategic score used for accept-or-revert)."""
        try:
            game_mod = importlib.import_module("game")
            return game_mod.evaluate_game_performance
        except ImportError:
            print("CRITICAL ERROR: game module not found")
            sys.exit()

    def train(self, train_set, val_set):
        """
        Block-wise Huber training with accept-or-revert on D_eval.

        1. Shuffle the unique training dates with RandomState(42) (fixed on
           purpose) and take the first split_ratio (85%) as backprop dates,
           the rest as D_eval. Fit the scaler on the backprop rows only.
        2. For each block of n_store epochs: load the best weights so far,
           train full-batch Huber with input noise (validation Huber loss and
           scheduler step at the first epoch of the block, every 5th epoch
           and the last epoch of the block), then score the block's weights
           by total defender utility on D_eval. Accept if strictly better
           than the best so far (the first block always beats -inf), else
           revert. If D_eval were empty every block would be accepted.
        3. Load the best accepted weights.
        Side effects: updates self.model / self.scaler; self.epoch_checkpoints
        {end_epoch: best state so far} at every block end that is a multiple
        of 50 and at the last epoch (used by train.py for the per-epoch MAEs).
        val_set is used only for the scheduler and logging.
        """
        split_ratio = float(self.params.get('split_ratio', 0.85))
        # Fixed RandomState(42) here is intentional: it is what produced the
        # paper's numbers (KKT and Hybrid use RandomState(random_seed) instead).
        rng = np.random.RandomState(42)
        unique_times = train_set['time_period'].drop_duplicates().to_numpy().copy()
        rng.shuffle(unique_times)
        split_idx = int(split_ratio * len(unique_times))

        backprop_times = set(unique_times[:split_idx])
        eval_times = set(unique_times[split_idx:])

        backprop_set = train_set[train_set['time_period'].isin(backprop_times)].copy()
        eval_set = train_set[train_set['time_period'].isin(eval_times)].copy()

        self.scaler.fit(backprop_set[self.feature_cols])

        X_train = torch.tensor(
            self.scaler.transform(backprop_set[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        Y_train = torch.tensor(
            backprop_set['elephant_count'].values, dtype=torch.float32
        ).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params.get('lr', 0.001),
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )
        criterion = nn.HuberLoss()

        num_epochs = int(self.params.get('epochs', 20))
        n_store = int(self.params.get('n_store', 50))
        noise_std = self.params.get('noise_std', 0.01)
        eval_fn = self._get_eval_fn()

        print(
            f"  [GSG2-ZO3] Backprop samples: {len(X_train)} | "
            f"Eval samples: {len(eval_set)} | device: {self.device}"
        )

        # best_state starts as the random initialisation with score -inf, so the
        # first block is always accepted.
        best_state = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        best_eval_score = -np.inf
        self.epoch_checkpoints = {}

        for start_epoch in range(1, num_epochs + 1, n_store):
            end_epoch = min(start_epoch + n_store - 1, num_epochs)

            # Every block starts from the best weights so far (revert-to-best);
            # the optimizer state is not reset.
            self.model.load_state_dict(best_state)

            for epoch in range(start_epoch, end_epoch + 1):
                self.model.train()
                optimizer.zero_grad()

                noisy_X = X_train + torch.randn_like(X_train) * noise_std
                output = self.model(noisy_X)
                loss = criterion(output, Y_train)
                loss.backward()
                optimizer.step()

                if epoch == start_epoch or epoch % 5 == 0 or epoch == end_epoch:
                    val_loss = self.get_loss(val_set)
                    scheduler.step(val_loss)
                    print(
                        f"    > Epoch {epoch:3d}/{num_epochs} | "
                        f"Train Loss: {loss.item():.5f} | Val Loss: {val_loss:.5f}"
                    )

            candidate_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }

            # Accept-or-revert: the block's weights are scored by the game on
            # D_eval (total defender utility, higher is better) and kept only
            # if they beat the best block so far.
            if len(eval_set) > 0:
                eval_preds = self.generate(eval_set)
                current_eval_score = eval_fn(eval_preds, eval_set)
                print(
                    f"    [Regime {start_epoch}-{end_epoch}] "
                    f"eval_fn score: {current_eval_score:.5f} | "
                    f"best: {best_eval_score:.5f}"
                )

                if current_eval_score > best_eval_score:
                    print(
                        f"    [Regime {start_epoch}-{end_epoch}] "
                        f"New best eval_fn score: {current_eval_score:.5f}"
                    )
                    best_eval_score = current_eval_score
                    best_state = candidate_state
            else:
                best_state = candidate_state

            # The epoch checkpoint records the best accepted weights at this
            # point (not necessarily the block just trained).
            if end_epoch % 50 == 0 or end_epoch == num_epochs:
                self.epoch_checkpoints[end_epoch] = {k: v.clone() for k, v in best_state.items()}

        self.model.load_state_dict(best_state)

    def get_loss(self, df):
        """Huber loss of the current model on df (eval mode, no noise); 0.0 if df is empty."""
        self.model.eval()
        if df.empty:
            return 0.0

        X = torch.tensor(
            self.scaler.transform(df[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        Y = torch.tensor(
            df['elephant_count'].values, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            preds = self.model(X)
            return nn.HuberLoss()(preds, Y).item()

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
