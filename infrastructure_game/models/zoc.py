"""
models/zoc.py -- the ZOC model of Table 1 of "Strategic Decision Focused
Learning" (model key `zoc` in train.py; originally regression_zo3_model_mta_v2.py,
"ZO3" in older file names).

Same MLP and full-batch Huber training as PTO (lr 0.01, noise_std 0.01), plus
the accept-or-revert mechanism that uses the game as a zeroth-order signal:

  * the training windows are split once, 85% "backprop" rows (used for the
    gradient steps and to fit the scaler) / 15% inner evaluation split D_eval,
    with a FIXED np.random.RandomState(42) (this file does not read
    params['random_seed']; seeds differ only through torch's unseeded RNG);
  * training is cut into blocks of n_store = 50 epochs. Every block starts
    from the best weights so far; at its end the game utility of the current
    weights on D_eval (game.evaluate_game_performance) is compared with the
    best so far: strictly better -> ACCEPT (new best), otherwise -> REVERT;
  * the model returned is the best accepted block.

No gradient flows from the game; it only selects among Huber-trained candidates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import importlib


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
            nn.ReLU()
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.network(x)


def get_runtime_hyperparams():
    """
    Default hyperparameters. train.py overrides epochs (500 in the paper run)
    and random_seed (--seed). The values that ran: hidden_dim 256, lr 0.01,
    noise_std 0.01, n_store 50, split_ratio 0.85. random_seed and
    eval_samples are not read by this file (the inner split uses
    RandomState(42) and the whole of D_eval is evaluated).
    """
    return {
        'hidden_dim': 256,
        'epochs': 100,
        'lr': 0.01,
        'noise_std': 0.01,
        'n_store': 50,
        'random_seed': 42,
        'eval_samples': 100,
        'split_ratio': 0.85
    }


class REGRESSION_ZO3_MODEL_MTA_V2:
    """
    ZOC model: Huber-trained MLP with block-wise accept-or-revert on the game
    utility of an inner evaluation split.

    params       : dict from get_runtime_hyperparams() (+ train.py overrides)
    feature_cols : the 132 feature columns from data_pipeline.load_and_prep_mta
    window_size  : stored only
    Attributes after train(): self.model (best accepted block), self.scaler
    (fitted on the backprop rows), self.epoch_checkpoints (copies of the best
    state after every block; informational).
    """
    def __init__(self, params, feature_cols, window_size):
        self.params = params
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.scaler = StandardScaler()
        from utils import best_device; self.device = best_device()

        input_dim  = len(self.feature_cols)
        hidden_dim = int(params.get('hidden_dim', 256))
        self.model = MTARidershipMLP(input_dim, hidden_dim).to(self.device)

    def _get_eval_fn(self):
        """Lazily import game.evaluate_game_performance (package root must be on sys.path); exits if missing."""
        try:
            game_mod = importlib.import_module("game")
            return game_mod.evaluate_game_performance
        except ImportError:
            print("CRITICAL ERROR: game module not found")
            sys.exit()

    def train(self, train_set, val_set):
        """
        Block-wise Huber training with accept-or-revert (see module docstring).

        train_set : training rows; split 85/15 by window into backprop rows and
            D_eval with RandomState(42). val_set : rows used only for the
            validation Huber loss that drives the LR scheduler.
        Updates self.model in place with the best accepted block.
        """
        # Inner split by window, fixed seed 42 (independent of params['random_seed']).
        split_ratio = float(self.params.get('split_ratio', 0.85))
        rng = np.random.RandomState(42)
        unique_times = train_set['time_period'].drop_duplicates().to_numpy().copy()
        rng.shuffle(unique_times)
        split_idx = int(split_ratio * len(unique_times))

        backprop_times = set(unique_times[:split_idx])
        eval_times     = set(unique_times[split_idx:])

        backprop_set = train_set[train_set['time_period'].isin(backprop_times)].copy()
        eval_set     = train_set[train_set['time_period'].isin(eval_times)].copy()

        # Scaler fitted on the backprop rows only.
        self.scaler.fit(backprop_set[self.feature_cols])

        X_train = torch.tensor(self.scaler.transform(backprop_set[self.feature_cols]), dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(backprop_set['ridership'].values, dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer  = optim.AdamW(self.model.parameters(), lr=self.params.get('lr', 0.001), weight_decay=0.01)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion  = nn.HuberLoss()
        num_epochs = int(self.params.get('epochs', 20))
        n_store    = int(self.params.get('n_store', 10))
        noise_std  = self.params.get('noise_std', 0.01)
        eval_fn    = self._get_eval_fn()

        print(f"  [MTA-ZOC-V2] Backprop samples: {len(X_train)} | Eval samples: {len(eval_set)} | device: {self.device}")

        best_state      = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_eval_score = -np.inf
        self.epoch_checkpoints = {}

        # One block ("regime") per n_store epochs. The optimiser and scheduler
        # state persist across blocks; only the weights are reverted.
        for start_epoch in range(1, num_epochs + 1, n_store):
            end_epoch = min(start_epoch + n_store - 1, num_epochs)
            self.model.load_state_dict(best_state)       # every block starts from the best weights so far

            for epoch in range(start_epoch, end_epoch + 1):
                self.model.train()
                optimizer.zero_grad()
                noisy_X = X_train + torch.randn_like(X_train) * noise_std
                output  = self.model(noisy_X)
                loss    = criterion(output, Y_train)
                loss.backward()
                optimizer.step()

                if epoch == start_epoch or epoch % 5 == 0 or epoch == end_epoch:
                    val_loss = self.get_loss(val_set)
                    scheduler.step(val_loss)
                    print(f"    > Epoch {epoch:3d}/{num_epochs} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss:.5f}")

            candidate_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            # Accept-or-revert: the game utility of the block's final weights on
            # D_eval must strictly beat the best so far to become the new best.
            # Otherwise best_state is unchanged and the next block reloads it.
            if len(eval_set) > 0:
                eval_preds         = self.generate(eval_set)
                current_eval_score = eval_fn(eval_preds, eval_set)
                print(f"    [Regime {start_epoch}-{end_epoch}] eval_fn score: {current_eval_score:.5f} | best: {best_eval_score:.5f}")
                if current_eval_score > best_eval_score:
                    print(f"    [Regime {start_epoch}-{end_epoch}] New best eval_fn score: {current_eval_score:.5f}")
                    best_eval_score = current_eval_score
                    best_state      = candidate_state
            else:
                best_state = candidate_state

            if end_epoch % 50 == 0 or end_epoch == num_epochs:
                self.epoch_checkpoints[end_epoch] = {k: v.clone() for k, v in best_state.items()}

        # The returned model is the best accepted block.
        self.model.load_state_dict(best_state)

    def get_loss(self, df):
        """Huber loss of the current weights on df (eval mode, no grad); 0.0 if df is empty."""
        self.model.eval()
        if df.empty: return 0.0
        X = torch.tensor(self.scaler.transform(df[self.feature_cols]), dtype=torch.float32).to(self.device)
        Y = torch.tensor(df['ridership'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            return nn.HuberLoss()(self.model(X), Y).item()

    def generate(self, test_df):
        """
        Predict ridership for every row of test_df (eval mode: dropout off).
        Returns a DataFrame with time_period, station_complex_id, ridership
        (the prediction), keeping test_df's index.
        """
        self.model.eval()
        X_test = torch.tensor(self.scaler.transform(test_df[self.feature_cols]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_test).cpu().numpy().flatten()
        results = test_df[['time_period', 'station_complex_id']].copy()
        results['ridership'] = preds
        return results
