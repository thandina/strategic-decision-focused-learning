"""
PTO (predict-then-optimise) baseline for the conservation game.

Serves Table 1, row PTO (conservation game), and the PTO curves of Figure 2
through the files written by train.py. Original file:
models/regression_model_gsg2.py (class name kept: REGRESSION_MODEL_GSG2).

What it does: an MLP is trained on per-cell elephant counts with a Huber loss
and never sees the game. The weights after the last epoch are used; there is
no checkpoint selection. train.py sets params['epochs'] = 500 and
params['random_seed'] = seed, but this model reads nothing seed-related:
initial weights, dropout masks and the input noise come from torch's default
generator, which the runner leaves unseeded (as in the paper's run).

Paper hyperparameters (get_runtime_hyperparams, epochs overridden to 500):
hidden_dim 256, lr 1e-3, AdamW weight_decay 0.01, ReduceLROnPlateau(min,
patience 5, factor 0.5) on the validation Huber loss, Gaussian input noise
noise_std 0.01, full batch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler


class CellCountMLP(nn.Module):
    """
    Per-row regressor: 9 features -> hidden -> hidden/2 -> 1, with LayerNorm,
    SiLU and Dropout(0.2) after the first layer and SiLU after the second.
    The output is unbounded; generate() clips predictions at 0.
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
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.network(x).squeeze(-1)


def get_runtime_hyperparams():
    """Default hyperparameters; train.py overrides 'epochs' (500 for the paper)."""
    return {
        'hidden_dim': 256,
        'epochs': 100,
        'lr': 0.001,
        'noise_std': 0.01,
    }


class REGRESSION_MODEL_GSG2:
    """
    PTO model. Construct with (params, feature_cols, window_size); call
    train(train_set, val_set) then generate(test_set). window_size is stored
    but unused (the lags are already columns of the data).
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

    def train(self, train_set, val_set):
        """
        Fit the scaler on train_set and run params['epochs'] full-batch epochs
        of Huber loss on noisy inputs (X + N(0, noise_std^2)).

        The validation Huber loss is computed at epoch 1, every 5th epoch and
        the last epoch; the LR scheduler is stepped only at those points.
        Side effects: updates self.model / self.scaler and fills
        self.epoch_checkpoints {epoch: state_dict} every 50 epochs and at the
        last epoch (train.py turns these into the per-epoch MAE file). The
        weights after the last epoch are kept.
        """
        self.scaler.fit(train_set[self.feature_cols])

        X_train = torch.tensor(
            self.scaler.transform(train_set[self.feature_cols]), dtype=torch.float32
        ).to(self.device)
        Y_train = torch.tensor(
            train_set['elephant_count'].values, dtype=torch.float32
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
        noise_std = self.params.get('noise_std', 0.01)
        ckpt_interval = int(self.params.get('checkpoint_interval', 50))

        print(f"  [GSG2-Regression] Training on {len(X_train)} samples on {self.device}...")

        self.epoch_checkpoints = {}

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            noisy_X = X_train + torch.randn_like(X_train) * noise_std
            output = self.model(noisy_X)
            loss = criterion(output, Y_train)

            loss.backward()
            optimizer.step()

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                val_loss = self.get_loss(val_set)
                scheduler.step(val_loss)
                print(
                    f"    > Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {loss.item():.5f} | Val Loss: {val_loss:.5f}"
                )

            if epoch % ckpt_interval == 0 or epoch == num_epochs:
                self.epoch_checkpoints[epoch] = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }

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
        Predict counts for every row of test_df (eval mode, no dropout).
        Returns a DataFrame [time_period, cell_id, elephant_count] with the same
        index as test_df; predictions are clipped at 0.
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
