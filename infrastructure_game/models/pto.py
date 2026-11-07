"""
models/pto.py -- the PTO (predict-then-optimise) baseline of Table 1 of
"Strategic Decision Focused Learning" (model key `pto` in train.py; originally
regression_model_mta_v2.py).

A 3-layer MLP trained on the Huber loss alone: one full-batch step per epoch
over all training rows, Gaussian input noise (std noise_std = 0.01) redrawn
every epoch, AdamW (lr 0.01, weight_decay 0.01), ReduceLROnPlateau
(patience 5, factor 0.5) stepped on the validation Huber loss at epoch 1,
every 5th epoch and the last epoch. The game is never seen during training
and the final-epoch weights are used (no checkpoint selection). At test time
train.py solves the game on the predictions (game.evaluate_game_performance).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class MTARidershipMLP(nn.Module):
    """
    Per-row ridership regressor shared by all four models:
        Linear(132 -> 256) -> LayerNorm -> SiLU -> Dropout(0.2)
        -> Linear(256 -> 128) -> SiLU -> Linear(128 -> 1) -> ReLU
    Input (B, 132) scaled features (any extra window dimension is flattened);
    output (B, 1), non-negative thanks to the final ReLU.
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
    and adds random_seed (unused by this model). The values that ran are
    hidden_dim 256, lr 0.01, noise_std 0.01.
    """
    return {
        'hidden_dim': 256,
        'epochs': 100,
        'lr': 0.01,
        'noise_std': 0.01
    }


class REGRESSION_MODEL_MTA_V2:
    """
    PTO model: MLP + StandardScaler, Huber training, no game during training.

    params       : dict from get_runtime_hyperparams() (+ train.py overrides)
    feature_cols : the 132 feature columns from data_pipeline.load_and_prep_mta
    window_size  : stored only (the loader has already built the lag columns)
    Attributes after train(): self.model (weights of the last epoch),
    self.scaler (fitted on all training rows), self.epoch_checkpoints
    (state-dict copies every checkpoint_interval = 50 epochs; informational).
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

    def train(self, train_set, val_set):
        """
        Full-batch Huber training for params['epochs'] epochs.

        train_set : training rows (scaler is fitted on them); val_set : rows
        used only for the validation Huber loss that drives the LR scheduler
        and the progress print. Updates self.model in place; the weights of
        the final epoch are kept.
        """
        self.scaler.fit(train_set[self.feature_cols])

        X_train = torch.tensor(self.scaler.transform(train_set[self.feature_cols]), dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(train_set['ridership'].values, dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer  = optim.AdamW(self.model.parameters(), lr=self.params.get('lr', 0.001), weight_decay=0.01)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion  = nn.HuberLoss()
        num_epochs = int(self.params.get('epochs', 20))
        noise_std  = self.params.get('noise_std', 0.01)
        ckpt_interval = int(self.params.get('checkpoint_interval', 50))

        print(f"  [MTA-Regression-V2] Training on {len(X_train)} samples on {self.device}...")
        self.epoch_checkpoints = {}

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            # Input perturbation X + eps, eps ~ N(0, noise_std^2), redrawn every epoch.
            noisy_X = X_train + torch.randn_like(X_train) * noise_std
            output  = self.model(noisy_X)
            loss    = criterion(output, Y_train)
            loss.backward()
            optimizer.step()

            # Validation Huber loss every 5 epochs (and at epochs 1 and last)
            # drives ReduceLROnPlateau.
            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                val_loss = self.get_loss(val_set)
                scheduler.step(val_loss)
                print(f"    > Epoch {epoch:3d}/{num_epochs} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss:.5f}")

            if epoch % ckpt_interval == 0 or epoch == num_epochs:
                self.epoch_checkpoints[epoch] = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }

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
