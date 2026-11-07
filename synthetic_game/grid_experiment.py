"""
grid_experiment.py -- synthetic Conservation Security Game (Appendix B.1)
==========================================================================

Trains and compares the paper's four learned methods on a synthetic security
game played on an N x N grid of cells (N = 10 in the paper, so 100 targets)
and writes per-scenario utilities, model weights and diagnostic figures.
Appendix B.1 of "Strategic Decision Focused Learning" describes the setting;
its regime panels (regime_panel_4row, regime_panel_hybrid_kkt,
regime_panel_failures) are drawn from this script's saved weights by
plot_regime_panels.py.

The game. Each cell holds an animal density v_i in [0, 10]. The defender
commits to a mixed patrol strategy p over the cells; the poacher best-responds
by attacking one cell. If cell a is attacked while cell d is patrolled the
defender gets +v_a and the attacker -P (catch penalty P = 10) when d == a, and
the defender gets -v_a and the attacker +v_a otherwise. A strategy is scored
against the TRUE density with defender-favoured tie-breaking (strong
Stackelberg equilibrium); see solve_sse_full_game() and eval_vs_truth().

The data. Densities come from three families drawn at random per scenario:
smooth Gaussian blobs (probability 1/2), sparse spikes (1/4) and concentric
rings (1/4). The model never sees the density itself, only n_obs sparse,
noisy "survey sweeps" of it (make_observations). generate_dataset() draws
n_train scenarios for training (the last 20% are held out for validation)
and n_test for testing; the data RNG is always seeded with SEED = 42, so the
test scenarios are identical for every training seed.

The predictor. A four-layer 3x3 CNN (GridPredictorCNN) maps the n_obs sweep
channels to a density map that is rescaled to a maximum of 10.

The four methods (one function each; all use AdamW with weight decay 1e-4,
a cosine learning-rate schedule, the Huber loss and gradient clipping at 1):
  PTO     train_pto()     Huber loss only; keep the final weights.
  ZOC     train_zoc()     Huber loss, but every n_store epochs the strategic
                          utility on a held-out 15% of the training scenarios
                          is measured and the new weights are kept only if it
                          improved; otherwise they revert to the best so far
                          (zeroth-order checkpointing).
  KKT     train_kkt()     Back-propagate the defender's utility through the
                          Stackelberg LP (cvxpylayers); checkpoint by
                          strategic utility.
  Hybrid  train_hybrid()  Alternate Huber regimes (odd) and KKT regimes (even)
                          of n_store epochs, each starting from the best
                          weights so far and accepted only if the strategic
                          utility improves (Algorithm 1 of the paper).

Outputs, one folder per run: <out>/grid<N>_seed<s>_<timestamp>/
  summary.txt, results.csv            utilities, gaps to optimal, MAE, winners
  {pto,zoc,kkt,hybrid}_model.pth      weights, saved as soon as each finishes
  {pto,zoc,kkt,hybrid}_results.csv    per-model per-scenario utility and MAE
  obs_test.npy, den_test.npy          the test scenarios (inputs, truth)
  heatmap_panel.png, utility_bar.png, utility_violin.png,
  utility_scatter.png, gap_to_optimal.png, compare_panels/   diagnostics

Usage (see README.md):
    python grid_experiment.py --grid 10 --seeds 1 --epochs 200   # paper's run
    python grid_experiment.py --grid 2 --seeds 1 --epochs 5      # smoke test

Requires torch, numpy, scipy and matplotlib; cvxpy + cvxpylayers for KKT and
Hybrid (without them both fall back to ZOC with a warning).
"""

# PYTORCH_ENABLE_MPS_FALLBACK must be set before torch is imported.
# nn.Softplus uses integer beta/threshold constants that MPS lowers to si64
# tensors, causing an f32×si64 type-mismatch crash.  The fallback flag makes
# those ops transparently run on CPU while everything else uses MPS.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linprog

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False
    print("  [WARNING] cvxpy/cvxpylayers not installed. "
          "KKT and Hybrid will fall back to ZOC.\n"
          "  Install with: pip install cvxpy cvxpylayers")


# ─── EXPERIMENT CONFIGURATION ─────────────────────────────────────────────────
# One entry per supported grid size (--grid). Keys:
#   grid               side length N; the game has N*N target cells
#   n_train            scenarios generated for training (the last 20% become
#                      the validation split, see run_one)
#   n_test             test scenarios
#   n_obs              sparse noisy survey sweeps per scenario = CNN input channels
#   k_samples          cells observed by each sweep (sampled without replacement)
#   n_epochs           training epochs (overridable with --epochs)
#   n_store            window / regime length in epochs for ZOC, KKT and Hybrid
#   lr                 AdamW learning rate (cosine-annealed over n_epochs)
#   poisson_scale      observed counts ~ Poisson(density * poisson_scale)
#   catch_penalty      the attacker's payoff when caught (P in the paper)
#   kkt_batch_size     scenarios per KKT gradient step
#   kkt_max_per_epoch  scenarios whose LP is differentiated in each KKT epoch
# The paper's Appendix B.1 run is the grid=10 entry (N = 100 cells, 1,000
# training scenarios, 200 epochs); grid=2 is a seconds-long smoke test.

GRID_CONFIGS = {
    2: dict(
        grid=2,
        n_train=20,
        n_test=5,
        n_obs=3,
        k_samples=3,
        n_epochs=5,
        n_store=2,
        lr=3e-3,
        poisson_scale=2,
        catch_penalty=10.0,
        label="2×2 grid (smoke test)",
        kkt_batch_size=2,
        kkt_max_per_epoch=4,
    ),
    10: dict(
        grid=10,
        n_train=1000,
        n_test=30,
        n_obs=5,
        k_samples=20,
        n_epochs=200,
        n_store=20,
        lr=3e-3,
        poisson_scale=2,
        catch_penalty=10.0,
        label="10×10 grid",
        kkt_batch_size=4,
        kkt_max_per_epoch=32,
    ),
    15: dict(
        grid=15,
        n_train=1000,
        n_test=30,
        n_obs=5,
        k_samples=20,
        n_epochs=200,
        n_store=20,
        lr=3e-3,
        poisson_scale=2,
        catch_penalty=10.0,
        label="15×15 grid",
        kkt_batch_size=4,
        kkt_max_per_epoch=32,
    ),
}

SEED    = 42                                # data RNG (always) and default training seed
OUT_DIR = Path(__file__).parent / "output"  # default --out; one sub-folder per run
N_PLOT  = 5                                 # scenarios shown in heatmap_panel.png
CMAP_D  = "YlOrRd"                          # colormap for densities
CMAP_S  = "Blues"                           # colormap for patrol strategies


# ─── PROGRESS BAR ─────────────────────────────────────────────────────────────

def _bar(current: int, total: int, prefix: str = "", width: int = 36) -> None:
    """Overwrite-in-place text progress bar; prints a newline once complete."""
    pct    = current / max(total, 1)
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix} [{bar}] {current}/{total}  ({100*pct:5.1f}%)",
          end="", flush=True)
    if current >= total:
        print()


# ─── DATA GENERATION ──────────────────────────────────────────────────────────

def make_gaussian_grid(rng: np.random.RandomState, grid: int) -> np.ndarray:
    """
    Density family 1: a sum of 2-4 anisotropic Gaussian blobs.

    Each blob has a random centre in [0.08, 0.92]^2, random widths in
    [0.05, 0.25] per axis and a random intensity in [0.4, 2.0]; the field is
    rescaled so that its maximum is 10. Inputs: rng (consumed in a fixed order,
    so scenarios are reproducible) and the grid side length. Returns a
    (grid, grid) float32 array.
    """
    n_modes = rng.randint(2, 5)
    xs, ys  = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
    field   = np.zeros((grid, grid), dtype=np.float64)
    for _ in range(n_modes):
        cx, cy    = rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)
        sx, sy    = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25)
        intensity = rng.uniform(0.4, 2.0)
        field    += intensity * np.exp(
            -((xs - cx) ** 2 / (2 * sx ** 2) + (ys - cy) ** 2 / (2 * sy ** 2))
        )
    return (field / (field.max() + 1e-12) * 10.0).astype(np.float32)


def make_spike_grid(rng: np.random.RandomState, grid: int) -> np.ndarray:
    """
    Density family 2: sparse spikes. Between 1 and 6 cells (fewer on tiny
    grids) are chosen without replacement and given a random value in
    [0.4, 1.0]; every other cell is exactly zero. Rescaled so the maximum is
    10. These are the hardest scenarios: all of the density sits in a handful
    of cells that the sparse sweeps can miss entirely.
    """
    n_spikes = rng.randint(1, max(2, min(7, grid * grid)))
    cells    = rng.choice(grid * grid, size=n_spikes, replace=False)
    field    = np.zeros(grid * grid, dtype=np.float32)
    field[cells] = rng.uniform(0.4, 1.0, size=n_spikes).astype(np.float32)
    return (field.reshape(grid, grid) / (field.max() + 1e-12) * 10.0)


def make_ring_grid(rng: np.random.RandomState, grid: int) -> np.ndarray:
    """
    Density family 3: 1-3 concentric rings around a random centre in
    [0.2, 0.8]^2. Each ring is a Gaussian shell of common width sigma in
    [0.04, 0.12] at a random radius (kept inside the grid) with a random
    intensity in [0.4, 1.0]. Rescaled so the maximum is 10.
    """
    xs, ys  = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
    cx, cy  = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    r       = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    n_rings = rng.randint(1, 4)
    sigma   = rng.uniform(0.04, 0.12)
    max_r   = max(0.08, min(cx, 1 - cx, cy, 1 - cy) * 0.9)
    radii   = np.sort(rng.uniform(0.04, max_r, size=n_rings))
    field   = np.zeros((grid, grid), dtype=np.float64)
    for radius in radii:
        field += rng.uniform(0.4, 1.0) * np.exp(-(r - radius) ** 2 / (2 * sigma ** 2))
    return (field / (field.max() + 1e-12) * 10.0).astype(np.float32)


def make_observations(density, n_obs, k_samples, scale, rng):
    """
    The sparse, noisy survey sweeps that are the model's only input.

    For each of the n_obs sweeps: choose k_samples cells uniformly without
    replacement, draw a Poisson count with mean density * scale for each,
    add a small Gaussian jitter (sd 0.02 * scale), clip at zero, and leave
    every unobserved cell at 0. Returns a (n_obs, grid, grid) float32 array,
    i.e. one CNN input channel per sweep. k_samples is clamped to the number
    of cells so that tiny grids still work.
    """
    flat    = density.flatten()
    n_cells = len(flat)
    k_samples = min(k_samples, n_cells)   # safety clamp for small grids
    obs     = []
    for _ in range(n_obs):
        idx  = rng.choice(n_cells, size=k_samples, replace=False)
        vals = rng.poisson(flat[idx] * scale).astype(np.float64)
        vals += rng.randn(k_samples) * 0.02 * scale
        vals  = np.clip(vals, 0.0, None)
        sparse       = np.zeros(n_cells, dtype=np.float64)
        sparse[idx]  = vals
        obs.append(sparse.reshape(density.shape).astype(np.float32))
    return np.stack(obs, axis=0)


def generate_dataset(n: int, cfg: dict, rng: np.random.RandomState):
    """
    Generate n scenarios. The density family is drawn by rng.randint(0, 4):
    0 or 1 -> Gaussian blobs, 2 -> sparse spikes, 3 -> rings; its sweeps are
    then drawn with make_observations(). Returns (observations of shape
    (n, n_obs, grid, grid), densities of shape (n, grid, grid)).
    """
    obs_list, den_list = [], []
    for i in range(n):
        kind = rng.randint(0, 4)
        if kind < 2:
            d = make_gaussian_grid(rng, cfg["grid"])
        elif kind == 2:
            d = make_spike_grid(rng, cfg["grid"])
        else:
            d = make_ring_grid(rng, cfg["grid"])
        o = make_observations(d, cfg["n_obs"], cfg["k_samples"], cfg["poisson_scale"], rng)
        den_list.append(d)
        obs_list.append(o)
        _bar(i + 1, n, prefix=f"  gen {n} grids")
    return np.stack(obs_list), np.stack(den_list)


# ─── MPS-COMPATIBLE SOFTPLUS ──────────────────────────────────────────────────
# nn.Softplus(beta=1, threshold=20) internally passes integer constants to the
# MPS Metal kernel, which lowers them to si64 tensors.  MPS then rejects the
# f32×si64 multiply with a fatal MLIR verification error.  This hand-rolled
# version uses only float operations and is numerically equivalent for all
# practical inputs (clamp(max=20) matches the threshold default).

class _Softplus(nn.Module):
    """Softplus written with float-only ops so that it also runs on Apple MPS."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x.clamp(max=20.0)))


# ─── NEURAL NETWORK MODEL ─────────────────────────────────────────────────────

class GridPredictorCNN(nn.Module):
    """
    The density predictor: a four-layer 3x3 convolutional network.

    n_obs input channels (one per survey sweep) -> 32 -> 64 -> 64 -> 32
    channels, each 3x3 convolution followed by SiLU (the first also by
    InstanceNorm and Dropout2d(0.10)), then a 1x1 convolution to one channel
    and a softplus so the density is non-negative. forward() rescales every
    predicted map so that its maximum equals 10, matching the normalisation
    of the true densities. Input (B, n_obs, G, G) -> output (B, G, G).
    """
    def __init__(self, n_obs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_obs, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.SiLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            _Softplus(),    # MPS-safe (avoids si64 constant bug in nn.Softplus)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.net(x).squeeze(1)
        vmax = out.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-9)
        return out / vmax * 10.0


# ─── GAME SOLVER ──────────────────────────────────────────────────────────────

def solve_sse_full_game(density: np.ndarray, catch_penalty: float):
    """
    Strong Stackelberg equilibrium of the patrol game for a given density.

    For N <= 500 cells (every configured grid) the exact multiple-LP method:
    for each candidate attacker target t solve the LP that maximises the
    defender's payoff against t subject to t being an attacker best response
    (sum p = 1, 0 <= p <= 1; scipy HiGHS) and keep the target with the
    highest defender value. Payoffs: D[d, a] = +v_a if d == a else -v_a and
    A[d, a] = -P if d == a else +v_a. For larger games a water-filling
    solution found by bisection on the attacker's equilibrium value is used.

    Inputs: density (G, G) array, catch_penalty P. Returns (strategy as a
    (G, G) float32 array, defender utility under *this* density). Score the
    strategy against the true density with eval_vs_truth().
    """
    flat = density.flatten().astype(np.float64)
    n    = len(flat)
    P    = catch_penalty

    if n <= 500:
        v   = flat
        eye = np.eye(n, dtype=bool)
        D   = np.where(eye,  v[np.newaxis, :], -v[np.newaxis, :])
        A   = np.where(eye, -P,                 v[np.newaxis, :])
        best_util = -np.inf
        best_p    = np.ones(n) / n
        for t in range(n):
            c    = -D[:, t]
            A_ub = np.array([A[:, tp] - A[:, t] for tp in range(n) if tp != t])
            res  = linprog(c, A_ub=A_ub, b_ub=np.zeros(n - 1),
                           A_eq=np.ones((1, n)), b_eq=[1.0],
                           bounds=[(0.0, 1.0)] * n, method="highs")
            if res.success and (-res.fun) > best_util:
                best_util = -res.fun
                best_p    = res.x
        return best_p.astype(np.float32).reshape(density.shape), float(best_util)

    # Water-filling solution for large games (not reached for the configured
    # grids, whose largest game has N = 225 cells).
    def coverage_sum(U):
        mask = flat > U
        if not mask.any(): return 0.0
        return float(np.sum((flat[mask] - U) / (flat[mask] + P)))

    if coverage_sum(float(flat.max()) - 1e-12) <= 1.0:
        p = np.zeros(n); p[int(np.argmax(flat))] = 1.0
        strat = p.reshape(density.shape)
        return strat, eval_vs_truth(strat, density, P)

    lo, hi = float(flat.min()) - 1.0, float(flat.max())
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if coverage_sum(mid) > 1.0: lo = mid
        else:                        hi = mid
    U_star  = (lo + hi) / 2.0
    p       = np.zeros(n)
    covered = flat > U_star
    p[covered] = (flat[covered] - U_star) / (flat[covered] + P)
    p = np.clip(p, 0.0, None)
    if p.sum() > 1e-12: p /= p.sum()
    strat = p.reshape(density.shape)
    return strat, eval_vs_truth(strat, density, P)


def eval_vs_truth(strategy: np.ndarray, true_density: np.ndarray,
                  catch_penalty: float) -> float:
    """
    Defender utility of a committed strategy p against the TRUE density.

    The attacker picks the cell maximising v_a - p_a (v_a + P); among cells
    tied for that maximum (within 1e-9) the one most favourable to the
    defender is chosen (strong Stackelberg tie-breaking). The defender's
    utility is then v_t (2 p_t - 1): +v_t if the attack is always caught,
    -v_t if the cell is never patrolled.
    """
    p          = strategy.flatten().astype(np.float64)
    v          = true_density.flatten().astype(np.float64)
    attacker_eu = v - p * (v + catch_penalty)
    max_eu     = float(attacker_eu.max())
    tied       = attacker_eu >= max_eu - 1e-9
    eu_d       = np.where(tied, v * (2.0 * p - 1.0), -np.inf)
    t_star     = int(np.argmax(eu_d))
    return float(v[t_star] * (2.0 * p[t_star] - 1.0))


# ─── SHARED TRAINING UTILITIES ────────────────────────────────────────────────

def _val_loss(model, Xvl, Yvl, criterion):
    """Huber loss of the model on the validation split (no gradient)."""
    model.eval()
    with torch.no_grad():
        return criterion(model(Xvl), Yvl).item()


def _strategic_score(model, obs, den_true, cfg, device):
    """
    Mean defender utility of the model's induced strategies on (obs, den_true):
    predict, solve the SSE of the predicted game, evaluate that strategy
    against the truth. This is the checkpoint-selection criterion of ZOC,
    KKT and Hybrid.
    """
    model.eval()
    X = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X).cpu().numpy()
    total = 0.0
    for i in range(len(obs)):
        strat, _ = solve_sse_full_game(preds[i], cfg["catch_penalty"])
        total   += eval_vs_truth(strat, den_true[i], cfg["catch_penalty"])
    return total / max(len(obs), 1)


# ─── KKT HELPERS ──────────────────────────────────────────────────────────────
#
# These enable backpropagation through the Stackelberg LP via cvxpylayers.
#
# Payoff structure (same as solve_sse_full_game):
#   D[d,a] = +V[a] if d==a,  −V[a] otherwise
#   A[d,a] = −P    if d==a,  +V[a] otherwise
#
# For fixed attacker target t*, the LP is:
#   max  D[:,t*] · p
#   s.t. (A[:,t'] − A[:,t*]) · p ≤ 0   ∀ t' ≠ t*   (incentive-compatibility)
#        sum(p) = 1,  p ≥ 0

def _kkt_build_lp_params(V_hat: torch.Tensor, target_t: int, P: float):
    """
    Build cvxpylayers parameters d_col (N,) and br_matrix (N-1, N) for target_t.
    V_hat must be a 1-D CPU tensor.  Gradient flows through both outputs.
    """
    N   = V_hat.shape[0]
    eye = torch.eye(N, dtype=V_hat.dtype, device=V_hat.device)
    D   = V_hat.unsqueeze(0) * (2.0 * eye - 1.0)       # (N, N)
    A   = V_hat.unsqueeze(0) * (1.0 - eye) - P * eye   # (N, N)
    d_col    = D[:, target_t]                            # (N,)
    t_primes = [tp for tp in range(N) if tp != target_t]
    br_rows  = torch.stack([A[:, tp] - A[:, target_t] for tp in t_primes])  # (N-1, N)
    return d_col, br_rows


def _make_sse_lp_layer(N: int) -> "CvxpyLayer":
    """Compile the Stackelberg LP as a differentiable cvxpylayers layer."""
    p        = cp.Variable(N, nonneg=True)
    d_col_p  = cp.Parameter(N,          name="d_col")
    br_mat_p = cp.Parameter((N - 1, N), name="br_matrix")
    prob     = cp.Problem(
        cp.Maximize(d_col_p @ p),
        [br_mat_p @ p <= 0, cp.sum(p) == 1],
    )
    return CvxpyLayer(prob, parameters=[d_col_p, br_mat_p], variables=[p])


def _find_sse_target(V_flat: np.ndarray, P: float) -> int:
    """
    Fast O(N·log(N)) binary search for the SSE attacker target.
    Returns the covered cell with the highest density (stable tie-breaking).
    """
    def coverage_sum(U):
        mask = V_flat > U
        if not mask.any(): return 0.0
        return float(np.sum((V_flat[mask] - U) / (V_flat[mask] + P)))

    if coverage_sum(float(V_flat.max()) - 1e-12) <= 1.0:
        return int(np.argmax(V_flat))

    lo, hi = float(V_flat.min()) - 1.0, float(V_flat.max())
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if coverage_sum(mid) > 1.0: lo = mid
        else:                        hi = mid
    U_star  = (lo + hi) / 2.0
    covered = np.where(V_flat > U_star)[0]
    return int(covered[np.argmax(V_flat[covered])]) if len(covered) > 0 else int(np.argmax(V_flat))


# ─── TRAINING: PTO ────────────────────────────────────────────────────────────

def train_pto(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=SEED):
    """
    PTO: full-batch Huber regression on the training split with cosine-
    annealed AdamW for cfg["n_epochs"] epochs; the final weights are returned
    (no strategic checkpointing). Inputs: train/validation observations and
    densities, cfg (a GRID_CONFIGS entry), device, seed (torch.manual_seed).
    Returns the trained GridPredictorCNN.
    """
    torch.manual_seed(seed)
    model     = GridPredictorCNN(cfg["n_obs"]).to(device)
    opt       = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, cfg["n_epochs"])
    criterion = nn.HuberLoss()

    Xtr = torch.tensor(obs_tr, dtype=torch.float32).to(device)
    Ytr = torch.tensor(den_tr, dtype=torch.float32).to(device)
    Xvl = torch.tensor(obs_vl, dtype=torch.float32).to(device)
    Yvl = torch.tensor(den_vl, dtype=torch.float32).to(device)

    ne = cfg["n_epochs"]
    print(f"\n  [PTO] {ne} epochs | {len(obs_tr)} train scenarios | device={device}")
    t0 = time.time()

    for epoch in range(1, ne + 1):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xtr), Ytr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        _bar(epoch, ne, prefix="PTO training")
        if epoch % max(1, ne // 5) == 0 or epoch == ne:
            vl = _val_loss(model, Xvl, Yvl, criterion)
            print(f"  epoch {epoch:4d}/{ne} | train={loss.item():.5f}"
                  f" | val={vl:.5f} | lr={opt.param_groups[0]['lr']:.2e}"
                  f" | {time.time()-t0:.0f}s")

    print(f"  [PTO] done in {time.time()-t0:.1f}s")
    return model


# ─── TRAINING: ZOC ────────────────────────────────────────────────────────────

def train_zoc(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=SEED):
    """
    ZOC (zeroth-order checkpointing). The same Huber training as PTO, but the
    training split is first divided 85/15 (permutation seeded by `seed`) into
    a backprop part and a strategic-evaluation part. Training proceeds in
    windows of n_store epochs; at the end of each window the strategic score
    (_strategic_score) on the evaluation part is measured and the window's
    weights are kept only if the score beats the best so far -- otherwise the
    next window restarts from the best weights (revert). The learning-rate
    schedule keeps advancing either way. Returns the best-scoring weights.
    """
    torch.manual_seed(seed)
    model     = GridPredictorCNN(cfg["n_obs"]).to(device)
    opt       = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, cfg["n_epochs"])
    criterion = nn.HuberLoss()

    n    = len(obs_tr)
    idx  = np.random.RandomState(seed).permutation(n)
    n_bp = int(0.85 * n)
    obs_bp, den_bp = obs_tr[idx[:n_bp]],  den_tr[idx[:n_bp]]
    obs_ev, den_ev = obs_tr[idx[n_bp:]], den_tr[idx[n_bp:]]

    Xbp = torch.tensor(obs_bp, dtype=torch.float32).to(device)
    Ybp = torch.tensor(den_bp, dtype=torch.float32).to(device)
    Xvl = torch.tensor(obs_vl, dtype=torch.float32).to(device)
    Yvl = torch.tensor(den_vl, dtype=torch.float32).to(device)

    ne, ns = cfg["n_epochs"], cfg["n_store"]
    print(f"\n  [ZOC] {ne} epochs | backprop={len(obs_bp)} | strat-eval={len(obs_ev)} | device={device}")

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_score = -np.inf
    t0         = time.time()

    for start in range(1, ne + 1, ns):
        end = min(start + ns - 1, ne)
        model.load_state_dict(best_state)   # every window starts from the best checkpoint (revert)

        for epoch in range(start, end + 1):
            model.train()
            opt.zero_grad()
            loss = criterion(model(Xbp), Ybp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            _bar(epoch, ne, prefix="ZOC training")

        vl        = _val_loss(model, Xvl, Yvl, criterion)
        score     = _strategic_score(model, obs_ev, den_ev, cfg, device) if len(obs_ev) else -np.inf
        candidate = {k: v.clone() for k, v in model.state_dict().items()}
        flag      = " ✓ new best" if score > best_score else ""
        print(f"  epochs {start:4d}-{end:4d} | val={vl:.5f}"
              f" | strat={score:+.5f} | best={best_score:+.5f}{flag}"
              f" | {time.time()-t0:.0f}s")
        if score > best_score:
            best_score = score; best_state = candidate

    model.load_state_dict(best_state)
    print(f"  [ZOC] done in {time.time()-t0:.1f}s | best strategic score: {best_score:+.5f}")
    return model


# ─── TRAINING: KKT ────────────────────────────────────────────────────────────

def train_kkt(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=SEED):
    """
    KKT end-to-end training.

    Directly maximises defender utility by backpropagating through the
    Stackelberg LP via cvxpylayers.  For each mini-batch of scenarios:
      1. CNN → predicted density V_hat  (with gradient)
      2. Fast binary search → attacker target t*  (no gradient)
      3. cvxpylayers LP for t* → p_star  (gradient flows through LP → V_hat)
      4. Evaluate p_star against TRUE density → utility loss

    Falls back to ZOC if cvxpy is not available.
    """
    if not _CVXPY_AVAILABLE:
        print("  [KKT] cvxpy unavailable — falling back to ZOC")
        return train_zoc(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=seed)

    torch.manual_seed(seed)
    model     = GridPredictorCNN(cfg["n_obs"]).to(device)
    opt       = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, cfg["n_epochs"])
    criterion = nn.HuberLoss()

    N = cfg["grid"] ** 2
    P = cfg["catch_penalty"]

    print(f"  [KKT] Compiling LP layer for N={N} cells...", end=" ", flush=True)
    lp_layer = _make_sse_lp_layer(N)
    print("done")

    n    = len(obs_tr)
    idx  = np.random.RandomState(seed).permutation(n)
    n_bp = int(0.85 * n)
    obs_bp, den_bp = obs_tr[idx[:n_bp]], den_tr[idx[:n_bp]]
    obs_ev, den_ev = obs_tr[idx[n_bp:]], den_tr[idx[n_bp:]]

    Xvl = torch.tensor(obs_vl, dtype=torch.float32).to(device)
    Yvl = torch.tensor(den_vl, dtype=torch.float32).to(device)

    ne          = cfg["n_epochs"]
    ns          = cfg["n_store"]
    kkt_batch   = cfg.get("kkt_batch_size", 4)
    kkt_max     = min(cfg.get("kkt_max_per_epoch", 32), len(obs_bp))
    all_bp_idx  = np.arange(len(obs_bp))

    print(f"\n  [KKT] {ne} epochs | backprop={len(obs_bp)} (sample {kkt_max}/epoch)"
          f" | batch={kkt_batch} | device={device}")

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_score = -np.inf
    t0         = time.time()

    for epoch in range(1, ne + 1):
        model.train()
        # Deterministic per-epoch subsample of kkt_max scenarios (seeded by the
        # epoch number, independent of `seed`).
        np.random.RandomState(epoch + 1000).shuffle(all_bp_idx)
        epoch_idx      = all_bp_idx[:kkt_max]
        epoch_kkt_loss = 0.0
        n_steps        = 0

        for start in range(0, len(epoch_idx), kkt_batch):
            batch_idx  = epoch_idx[start:start + kkt_batch]
            opt.zero_grad()
            total_util = torch.tensor(0.0)   # CPU accumulation (LP always on CPU)
            n_valid    = 0

            for ii in batch_idx:
                x    = torch.tensor(obs_bp[ii:ii+1], dtype=torch.float32).to(device)
                pred = model(x)[0]           # (G, G) on device, with grad

                # Move to CPU for LP — PyTorch preserves gradient across device ops
                V_hat = pred.flatten().cpu() if pred.device.type != "cpu" else pred.flatten()

                with torch.no_grad():
                    best_t = _find_sse_target(V_hat.detach().numpy(), P)

                d_col, br_matrix = _kkt_build_lp_params(V_hat, best_t, P)
                # Differentiable LP solve (SCS, tolerance 1e-2): d(p_star)/d(V_hat)
                # flows through the LP's KKT conditions into the CNN.
                try:
                    (p_star,) = lp_layer(
                        d_col, br_matrix,
                        solver_args={"solve_method": "SCS", "eps": 1e-2},
                    )
                    p_star = p_star.float()

                    # Evaluate p_star against TRUE density (CPU)
                    V_true       = torch.tensor(den_bp[ii].flatten(), dtype=torch.float32)
                    eye_t        = torch.eye(N, dtype=torch.float32)
                    D_act        = V_true.unsqueeze(0) * (2.0 * eye_t - 1.0)
                    A_act        = V_true.unsqueeze(0) * (1.0 - eye_t) - P * eye_t
                    with torch.no_grad():
                        t_act = int((p_star.detach() @ A_act).argmax().item())
                    util        = p_star @ D_act[:, t_act]   # scalar with grad
                    total_util  = total_util + util
                    n_valid    += 1
                except Exception:
                    pass   # LP infeasible for this scenario — skip

            if n_valid == 0:
                sched.step(); continue

            loss = -total_util / n_valid   # minimise negative utility
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_kkt_loss += loss.item()
            sched.step()
            n_steps += 1

        _bar(epoch, ne, prefix="KKT training")

        # Checkpoint selection: every n_store epochs keep the weights iff the
        # strategic score on the held-out 15% improved.
        if epoch % ns == 0 or epoch == ne:
            vl      = _val_loss(model, Xvl, Yvl, criterion)
            score   = _strategic_score(model, obs_ev, den_ev, cfg, device) if len(obs_ev) else -np.inf
            flag    = " ✓ new best" if score > best_score else ""
            print(f"  epoch {epoch:4d}/{ne} | kkt_loss={epoch_kkt_loss/max(n_steps,1):.5f}"
                  f" | val={vl:.5f} | strat={score:+.5f}{flag} | {time.time()-t0:.0f}s")
            if score > best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  [KKT] done in {time.time()-t0:.1f}s | best strategic score: {best_score:+.5f}")
    return model


# ─── TRAINING: HYBRID ─────────────────────────────────────────────────────────

def train_hybrid(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=SEED):
    """
    Hybrid training: alternating Huber and KKT regimes with ZO3 revert-to-best.

    Odd  regimes (1, 3, 5, …): Huber loss + noise injection  [stable anchor]
    Even regimes (2, 4, 6, …): KKT differentiable LP loss     [strategic signal]

    Each regime starts from the best checkpoint so far.  After the regime,
    the checkpoint is accepted only if strategic utility improves — otherwise
    we revert.  Falls back to ZOC if cvxpy is not available.
    """
    if not _CVXPY_AVAILABLE:
        print("  [Hybrid] cvxpy unavailable — falling back to ZOC")
        return train_zoc(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=seed)

    torch.manual_seed(seed)
    model     = GridPredictorCNN(cfg["n_obs"]).to(device)
    opt       = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, cfg["n_epochs"])
    criterion = nn.HuberLoss()

    N = cfg["grid"] ** 2
    P = cfg["catch_penalty"]

    print(f"  [Hybrid] Compiling LP layer for N={N} cells...", end=" ", flush=True)
    lp_layer = _make_sse_lp_layer(N)
    print("done")

    n    = len(obs_tr)
    idx  = np.random.RandomState(seed).permutation(n)
    n_bp = int(0.85 * n)
    obs_bp, den_bp = obs_tr[idx[:n_bp]], den_tr[idx[:n_bp]]
    obs_ev, den_ev = obs_tr[idx[n_bp:]], den_tr[idx[n_bp:]]

    Xbp = torch.tensor(obs_bp, dtype=torch.float32).to(device)
    Ybp = torch.tensor(den_bp, dtype=torch.float32).to(device)
    Xvl = torch.tensor(obs_vl, dtype=torch.float32).to(device)
    Yvl = torch.tensor(den_vl, dtype=torch.float32).to(device)

    ne          = cfg["n_epochs"]
    ns          = cfg["n_store"]
    kkt_batch   = cfg.get("kkt_batch_size", 4)
    kkt_max     = min(cfg.get("kkt_max_per_epoch", 32), len(obs_bp))
    noise_std   = 0.01
    all_bp_idx  = np.arange(len(obs_bp))

    print(f"\n  [Hybrid] {ne} epochs | ns={ns} | kkt_max={kkt_max}/epoch | device={device}")

    best_state  = {k: v.clone() for k, v in model.state_dict().items()}
    best_score  = -np.inf
    regime_num  = 0
    t0          = time.time()

    for start in range(1, ne + 1, ns):
        end        = min(start + ns - 1, ne)
        regime_num += 1
        use_kkt    = (regime_num % 2 == 0)
        phase      = "KKT  " if use_kkt else "Huber"

        model.load_state_dict(best_state)   # revert-to-best at the start of every regime

        for epoch in range(start, end + 1):
            model.train()

            if use_kkt:
                # per-epoch subsample, seeded by the epoch number (offset 2000)
                np.random.RandomState(epoch + 2000).shuffle(all_bp_idx)
                epoch_idx  = all_bp_idx[:kkt_max]
                epoch_loss = 0.0; n_steps = 0

                for s in range(0, len(epoch_idx), kkt_batch):
                    bidx       = epoch_idx[s:s + kkt_batch]
                    opt.zero_grad()
                    total_util = torch.tensor(0.0)
                    n_valid    = 0

                    for ii in bidx:
                        x     = torch.tensor(obs_bp[ii:ii+1], dtype=torch.float32).to(device)
                        pred  = model(x)[0]
                        V_hat = pred.flatten().cpu() if pred.device.type != "cpu" else pred.flatten()
                        with torch.no_grad():
                            best_t = _find_sse_target(V_hat.detach().numpy(), P)
                        d_col, br_matrix = _kkt_build_lp_params(V_hat, best_t, P)
                        # differentiable LP solve, as in train_kkt
                        try:
                            (p_star,) = lp_layer(d_col, br_matrix,
                                                  solver_args={"solve_method": "SCS", "eps": 1e-2})
                            p_star = p_star.float()
                            V_true = torch.tensor(den_bp[ii].flatten(), dtype=torch.float32)
                            eye_t  = torch.eye(N, dtype=torch.float32)
                            D_act  = V_true.unsqueeze(0) * (2.0 * eye_t - 1.0)
                            A_act  = V_true.unsqueeze(0) * (1.0 - eye_t) - P * eye_t
                            with torch.no_grad():
                                t_act = int((p_star.detach() @ A_act).argmax().item())
                            total_util += p_star @ D_act[:, t_act]
                            n_valid    += 1
                        except Exception:
                            pass

                    if n_valid == 0:
                        sched.step(); continue
                    loss = -total_util / n_valid
                    if loss.requires_grad:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        epoch_loss += loss.item()
                    sched.step(); n_steps += 1

            else:   # Huber regime with noise injection for robustness
                opt.zero_grad()
                noisy_X = Xbp + torch.randn_like(Xbp) * noise_std
                loss    = criterion(model(noisy_X), Ybp)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()

            _bar(epoch, ne, prefix=f"Hyb[{phase}]")

        # End of regime: strategic evaluation and revert-to-best decision
        candidate = {k: v.clone() for k, v in model.state_dict().items()}
        score     = _strategic_score(model, obs_ev, den_ev, cfg, device) if len(obs_ev) else -np.inf
        vl        = _val_loss(model, Xvl, Yvl, criterion)
        flag      = "✓ ACCEPTED" if score > best_score else "✗ REVERTED"
        print(f"\n  [Hybrid] Regime {regime_num} [{phase}] epochs {start}-{end}"
              f" | strat={score:+.5f} | best={best_score:+.5f} | {flag}"
              f" | val={vl:.5f} | {time.time()-t0:.0f}s")
        if score > best_score:
            best_score = score; best_state = candidate

    model.load_state_dict(best_state)
    print(f"  [Hybrid] done in {time.time()-t0:.1f}s | best score: {best_score:+.5f}")
    return model


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate_all(obs, den_true, pto_model, zoc_model, kkt_model, hyb_model, cfg, device):
    """
    Evaluate all four trained models on the test set alongside the Optimal baseline.
    Winner per scenario = the method with the highest utility (among the 4 learned).
    """
    for m in [pto_model, zoc_model, kkt_model, hyb_model]:
        m.eval()

    X = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        pto_preds = pto_model(X).cpu().numpy()
        zoc_preds = zoc_model(X).cpu().numpy()
        kkt_preds = kkt_model(X).cpu().numpy()
        hyb_preds = hyb_model(X).cpu().numpy()

    rows                                                   = []
    opt_s_all, zoc_s_all, pto_s_all, kkt_s_all, hyb_s_all = [], [], [], [], []
    n = len(obs); P = cfg["catch_penalty"]

    hdr = f"  {'Scen':>4} | {'Optimal':>9} | {'ZOC':>9} | {'PTO':>9} | {'KKT':>9} | {'Hybrid':>9} | Best"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for i in range(n):
        _bar(i + 1, n, prefix="  eval")
        gt = den_true[i]

        opt_s, opt_u = solve_sse_full_game(gt, P)

        pto_s, _ = solve_sse_full_game(pto_preds[i], P)
        zoc_s, _ = solve_sse_full_game(zoc_preds[i], P)
        kkt_s, _ = solve_sse_full_game(kkt_preds[i], P)
        hyb_s, _ = solve_sse_full_game(hyb_preds[i], P)

        pto_u = eval_vs_truth(pto_s, gt, P)
        zoc_u = eval_vs_truth(zoc_s, gt, P)
        kkt_u = eval_vs_truth(kkt_s, gt, P)
        hyb_u = eval_vs_truth(hyb_s, gt, P)

        opt_s_all.append(opt_s); zoc_s_all.append(zoc_s); pto_s_all.append(pto_s)
        kkt_s_all.append(kkt_s); hyb_s_all.append(hyb_s)

        # Determine the winner among the four learned methods: a method wins
        # only if it beats every other method by more than eps; otherwise the
        # scenario is a tie (shared best utility).
        best_u = max(zoc_u, pto_u, kkt_u, hyb_u)
        eps    = 1e-6
        if   zoc_u >= best_u - eps and zoc_u > pto_u + eps and zoc_u > kkt_u + eps and zoc_u > hyb_u + eps:
            winner = "ZOC"
        elif pto_u >= best_u - eps and pto_u > zoc_u + eps and pto_u > kkt_u + eps and pto_u > hyb_u + eps:
            winner = "PTO"
        elif kkt_u >= best_u - eps and kkt_u > zoc_u + eps and kkt_u > pto_u + eps and kkt_u > hyb_u + eps:
            winner = "KKT"
        elif hyb_u >= best_u - eps and hyb_u > zoc_u + eps and hyb_u > pto_u + eps and hyb_u > kkt_u + eps:
            winner = "Hyb"
        else:
            winner = "tie"

        rows.append(dict(
            scenario=i+1, opt=opt_u, zoc=zoc_u, pto=pto_u, kkt=kkt_u, hyb=hyb_u,
            winner=winner,
            gap_zoc=opt_u-zoc_u, gap_pto=opt_u-pto_u, gap_kkt=opt_u-kkt_u, gap_hyb=opt_u-hyb_u,
        ))
        print(f"  {i+1:>4d} | {opt_u:>+9.4f} | {zoc_u:>+9.4f} | {pto_u:>+9.4f}"
              f" | {kkt_u:>+9.4f} | {hyb_u:>+9.4f} | {winner}")

    opt_u_all = [r["opt"] for r in rows]
    zoc_u_all = [r["zoc"] for r in rows]
    pto_u_all = [r["pto"] for r in rows]
    kkt_u_all = [r["kkt"] for r in rows]
    hyb_u_all = [r["hyb"] for r in rows]
    zoc_wins  = sum(1 for r in rows if r["winner"] == "ZOC")
    pto_wins  = sum(1 for r in rows if r["winner"] == "PTO")
    kkt_wins  = sum(1 for r in rows if r["winner"] == "KKT")
    hyb_wins  = sum(1 for r in rows if r["winner"] == "Hyb")
    ties      = sum(1 for r in rows if r["winner"] == "tie")

    print(f"\n  {'MEAN':>4} | {np.mean(opt_u_all):>+9.4f} | {np.mean(zoc_u_all):>+9.4f}"
          f" | {np.mean(pto_u_all):>+9.4f} | {np.mean(kkt_u_all):>+9.4f} | {np.mean(hyb_u_all):>+9.4f}")
    print(f"  {'STD':>4} | {np.std(opt_u_all):>9.4f} | {np.std(zoc_u_all):>9.4f}"
          f" | {np.std(pto_u_all):>9.4f} | {np.std(kkt_u_all):>9.4f} | {np.std(hyb_u_all):>9.4f}")
    print(f"\n  Win rates: ZOC={zoc_wins}/{n}  PTO={pto_wins}/{n}"
          f"  KKT={kkt_wins}/{n}  Hyb={hyb_wins}/{n}  tie={ties}")
    print(f"  Gap-to-optimal: ZOC={np.mean([r['gap_zoc'] for r in rows]):.4f}"
          f"  PTO={np.mean([r['gap_pto'] for r in rows]):.4f}"
          f"  KKT={np.mean([r['gap_kkt'] for r in rows]):.4f}"
          f"  Hyb={np.mean([r['gap_hyb'] for r in rows]):.4f}")

    # ── per-scenario MAE (predicted density vs true density) ──────────────────
    pto_maes = [float(np.mean(np.abs(pto_preds[i] - den_true[i]))) for i in range(n)]
    zoc_maes = [float(np.mean(np.abs(zoc_preds[i] - den_true[i]))) for i in range(n)]
    kkt_maes = [float(np.mean(np.abs(kkt_preds[i] - den_true[i]))) for i in range(n)]
    hyb_maes = [float(np.mean(np.abs(hyb_preds[i] - den_true[i]))) for i in range(n)]

    for i, r in enumerate(rows):
        r["mae_pto"] = pto_maes[i]
        r["mae_zoc"] = zoc_maes[i]
        r["mae_kkt"] = kkt_maes[i]
        r["mae_hyb"] = hyb_maes[i]

    print(f"\n  MAE (pred vs truth): PTO={np.mean(pto_maes):.4f}"
          f"  ZOC={np.mean(zoc_maes):.4f}"
          f"  KKT={np.mean(kkt_maes):.4f}"
          f"  Hyb={np.mean(hyb_maes):.4f}")

    return dict(
        rows=rows,
        opt_utils=opt_u_all, zoc_utils=zoc_u_all, pto_utils=pto_u_all,
        kkt_utils=kkt_u_all, hyb_utils=hyb_u_all,
        pto_maes=pto_maes, zoc_maes=zoc_maes, kkt_maes=kkt_maes, hyb_maes=hyb_maes,
        opt_strats=opt_s_all, zoc_strats=zoc_s_all, pto_strats=pto_s_all,
        kkt_strats=kkt_s_all, hyb_strats=hyb_s_all,
        zoc_preds=zoc_preds, pto_preds=pto_preds,
        kkt_preds=kkt_preds, hyb_preds=hyb_preds,
        zoc_wins=zoc_wins, pto_wins=pto_wins, kkt_wins=kkt_wins,
        hyb_wins=hyb_wins, ties=ties,
    )


# ─── SAVING ───────────────────────────────────────────────────────────────────

def save_csv(results: dict, path: Path):
    """Write results["rows"] (one line per test scenario) to CSV; floats signed, 6 decimals."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scenario", "opt", "zoc", "pto", "kkt", "hyb",
            "winner", "gap_zoc", "gap_pto", "gap_kkt", "gap_hyb",
            "mae_pto", "mae_zoc", "mae_kkt", "mae_hyb",
        ])
        w.writeheader()
        for r in results["rows"]:
            w.writerow({k: (f"{v:+.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {path}")


def save_summary(results: dict, cfg: dict, run_tag: str, path: Path):
    """Write summary.txt: config, utility and MAE statistics, gaps to optimal, win rates."""
    n   = len(results["opt_utils"])
    opt = results["opt_utils"]
    zoc = results["zoc_utils"]; pto = results["pto_utils"]
    kkt = results["kkt_utils"]; hyb = results["hyb_utils"]
    lines = [
        f"Run tag : {run_tag}",
        f"Config  : {cfg['label']}  catch_penalty={cfg['catch_penalty']}",
        f"Training: N_TRAIN={cfg['n_train']}  N_TEST={n}  "
        f"N_EPOCHS={cfg['n_epochs']}  N_STORE={cfg['n_store']}",
        "",
        "─── Utility statistics ───",
        f"{'Method':<12} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}",
    ]
    for name, vals in [("Optimal", opt), ("ZOC", zoc), ("PTO", pto), ("KKT", kkt), ("Hybrid", hyb)]:
        lines.append(f"{name:<12} {np.mean(vals):>+9.4f} {np.std(vals):>9.4f}"
                     f" {np.min(vals):>+9.4f} {np.max(vals):>+9.4f}")
    lines += [
        "",
        "─── Gap to optimal (mean) ───",
        f"  ZOC: {np.mean([r['gap_zoc'] for r in results['rows']]):.4f}",
        f"  PTO: {np.mean([r['gap_pto'] for r in results['rows']]):.4f}",
        f"  KKT: {np.mean([r['gap_kkt'] for r in results['rows']]):.4f}",
        f"  Hyb: {np.mean([r['gap_hyb'] for r in results['rows']]):.4f}",
        "",
        "─── MAE statistics (predicted density vs ground truth) ───",
        f"{'Method':<12} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}",
    ]
    for name, key in [("ZOC", "zoc"), ("PTO", "pto"), ("KKT", "kkt"), ("Hybrid", "hyb")]:
        vals = results[f"{key}_maes"]
        lines.append(f"{name:<12} {np.mean(vals):>9.4f} {np.std(vals):>9.4f}"
                     f" {np.min(vals):>9.4f} {np.max(vals):>9.4f}")
    lines += [
        "",
        "─── Win rates (best of 4 methods per scenario) ───",
        f"  ZOC: {results['zoc_wins']}/{n}  ({100*results['zoc_wins']/n:.1f}%)",
        f"  PTO: {results['pto_wins']}/{n}  ({100*results['pto_wins']/n:.1f}%)",
        f"  KKT: {results['kkt_wins']}/{n}  ({100*results['kkt_wins']/n:.1f}%)",
        f"  Hyb: {results['hyb_wins']}/{n}  ({100*results['hyb_wins']/n:.1f}%)",
        f"  Ties: {results['ties']}/{n}  ({100*results['ties']/n:.1f}%)",
    ]
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_heatmap_panel(results, den_test, obs_test, indices, cfg, out_dir, run_tag):
    """
    6-column heatmap panel per scenario row.
      Col 0 — Ground Truth : true density (top) | Optimal strategy (bottom)
      Col 1 — Input        : mean noisy observation (top only)
      Col 2 — ZOC          : predicted density (top) | ZOC strategy (bottom)
      Col 3 — PTO          : predicted density (top) | PTO strategy (bottom)
      Col 4 — KKT          : predicted density (top) | KKT strategy (bottom)
      Col 5 — Hybrid       : predicted density (top) | Hybrid strategy (bottom)
    """
    n   = len(indices)
    fig = plt.figure(figsize=(4.2 * 6, 3.8 * n))
    fig.suptitle(
        f"PTO / ZOC / KKT / Hybrid — {cfg['label']} — {run_tag}\n"
        "Top row: predicted density  |  Bottom row: SSE patrol strategy",
        fontsize=12, fontweight="bold", y=1.003,
    )
    gs         = gridspec.GridSpec(n * 2, 6, hspace=0.55, wspace=0.35)
    col_titles = ["Ground Truth", "Input", "ZOC", "PTO", "KKT", "Hybrid"]
    util_cols  = ["darkgreen", None, "navy", "firebrick", "darkorchid", "darkorange"]

    for ri, sc in enumerate(indices):
        gt  = den_test[sc]
        inp = obs_test[sc].mean(axis=0)
        dens   = [gt,   None, results["zoc_preds"][sc], results["pto_preds"][sc],
                  results["kkt_preds"][sc], results["hyb_preds"][sc]]
        strats = [results["opt_strats"][sc], None, results["zoc_strats"][sc],
                  results["pto_strats"][sc], results["kkt_strats"][sc], results["hyb_strats"][sc]]
        utils  = [results["opt_utils"][sc], None, results["zoc_utils"][sc],
                  results["pto_utils"][sc], results["kkt_utils"][sc], results["hyb_utils"][sc]]

        vmax_d = max(d.max() for d in [gt, inp,
                     results["zoc_preds"][sc], results["pto_preds"][sc],
                     results["kkt_preds"][sc], results["hyb_preds"][sc]]) * 1.05
        vmax_s = max(s.max() for s in [results["opt_strats"][sc], results["zoc_strats"][sc],
                     results["pto_strats"][sc], results["kkt_strats"][sc],
                     results["hyb_strats"][sc]]) * 1.05 + 1e-9

        for col in range(6):
            if col == 1:
                ax_i = fig.add_subplot(gs[ri * 2, 1])
                im   = ax_i.imshow(inp, origin="lower", cmap=CMAP_D,
                                   vmin=0, vmax=vmax_d, aspect="auto")
                if ri == 0:
                    ax_i.set_title(col_titles[1], fontsize=10, fontweight="bold", pad=5)
                ax_i.set_xlabel("Mean noisy input", fontsize=8)
                ax_i.set_xticks([]); ax_i.set_yticks([])
                plt.colorbar(im, ax=ax_i, fraction=0.046, pad=0.03)
                fig.add_subplot(gs[ri * 2 + 1, 1]).axis("off")
                continue

            ax_d = fig.add_subplot(gs[ri * 2, col])
            im_d = ax_d.imshow(dens[col], origin="lower", cmap=CMAP_D,
                               vmin=0, vmax=vmax_d, aspect="auto")
            if ri == 0:
                ax_d.set_title(col_titles[col], fontsize=10, fontweight="bold", pad=5)
            if col == 0:
                ax_d.set_ylabel(f"Scenario {sc+1}\nDensity", fontsize=8)
            ax_d.set_xticks([]); ax_d.set_yticks([])
            plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.03)

            ax_s = fig.add_subplot(gs[ri * 2 + 1, col])
            im_s = ax_s.imshow(strats[col], origin="lower", cmap=CMAP_S,
                               vmin=0, vmax=vmax_s, aspect="auto")
            ax_s.set_xlabel(f"Utility: {utils[col]:+.4f}", fontsize=8,
                            color=util_cols[col], fontweight="bold")
            if col == 0:
                ax_s.set_ylabel("Patrol Strategy (SSE)", fontsize=8)
            ax_s.set_xticks([]); ax_s.set_yticks([])
            plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.03)

    out = out_dir / "heatmap_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_utility_bar(results, cfg, out_dir, run_tag):
    """Grouped bar chart: Optimal, ZOC, PTO, KKT, Hybrid per test scenario."""
    n = len(results["opt_utils"])
    x = np.arange(n)
    w = 0.15

    fig, ax = plt.subplots(figsize=(max(14, n * 1.0), 5))
    ax.bar(x - 2*w, results["opt_utils"], w, label="Optimal", color="steelblue",  alpha=0.88)
    ax.bar(x - 1*w, results["zoc_utils"], w, label="ZOC",     color="navy",       alpha=0.88)
    ax.bar(x,       results["pto_utils"], w, label="PTO",     color="firebrick",  alpha=0.88)
    ax.bar(x + 1*w, results["kkt_utils"], w, label="KKT",     color="darkorchid", alpha=0.88)
    ax.bar(x + 2*w, results["hyb_utils"], w, label="Hybrid",  color="darkorange", alpha=0.88)

    for vals, col in zip(
        [results["opt_utils"], results["zoc_utils"], results["pto_utils"],
         results["kkt_utils"], results["hyb_utils"]],
        ["steelblue", "navy", "firebrick", "darkorchid", "darkorange"],
    ):
        ax.axhline(np.mean(vals), color=col, linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_title(
        f"Defender Utility per Scenario — {cfg['label']} — {run_tag}\n"
        f"Mean → Opt:{np.mean(results['opt_utils']):+.4f} | "
        f"ZOC:{np.mean(results['zoc_utils']):+.4f} | "
        f"PTO:{np.mean(results['pto_utils']):+.4f} | "
        f"KKT:{np.mean(results['kkt_utils']):+.4f} | "
        f"Hyb:{np.mean(results['hyb_utils']):+.4f}", fontsize=9)
    ax.set_xlabel("Test Scenario"); ax.set_ylabel("Defender Utility (SSE)")
    ax.set_xticks(x); ax.set_xticklabels([str(i+1) for i in range(n)], fontsize=7)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "utility_bar.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_utility_violin(results, cfg, out_dir, run_tag):
    """Violin distributions for all five methods (Optimal + 4 learned)."""
    data   = [results["opt_utils"], results["zoc_utils"], results["pto_utils"],
               results["kkt_utils"], results["hyb_utils"]]
    labels = ["Optimal", "ZOC", "PTO", "KKT", "Hybrid"]
    colors = ["steelblue", "navy", "firebrick", "darkorchid", "darkorange"]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts   = ax.violinplot(data, positions=[1, 2, 3, 4, 5], showmeans=True, showmedians=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.60)
    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(1.4)

    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Defender Utility (SSE)", fontsize=11)
    ax.set_title(f"Utility Distributions — {cfg['label']} — {run_tag}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[mpatches.Patch(facecolor=c, alpha=0.7, label=l)
                       for c, l in zip(colors, labels)], fontsize=9)
    fig.tight_layout()
    out = out_dir / "utility_violin.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_utility_scatter(results, cfg, out_dir, run_tag):
    """
    3-panel scatter: ZOC vs PTO | KKT vs PTO | Hybrid vs PTO.
    Points above the diagonal = that method beats PTO on that scenario.
    """
    pto = np.array(results["pto_utils"])
    zoc = np.array(results["zoc_utils"])
    kkt = np.array(results["kkt_utils"])
    hyb = np.array(results["hyb_utils"])
    opt = np.array(results["opt_utils"])

    methods = [("ZOC", zoc, "navy"), ("KKT", kkt, "darkorchid"), ("Hybrid", hyb, "darkorange")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"ZOC / KKT / Hybrid vs PTO — {cfg['label']} — {run_tag}\n"
                 "Points above diagonal → method beats PTO on that scenario",
                 fontsize=11, fontweight="bold")

    for ax, (name, mvals, col) in zip(axes, methods):
        all_min = min(pto.min(), mvals.min()) - 0.02
        all_max = max(pto.max(), mvals.max()) + 0.02
        wins    = int((mvals > pto + 1e-6).sum())
        losses  = int((pto > mvals + 1e-6).sum())
        ties_n  = len(pto) - wins - losses
        colors_pt = ["firebrick" if p > m + 1e-6 else (col if m > p + 1e-6 else "grey")
                     for m, p in zip(mvals, pto)]

        ax.plot([all_min, all_max], [all_min, all_max], "k--", lw=1.2, label="Tie line")
        ax.scatter(pto, mvals, c=colors_pt, s=60, edgecolors="black", lw=0.5, zorder=3)
        ax.axhline(np.mean(opt), color="steelblue", ls=":", lw=1.0, alpha=0.6)
        ax.axvline(np.mean(opt), color="steelblue", ls=":", lw=1.0, alpha=0.6)

        handles = [
            mpatches.Patch(color=col,        label=f"{name} wins ({wins}/{len(pto)})"),
            mpatches.Patch(color="firebrick", label=f"PTO wins ({losses}/{len(pto)})"),
            mpatches.Patch(color="grey",      label=f"Tie ({ties_n}/{len(pto)})"),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel("PTO Defender Utility", fontsize=10)
        ax.set_ylabel(f"{name} Defender Utility", fontsize=10)
        ax.set_title(f"{name} vs PTO", fontsize=11, fontweight="bold")
        ax.set_xlim(all_min, all_max); ax.set_ylim(all_min, all_max)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "utility_scatter.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_gap_to_optimal(results, cfg, out_dir, run_tag):
    """
    Line plot: gap-to-optimal (Optimal − method) per scenario, sorted by Optimal.
    Smaller gap = closer to optimal = better.
    """
    opt = np.array(results["opt_utils"])
    order = np.argsort(opt)   # sort scenarios from easiest to hardest

    fig, ax = plt.subplots(figsize=(max(10, len(opt) * 0.5), 4))
    xs = np.arange(len(opt))

    methods = [
        ("ZOC",    results["zoc_utils"], "navy",       "o"),
        ("PTO",    results["pto_utils"], "firebrick",  "s"),
        ("KKT",    results["kkt_utils"], "darkorchid", "^"),
        ("Hybrid", results["hyb_utils"], "darkorange", "D"),
    ]
    for name, vals, col, mk in methods:
        gaps = opt[order] - np.array(vals)[order]
        ax.plot(xs, gaps, color=col, marker=mk, markersize=5, lw=1.5,
                label=f"{name} (mean gap={np.mean(gaps):.4f})")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Scenario (sorted by Optimal utility, left=easiest)")
    ax.set_ylabel("Gap to Optimal (lower = better)")
    ax.set_title(f"Gap to Optimal per Scenario — {cfg['label']} — {run_tag}", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "gap_to_optimal.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


# ─── COMPARE PANELS ──────────────────────────────────────────────────────────

_COMPARE_COLS  = ["pto", "zoc", "kkt", "hyb"]
_COMPARE_LABEL = {"pto": "PTO", "zoc": "ZOC", "kkt": "KKT", "hyb": "Hybrid"}
_COMPARE_CMAP  = "Blues"


def _one_compare_row(fig, gs_row, ri, sc_idx, den_test, results, show_header):
    """Fill one row (5 axes): GT | PTO | ZOC | KKT | Hybrid."""
    row    = results["rows"][sc_idx]
    gt     = den_test[sc_idx]
    winner = row["winner"]          # e.g. "KKT", "ZOC", "Hyb", "PTO", "tie"

    preds = {k: results[f"{k}_preds"][sc_idx] for k in _COMPARE_COLS}
    utils = {k: row[k if k != "hyb" else "hyb"] for k in _COMPARE_COLS}

    den_max = max(gt.max(), *(p.max() for p in preds.values())) * 1.05
    den_max = max(den_max, 1e-3)

    # ── Ground Truth ──
    ax = fig.add_subplot(gs_row[ri, 0])
    ax.imshow(gt, origin="lower", cmap=_COMPARE_CMAP,
              vmin=0, vmax=den_max, aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if show_header:
        ax.set_title("Ground Truth", fontsize=10, fontweight="bold", color="#222222", pad=4)

    # ── Learned methods ──
    for ci, key in enumerate(_COMPARE_COLS):
        ax      = fig.add_subplot(gs_row[ri, ci + 1])
        u       = utils[key]
        is_last = ci == len(_COMPARE_COLS) - 1
        is_win  = (winner != "tie" and
                   ((key == "hyb" and winner == "Hyb") or
                    (key != "hyb" and winner.upper() == key.upper())))
        # also highlight on ties (all methods share win)
        if winner == "tie":
            is_win = True

        im = ax.imshow(preds[key], origin="lower", cmap=_COMPARE_CMAP,
                       vmin=0, vmax=den_max, aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

        if is_last:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

        if show_header:
            ax.set_title(
                _COMPARE_LABEL[key], fontsize=10, pad=4,
                fontweight="bold" if is_win else "normal",
                color="#1a7a1a" if is_win else "#555555",
            )
        ax.set_xlabel(
            f"$u = {u:+.3f}$", fontsize=9, labelpad=2,
            color="#1a7a1a" if is_win else "#444444",
            fontweight="bold" if is_win else "normal",
        )


def _make_compare_figure(scenario_indices, den_test, results, out_path):
    """Build and save a compare figure for a list of scenario indices (0-based)."""
    n_rows = len(scenario_indices)
    n_cols = 1 + len(_COMPARE_COLS)
    fig    = plt.figure(figsize=(3.1 * n_cols, 3.0 * n_rows + 0.4))
    gs     = gridspec.GridSpec(n_rows, n_cols, hspace=0.55, wspace=0.28,
                               left=0.04, right=0.94, figure=fig)
    for ri, sc in enumerate(scenario_indices):
        _one_compare_row(fig, gs, ri, sc, den_test, results, show_header=(ri == 0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_panels(results, den_test, out_dir):
    """
    Generate per-scenario compare panels and summary regime panels.

    Outputs
    -------
    compare_panels/
        compare_sc{N}.png          — one panel per test scenario
        regime_top6.png            — 6 scenarios with the largest inter-method spread
        wins_pto.png               — up to 4 scenarios where PTO wins (if any)
        wins_zoc.png               — same for ZOC
        wins_kkt.png               — same for KKT
        wins_hyb.png               — same for Hybrid
    """
    panels_dir = out_dir / "compare_panels"
    panels_dir.mkdir(exist_ok=True)

    n = len(results["rows"])

    # ── 1. Individual scenario panels ─────────────────────────────────────────
    print(f"\n  Saving {n} individual compare panels…")
    for i in range(n):
        out = panels_dir / f"compare_sc{i+1}.png"
        _make_compare_figure([i], den_test, results, out)
        _bar(i + 1, n, prefix="  compare panels")
    print(f"  Saved: {panels_dir}/compare_sc1.png … compare_sc{n}.png")

    # ── 2. Top-6 by inter-method spread (most diagnostic) ────────────────────
    rows = results["rows"]
    spreads = sorted(
        range(n),
        key=lambda i: max(rows[i]["pto"], rows[i]["zoc"], rows[i]["kkt"], rows[i]["hyb"])
                    - min(rows[i]["pto"], rows[i]["zoc"], rows[i]["kkt"], rows[i]["hyb"]),
        reverse=True,
    )
    top6 = sorted(spreads[:6])   # chronological order
    out  = panels_dir / "regime_top6.png"
    _make_compare_figure(top6, den_test, results, out)
    print(f"  Saved: {out}  (scenarios {[i+1 for i in top6]})")

    # ── 3. Per-winner panels (up to 4 scenarios each) ────────────────────────
    winner_key = {"PTO": "pto", "ZOC": "zoc", "KKT": "kkt", "Hyb": "hyb"}
    for w_label, w_key in winner_key.items():
        idxs = [i for i, r in enumerate(rows) if r["winner"] == w_label]
        if not idxs:
            continue
        sel = sorted(idxs[:4])
        out = panels_dir / f"wins_{w_key}.png"
        _make_compare_figure(sel, den_test, results, out)
        print(f"  Saved: {out}  ({w_label} wins in scenarios {[i+1 for i in sel]})")


# ─── MAIN ORCHESTRATION ───────────────────────────────────────────────────────

def _checkpoint_model(model, name, obs_test, den_test, cfg, device, run_dir):
    """
    Immediately after a model finishes training:
      · save its weights to disk
      · evaluate on test set and write a per-model CSV + mini-summary
    This ensures partial results are always recoverable even if a later
    model (e.g. KKT) is interrupted.
    """
    key = name.lower()   # file prefix: pto_, zoc_, kkt_, hybrid_ (the paper's stored
                         # run, made by an earlier revision, used hyb_model.pth;
                         # plot_regime_panels.py accepts both names)
    torch.save(model.state_dict(), run_dir / f"{key}_model.pth")

    P = cfg["catch_penalty"]
    model.eval()
    X = torch.tensor(obs_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    utils, maes = [], []
    for i in range(len(obs_test)):
        strat, _ = solve_sse_full_game(preds[i], P)
        utils.append(eval_vs_truth(strat, den_test[i], P))
        maes.append(float(np.mean(np.abs(preds[i] - den_test[i]))))

    mean_u  = float(np.mean(utils))
    std_u   = float(np.std(utils))
    mean_mae = float(np.mean(maes))

    # write per-model CSV
    csv_path = run_dir / f"{key}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "utility", "mae"])
        for i, (u, m) in enumerate(zip(utils, maes)):
            w.writerow([i + 1, f"{u:+.6f}", f"{m:.6f}"])

    print(f"\n  ── {name} checkpoint saved ──")
    print(f"     utility : {mean_u:+.4f} ± {std_u:.4f}")
    print(f"     MAE     : {mean_mae:.4f}")
    print(f"     weights : {key}_model.pth")
    print(f"     results : {key}_results.csv")

    return preds, utils, maes


def run_one(grid_size: int, device: torch.device, seed: int = SEED,
            epochs_override: int = None):
    """
    One complete run for one training seed: generate the data (the data RNG
    is always SEED = 42, so every seed sees the same scenarios), hold out the
    last 20% of the training scenarios for validation, train PTO -> ZOC ->
    Hybrid -> KKT (each is saved and evaluated on the test set as soon as it
    finishes, so partial results survive an interruption), then run the
    combined evaluation, plots, results.csv and summary.txt. Writes to
    OUT_DIR/grid<grid_size>_seed<seed>_<timestamp>/ and returns the
    evaluate_all() dictionary.
    """
    cfg     = GRID_CONFIGS[grid_size]
    if epochs_override is not None:
        cfg = dict(cfg); cfg["n_epochs"] = epochs_override
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"grid{grid_size}_seed{seed}_{ts}"
    run_dir = OUT_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RUN: {run_tag}  seed={seed}")
    print(f"  Config: {cfg['label']}  catch_penalty={cfg['catch_penalty']}")
    print(f"  N_TRAIN={cfg['n_train']}  N_TEST={cfg['n_test']}"
          f"  N_EPOCHS={cfg['n_epochs']}  N_STORE={cfg['n_store']}  LR={cfg['lr']}")
    print(f"  KKT: batch={cfg['kkt_batch_size']}  max_per_epoch={cfg['kkt_max_per_epoch']}")
    print(f"  Order: PTO → ZOC → Hybrid → KKT  (KKT last: slowest)")
    print(f"  cvxpy available: {_CVXPY_AVAILABLE}")
    print(f"{'='*70}")

    rng = np.random.RandomState(SEED)

    print("\n── Generating training data ──")
    obs_all, den_all = generate_dataset(cfg["n_train"], cfg, rng)

    print("\n── Generating test data ──")
    obs_test, den_test = generate_dataset(cfg["n_test"], cfg, rng)

    np.save(run_dir / "obs_test.npy", obs_test)
    np.save(run_dir / "den_test.npy", den_test)

    n_val          = max(int(0.2 * cfg["n_train"]), 4)   # validation = last 20%
    obs_vl, den_vl = obs_all[-n_val:], den_all[-n_val:]
    obs_tr, den_tr = obs_all[:-n_val], den_all[:-n_val]
    print(f"\n  Split → train={len(obs_tr)}  val={len(obs_vl)}  test={len(obs_test)}")

    # ── Train in order: PTO → ZOC → Hybrid → KKT ─────────────────────────────
    # Each model is saved to disk immediately after training completes so that
    # results are preserved even if a subsequent (slower) model is interrupted.

    print("\n── Training PTO ──")
    pto_model = train_pto(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=seed)
    pto_preds, pto_utils, pto_maes = _checkpoint_model(
        pto_model, "PTO", obs_test, den_test, cfg, device, run_dir)

    print("\n── Training ZOC ──")
    zoc_model = train_zoc(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=seed)
    zoc_preds, zoc_utils, zoc_maes = _checkpoint_model(
        zoc_model, "ZOC", obs_test, den_test, cfg, device, run_dir)

    print("\n── Training Hybrid (Alternating Huber + KKT) ──")
    hyb_model = train_hybrid(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=seed)
    hyb_preds, hyb_utils, hyb_maes = _checkpoint_model(
        hyb_model, "Hybrid", obs_test, den_test, cfg, device, run_dir)

    print("\n── Training KKT (Differentiable LP) ──")
    kkt_model = train_kkt(obs_tr, den_tr, obs_vl, den_vl, cfg, device, seed=seed)
    kkt_preds, kkt_utils, kkt_maes = _checkpoint_model(
        kkt_model, "KKT", obs_test, den_test, cfg, device, run_dir)

    # ── Full combined evaluation + plots ──────────────────────────────────────
    print("\n── Evaluating on test set ──")
    results = evaluate_all(obs_test, den_test, pto_model, zoc_model, kkt_model, hyb_model, cfg, device)

    plot_indices = np.linspace(0, cfg["n_test"] - 1, N_PLOT, dtype=int).tolist()

    print("\n── Saving plots and reports ──")
    plot_heatmap_panel(results, den_test, obs_test, plot_indices, cfg, run_dir, run_tag)
    plot_utility_bar(results, cfg, run_dir, run_tag)
    plot_utility_violin(results, cfg, run_dir, run_tag)
    plot_utility_scatter(results, cfg, run_dir, run_tag)
    plot_gap_to_optimal(results, cfg, run_dir, run_tag)
    plot_compare_panels(results, den_test, run_dir)
    save_csv(results, run_dir / "results.csv")
    save_summary(results, cfg, run_tag, run_dir / "summary.txt")

    print(f"\n  All outputs → {run_dir}")
    return results


def _print_multiseed_summary(all_results, grid_size, epochs):
    """Print and return the across-seed utility table (mean +- std over all test
    scenarios of all seeds, plus the per-seed means)."""
    cfg = GRID_CONFIGS[grid_size]
    n_seeds = len(all_results)

    def _agg(key):
        arrs = [np.array(r[f"{key}_utils"]) for r in all_results]
        flat = np.concatenate(arrs)
        per_seed_means = [float(a.mean()) for a in arrs]
        return float(np.mean(flat)), float(np.std(flat)), per_seed_means

    lines = []
    def emit(s=""): print(s); lines.append(s)

    emit(f"\n{'='*68}")
    emit(f"  Synthetic Grid Results — {cfg['label']} (N={grid_size**2})")
    emit(f"  epochs={epochs}  seeds={n_seeds}")
    emit(f"{'='*68}")
    emit(f"  {'Method':<10} {'Utility (mean±std)':>22}  {'Per-seed means'}")
    emit(f"  {'-'*60}")
    for name in ("opt", "pto", "zoc", "kkt", "hyb"):
        label = {"opt":"Optimal","pto":"PTO","zoc":"ZOC","kkt":"KKT","hyb":"Hybrid"}[name]
        mn, sd, per = _agg(name)
        per_str = "  ".join(f"{v:+.3f}" for v in per)
        emit(f"  {label:<10} {mn:>+8.4f} ± {sd:<8.4f}  {per_str}")
    emit(f"{'='*68}\n")
    return "\n".join(lines)


def main():
    """CLI entry point: parse the arguments, run every seed, write the multi-seed summary."""
    global OUT_DIR
    ap = argparse.ArgumentParser(
        description="Synthetic Conservation Security Game (Appendix B.1): "
                    "train and evaluate PTO, ZOC, KKT and Hybrid.")
    ap.add_argument("--grid",   type=int, default=15,
                    help="Grid size (e.g. 10 or 15). Must be in GRID_CONFIGS. "
                         "The paper's run used 10.")
    ap.add_argument("--seeds",  type=int, default=3,
                    help="Number of random seeds to average over.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override n_epochs from config.")
    ap.add_argument("--seed-start", type=int, default=1,
                    help="First training seed; the seeds run are seed-start .. "
                         "seed-start + seeds - 1 (default 1, i.e. seeds 1..N).")
    ap.add_argument("--out",    type=str, default=str(OUT_DIR),
                    help="Output directory; one sub-folder per run is created "
                         "inside it (default: output/ next to this script).")
    args = ap.parse_args()

    if args.grid not in GRID_CONFIGS:
        raise ValueError(f"--grid {args.grid} not in GRID_CONFIGS (have: {list(GRID_CONFIGS)})")

    OUT_DIR = Path(args.out).expanduser()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from utils import log_device
    device = log_device()
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  KKT/Hybrid LP differentiation: {'ENABLED' if _CVXPY_AVAILABLE else 'DISABLED (install cvxpy)'}")

    cfg_epochs = args.epochs if args.epochs is not None else GRID_CONFIGS[args.grid]["n_epochs"]

    all_results = []
    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds), start=1):
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}  ({i}/{args.seeds})")
        print(f"{'#'*70}")
        res = run_one(args.grid, device, seed=seed, epochs_override=args.epochs)
        all_results.append(res)

    if args.seeds > 1:
        summary = _print_multiseed_summary(all_results, args.grid, cfg_epochs)
        summary_path = OUT_DIR / f"summary_grid{args.grid}_{args.seeds}seeds.txt"
        summary_path.write_text(summary)
        print(f"  Multi-seed summary saved → {summary_path}")

    print(f"\n{'='*70}")
    print(f"  All runs complete.  Outputs in: {OUT_DIR}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
