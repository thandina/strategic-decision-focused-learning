#!/usr/bin/env bash
# setup.sh -- create a virtual environment with everything needed to run the
# experiments in this repository.
#
#   bash setup.sh            # creates ./.venv with CPU PyTorch
#   source .venv/bin/activate
#
# Requirements: Python 3.10 or newer on PATH as `python3` (3.11 was used for
# testing). On a CUDA machine install the matching PyTorch wheel afterwards,
# see https://pytorch.org/get-started/locally/ ; everything else is the same.
#
# `cvxpy` and `cvxpylayers` are only needed by the KKT and Hybrid methods.
# If they are missing, those two methods fall back with a warning (see
# conservation_game/models/kkt.py and infrastructure_game/models/kkt.py).

set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "=== Environment ready ==="
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "import cvxpy, cvxpylayers; print('cvxpy', cvxpy.__version__, '| cvxpylayers', cvxpylayers.__version__)"
echo "Activate with:  source .venv/bin/activate"
