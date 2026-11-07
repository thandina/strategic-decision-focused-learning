"""
Device selection shared by every model in this package.

Used by models/pto.py, models/zoc.py, models/kkt.py and models/hybrid.py
(conservation game, Table 1 of the paper). The device changes speed and the
floating-point kernels, not the algorithm; training is deterministic per seed
only on the same device class.
"""

import os

import torch


def best_device() -> torch.device:
    """
    Return the compute device to train on.

    Priority: CUDA GPU > Apple Silicon MPS > CPU, unless the environment
    variable ``SDFL_DEVICE`` is set (e.g. ``SDFL_DEVICE=cpu``), in which case
    that device is used verbatim. The paper's runs used CUDA (conservation
    game) and CPU (infrastructure game); MPS was not validated and under some
    PyTorch builds training on it stalls, so on a Mac set ``SDFL_DEVICE=cpu``.
    """
    forced = os.environ.get("SDFL_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device(prefix: str = "") -> torch.device:
    """
    Return best_device() and print a one-line banner describing it.
    Side effect: one line on stdout. Not called by train.py; kept as a helper.
    """
    device = best_device()
    if device.type == "cuda":
        name  = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        extra = f"  ({count} GPU(s) found; using: {name})"
    elif device.type == "mps":
        extra = "  (Apple Silicon MPS)"
    else:
        extra = "  (no GPU found)"
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Device: {device}{extra}")
    return device
