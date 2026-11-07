"""
Shared device helpers for the synthetic security-game experiment.

best_device() prefers CUDA, then Apple MPS, then CPU. The device only decides
where the CNN runs; the Stackelberg LPs (scipy HiGHS and cvxpylayers) always
run on the CPU.
"""

import os

import torch


def best_device() -> torch.device:
    """
    Return the compute device to train on.

    Priority: CUDA GPU > Apple Silicon MPS > CPU, unless the environment
    variable ``SDFL_DEVICE`` is set (e.g. ``SDFL_DEVICE=cpu``), in which case
    that device is used verbatim. The paper's synthetic run used MPS; the
    real-data runs used CUDA and CPU. Under some PyTorch builds training on
    MPS stalls, so check the loss curve or set ``SDFL_DEVICE=cpu``.
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
    Call once at the top of each main entry point.
    """
    device = best_device()
    if device.type == "cuda":
        name  = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        extra = f"  ({count} GPU(s) found — using: {name})"
    elif device.type == "mps":
        extra = "  (Apple Silicon MPS)"
    else:
        extra = "  (no GPU found)"
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Device: {device}{extra}")
    return device
