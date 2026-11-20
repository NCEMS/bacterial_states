"""Seeds, devices, small numeric helpers, logging."""
from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger("ecoli_atlas")


def setup_logging(verbose: int = 0) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(level if verbose else logging.INFO)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) and request deterministic kernels."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str = "auto"):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return torch.device(name)


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)
