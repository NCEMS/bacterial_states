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


def spearman_vec(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of every column of X with y (NaN for a constant column)."""
    from scipy.stats import rankdata
    rx = rankdata(np.asarray(X, dtype=float), axis=0)
    ry = rankdata(np.asarray(y, dtype=float))
    rx = rx - rx.mean(0, keepdims=True)
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry[:, None]).sum(0) / denom
    return np.where(denom > 0, r, np.nan)


def boot_ci(values: np.ndarray, stat=np.mean, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boots = np.array([stat(values[rng.randint(0, len(values), len(values))]) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Bootstrap of mean(a) - mean(b) over paired observations: (difference, lo, hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    n = len(a)
    bs = np.array([(a[i] - b[i]).mean() for i in (rng.randint(0, n, n) for _ in range(n_boot))])
    return float((a - b).mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
