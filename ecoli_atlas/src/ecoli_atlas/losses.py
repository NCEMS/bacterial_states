"""Loss terms and class weighting used in training."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EFFECTIVE_NUM_BETA = 0.999
MIN_MINORITY_FRACTION = 0.40


def class_weights(lab_rows: np.ndarray, n_classes: int, beta: float = EFFECTIVE_NUM_BETA,
                  min_minority_frac: float = MIN_MINORITY_FRACTION) -> np.ndarray:
    """Effective-number-of-samples class weights (Cui et al.), with the minority class floored at
    ``min_minority_frac`` of the weighted signal for binary factors."""
    cls_counts = np.zeros(n_classes, dtype=np.float64)
    cls, cnt = np.unique(lab_rows[lab_rows >= 0], return_counts=True)
    cls_counts[cls] = cnt
    w = (1.0 - beta) / (1.0 - np.power(beta, np.clip(cls_counts, 1, None)))
    w = w / w.mean()
    if n_classes == 2 and cls_counts.min() > 0:
        lo, hi = int(cls_counts.argmin()), int(cls_counts.argmax())
        if (cls_counts[lo] * w[lo]) / (cls_counts @ w) < min_minority_frac:
            t = min_minority_frac
            w[lo] = (t * cls_counts[hi] * w[hi]) / ((1 - t) * cls_counts[lo])
    return w


def cos_dist(z: torch.Tensor) -> torch.Tensor:
    zc = F.normalize(z, dim=1)
    return 1.0 - zc @ zc.t()


def recon_loss(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE normalised by the batch variance of the input."""
    return ((xhat - x) ** 2).mean() / x.var(unbiased=False).detach().clamp_min(1e-8)


def within_study_push(d: torch.Tensor, gse: torch.Tensor, cid: torch.Tensor, annot: torch.Tensor,
                      margin: float) -> torch.Tensor:
    """Hinge apart same-study pairs known to differ in condition. Pairs where neither sample is
    annotated are excluded, since their relationship is unknown."""
    same_g = (gse.unsqueeze(0) == gse.unsqueeze(1)).clone()
    same_g.fill_diagonal_(False)
    ai, aj = annot.unsqueeze(0), annot.unsqueeze(1)
    same_id = cid.unsqueeze(0) == cid.unsqueeze(1)
    mask = same_g & (((ai & aj) & ~same_id) | (ai ^ aj))
    return F.relu(margin - d[mask]).mean() if mask.any() else d.new_tensor(0.0)


def cross_study_control_pull(d: torch.Tensor, gse: torch.Tensor, ccond: torch.Tensor,
                             is_ctrl: torch.Tensor) -> torch.Tensor:
    """Draw matched wild-type controls from different studies together."""
    if int(is_ctrl.sum()) < 2:
        return d.new_tensor(0.0)
    mask = ((gse.unsqueeze(0) != gse.unsqueeze(1))
            & (is_ctrl.unsqueeze(0) & is_ctrl.unsqueeze(1))
            & (ccond.unsqueeze(0) == ccond.unsqueeze(1)))
    return d[mask].mean() if mask.any() else d.new_tensor(0.0)
