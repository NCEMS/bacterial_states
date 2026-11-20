"""Evaluation metrics: classification on held-out studies, study mixing and neighbour agreement."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .utils import l2norm


def classification_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Balanced accuracy with bootstrap CI, AUROC and per-class recall at a fixed threshold."""
    y_pred = (p_pos >= threshold).astype(int)
    ba = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, p_pos) if len(np.unique(y_true)) > 1 else float("nan")
    rng = np.random.RandomState(seed)
    n = len(y_true)
    bas = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(np.unique(y_true[i])) > 1:
            bas.append(balanced_accuracy_score(y_true[i], y_pred[i]))
    lo, hi = (np.percentile(bas, [2.5, 97.5]) if bas else (float("nan"), float("nan")))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    recall = [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(2)]
    return {"n": int(n), "threshold": float(threshold), "balanced_acc": float(ba),
            "ba_ci": [float(lo), float(hi)], "auroc": float(auc), "recall": recall,
            "confusion": cm.tolist()}


def same_study_rate(E: np.ndarray, gse: np.ndarray, cond_id: np.ndarray, annot: np.ndarray, k: int = 25) -> np.ndarray:
    """Fraction of a sample's k nearest neighbours (cosine) that come from its own study, counting
    only annotated neighbours that are not same-study replicates. Lower = better study mixing."""
    E = l2norm(np.asarray(E, float))
    idx = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E).kneighbors(E, return_distance=False)[:, 1:]
    same_gse = gse[idx] == gse[:, None]
    consider = (annot[idx] & annot[:, None]) & ~(same_gse & (cond_id[idx] == cond_id[:, None]))
    num, den = (same_gse & consider).sum(1), consider.sum(1)
    return (num / np.maximum(den, 1))[(den > 0) & annot]


def knn_label_agreement(E_fit: np.ndarray, y_fit: np.ndarray, gse_fit: np.ndarray, E_val: np.ndarray,
                        y_val: np.ndarray, gse_val: np.ndarray, k: int) -> float:
    """Class-balanced fraction of cross-study nearest neighbours sharing the sample's label."""
    kk = int(min(k, len(E_fit)))
    sims = l2norm(E_val) @ l2norm(E_fit).T
    sims[gse_val[:, None] == gse_fit[None, :]] = -np.inf
    nbr = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    same = y_fit[nbr] == y_val[:, None]
    return float(np.mean([same[y_val == c].mean() for c in np.unique(y_val)]))


def within_study_order(score: np.ndarray, level: np.ndarray, gse: np.ndarray, rows: np.ndarray) -> dict:
    """Among pairs of samples of one study at different levels of an ordered scale (``level`` >= 0),
    the share whose scores are in the order of the levels (ties count half). Reported as the mean
    over studies, so that one large study does not decide it, and over all pairs."""
    rows = np.asarray(rows)
    rows = rows[level[rows] >= 0]
    per_study, total, right = [], 0, 0.0
    for g in np.unique(gse[rows]):
        r = rows[gse[rows] == g]
        i, j = np.where(level[r][:, None] < level[r][None, :])
        if not len(i):
            continue
        ok = (score[r][i] < score[r][j]) + 0.5 * (score[r][i] == score[r][j])
        per_study.append(float(ok.mean()))
        total, right = total + len(i), right + float(ok.sum())
    return {"n_studies": len(per_study), "n_pairs": int(total),
            "share_in_order_mean_over_studies": float(np.mean(per_study)) if per_study else None,
            "share_in_order": right / total if total else None}
