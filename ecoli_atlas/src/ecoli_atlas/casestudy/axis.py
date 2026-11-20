"""Discriminant axes and paired tests shared by the case studies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def lopo_axis(E: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out discriminant score, oriented (positive class high) with training scores only."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te, tr = np.where(groups == g)[0], np.where(groups != g)[0]
        if len(np.unique(y[tr])) < 2:
            continue
        m = lda().fit(E[tr], y[tr])
        s_tr = m.decision_function(E[tr])
        flip = -1.0 if s_tr[y[tr] == 1].mean() < s_tr[y[tr] == 0].mean() else 1.0
        oof[te] = flip * m.decision_function(E[te])
    return oof


def center_by_group(E: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Remove each group's mean, leaving only the within-pair shift."""
    Ec = np.array(E, dtype=float, copy=True)
    for g in np.unique(groups):
        m = groups == g
        Ec[m] -= Ec[m].mean(0, keepdims=True)
    return Ec


def oriented_lda_score(E: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One discriminant fitted on all rows, oriented so the positive class scores high."""
    score = lda().fit(E, y).decision_function(E)
    if score[y == 1].mean() < score[y == 0].mean():
        score = -score
    return score


def paired_pivot(groups, y, score) -> pd.DataFrame:
    return pd.DataFrame({"g": groups, "c": y, "s": score}).pivot_table(index="g", columns="c", values="s")


def paired_axis_test(Ec: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int = 1000, seed: int = 77) -> dict:
    """Leave-one-group-out axis on group-centred data; sign test, Wilcoxon and a permutation null that
    flips the labels within random pairs (the design's own geometry under H0)."""
    from scipy.stats import binomtest, wilcoxon
    oof = lopo_axis(Ec, y, groups)
    piv = paired_pivot(groups, y, oof)
    correct = int((piv[1] > piv[0]).sum())
    n_pairs = int(len(piv))
    p_sign = float(binomtest(correct, n_pairs, 0.5, alternative="greater").pvalue)
    try:
        p_w = float(wilcoxon(piv[1], piv[0], alternative="greater").pvalue)
    except ValueError:
        p_w = float("nan")
    rng, null = np.random.RandomState(seed), []
    for _ in range(n_perm):
        y_perm = y.copy()
        for g in np.unique(groups):
            m = groups == g
            if rng.rand() < 0.5:
                y_perm[m] = 1 - y_perm[m]
        pv = paired_pivot(groups, y_perm, lopo_axis(Ec, y_perm, groups))
        null.append(int((pv[1] > pv[0]).sum()) if (1 in pv.columns and 0 in pv.columns) else 0)
    null = np.array(null)
    return {"axis_oof": oof, "n_pairs": n_pairs, "correct": correct, "p_sign": p_sign, "p_wilcoxon": p_w,
            "null_median": float(np.median(null)), "p_perm": float((null >= correct).mean()), "n_perm": int(n_perm)}


def perm_p_text(test: dict) -> str:
    """'p=0.012', or 'p<0.001' when no permutation reached the observed count."""
    return f"p<{1 / test['n_perm']:.3g}" if test["p_perm"] == 0 else f"p={test['p_perm']:.3g}"
