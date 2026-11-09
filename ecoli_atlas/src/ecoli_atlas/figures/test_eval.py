"""Held-out evaluation of the factor encoders (notebook cell 29) and the growth-phase
misclassification table (cell 35)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import classification_metrics
from .plotting import savefig

MISCLASS_COLS = ["gse", "growth_phase", "medium_type_c", "base_medium", "strain_background", "carbon_source",
                 "is_stressed", "stress", "OD600", "temperature(C)", "oxygen"]


def test_metrics(ctx) -> dict:
    """TEST metrics per factor at the saved operating threshold (same bootstrap as the notebook)."""
    out = {}
    te_rows = ctx.data.te_rows
    for f in ctx.factors:
        y = ctx.codes(f)
        te = te_rows[y[te_rows] >= 0]
        if len(te) == 0 or len(np.unique(y[te])) < 2:
            continue
        pos = ctx.positive_index(f)
        p_pos = ctx.res.proba[f][te, pos]
        thr = ctx.bundle.factors[f].threshold
        thr = 0.5 if thr is None else float(thr)
        m = classification_metrics((y[te] == pos).astype(int), p_pos, thr)
        m["classes"] = ctx.classes(f)
        m["positive_class"] = ctx.classes(f)[pos]
        out[f] = m
    return out


def fig_factor_test_eval(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    metrics = test_metrics(ctx)
    factors = [f for f in ctx.factors if f in metrics]
    fig, axes = plt.subplots(1, max(len(factors), 1), figsize=(6 * max(len(factors), 1), 5))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, factors):
        m = metrics[f]
        cats = m["classes"]
        cm = np.asarray(m["confusion"])          # rows / cols ordered [negative, positive]
        pos = ctx.positive_index(f)
        if pos == 0:                              # display in class order
            cm = cm[::-1, ::-1]
        rs = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, rs, out=np.zeros_like(cm, dtype=float), where=rs != 0)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cats)))
        ax.set_yticks(range(len(cats)))
        ax.set_xticklabels([str(c) for c in cats])
        ax.set_yticklabels([str(c) for c in cats])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{f}\nbalanced acc={m['balanced_acc']:.2f}, AUROC={m['auroc']:.2f}", fontsize=10)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=14)
    for ax in axes[len(factors):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_factor_test_eval", formats)
    return metrics


def growth_phase_misclassified_table(ctx, out_dir, factor: str = "growth_phase") -> dict:
    """Samples in the pool or TEST whose thresholded growth-phase call disagrees with the label."""
    if factor not in ctx.factors:
        return {"skipped": f"{factor} not in the bundle"}
    meta = ctx.meta
    cats = ctx.classes(factor)
    pos = ctx.positive_index(factor)
    y = ctx.codes(factor)
    thr = ctx.bundle.factors[factor].threshold
    thr = 0.5 if thr is None else float(thr)
    rows = np.where(meta["split"].isin([1, 3]).values & (y >= 0))[0]
    p = ctx.res.proba[factor][rows, pos]
    true = np.where(y[rows] == pos, cats[pos], cats[1 - pos])
    pred = np.where(p >= thr, cats[pos], cats[1 - pos])
    wrong = true != pred
    cols = [c for c in MISCLASS_COLS if c in meta.columns]
    fp = meta.iloc[rows[wrong]][cols].copy()
    fp.insert(0, "set", np.where(meta["split"].values[rows[wrong]] == 3, "TEST", "train"))
    fp.insert(1, "true", true[wrong])
    fp.insert(2, "pred", pred[wrong])
    fp.insert(3, f"P_{cats[pos][:4]}", np.round(p[wrong], 3))
    fp = fp.sort_values(["set", "true", fp.columns[3]], ascending=[True, True, False])
    fp.index.name = "sample"
    fp.to_csv(f"{out_dir}/{factor}_misclassified.tsv", sep="\t")
    ct = pd.crosstab(fp["set"], fp["true"])
    return {"threshold": thr, "n_evaluated": int(len(rows)), "n_misclassified": int(wrong.sum()),
            "by_set": {s: {str(c): int(ct.loc[s, c]) for c in ct.columns} for s in ct.index}}
