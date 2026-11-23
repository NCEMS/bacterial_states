"""The calls on the TEST studies: one confusion matrix per head, and the table of samples whose
growth-phase call disagrees with the label."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HEADS
from ..train import head_metrics
from .plotting import savefig
from .split_balance import TITLES

THRESHOLD = 0.5
MISCLASS_COLS = ("GSE", "growth_phase", "growth_phase_basis", "medium_class", "base_medium", "strain_name", "carbon_source",
                 "treated", "oxygen", "growth_temperature")


def test_metrics(ctx) -> dict:
    """As the manifest's ``test_metrics``: every head on the labelled TEST samples at 0.5."""
    return head_metrics(ctx.data.y_eval, ctx.probs, ctx.data.te_rows)


def fig_head_test_eval(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    metrics = test_metrics(ctx)
    heads = [h for h in HEADS if h in metrics]
    fig, axes = plt.subplots(1, max(len(heads), 1), figsize=(6 * max(len(heads), 1), 5))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, heads):
        m = metrics[h]
        cm = np.asarray(m["confusion"])          # rows true, columns predicted, [negative, positive]
        rs = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, rs, out=np.zeros_like(cm, dtype=float), where=rs != 0)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(HEADS[h], fontsize=11)
        ax.set_yticklabels(HEADS[h], fontsize=11)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{TITLES[h]}\nbalanced acc={m['balanced_acc']:.2f}, AUROC={m['auroc']:.2f}", fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=14)
    for ax in axes[len(heads):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_head_test_eval", formats)
    return metrics


def misclassified_table(ctx, out_dir, head: str = "phase") -> dict:
    """Samples of the training pool or TEST whose call at 0.5 disagrees with the label. The training
    pool rows are calls on samples the model was fitted on."""
    neg, pos = HEADS[head]
    y = ctx.data.y[head]
    rows = np.where(np.isin(ctx.split, (1, 3)) & (y >= 0))[0]
    p = ctx.probs[head][rows]
    true = np.where(y[rows] == 1, pos, neg)
    pred = np.where(p >= THRESHOLD, pos, neg)
    wrong = true != pred
    cols = [c for c in MISCLASS_COLS if c in ctx.meta.columns]
    tab = ctx.meta.iloc[rows[wrong]][cols].copy()
    tab.insert(0, "set", np.where(ctx.split[rows[wrong]] == 3, "TEST", "train"))
    tab.insert(1, "true", true[wrong])
    tab.insert(2, "pred", pred[wrong])
    tab.insert(3, f"p_{pos}", np.round(p[wrong], 3))
    tab.insert(4, "scored", ctx.data.y_eval[head][rows[wrong]] >= 0)
    tab = tab.sort_values(["set", "true", f"p_{pos}"], ascending=[True, True, False])
    tab.index.name = "sample"
    tab.to_csv(f"{out_dir}/{head}_misclassified.tsv", sep="\t")
    ct = pd.crosstab(tab["set"], tab["true"])
    return {"threshold": THRESHOLD, "n_evaluated": int(len(rows)), "n_misclassified": int(wrong.sum()),
            "by_set": {s: {str(c): int(ct.loc[s, c]) for c in ct.columns} for s in ct.index}}
