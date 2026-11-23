"""The score of each head (the log-odds of its call, which is also the head's one-dimensional block)
by class, for the training pool and TEST."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HEADS
from .plotting import savefig, tab20_pairs
from .split_balance import TITLES

SCORES_FILE = "head_scores.tsv"


def _auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    if positive.all() or not positive.any():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(positive, score))


def fig_head_scores(ctx, out_dir, formats) -> dict:
    """One panel per head: histograms of its score for each class, training pool (filled) and TEST
    (outlined); samples without a label are left out. Writes ``head_scores.tsv`` (every sample's
    score per head) and returns the class counts and the AUROC of the score per split."""
    import matplotlib.pyplot as plt
    heads = [h for h in HEADS if h in ctx.scores]
    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 4.5), squeeze=False)
    result, table = {}, pd.DataFrame(index=pd.Index(ctx.ids, name="sample"))
    for ax, h in zip(axes[0], heads):
        score, lab = np.asarray(ctx.scores[h], dtype=float), ctx.labels(h)
        table[f"{h}_score"] = score
        dark, light = tab20_pairs(HEADS[h])
        bins = np.linspace(np.nanpercentile(score, 0.5), np.nanpercentile(score, 99.5), 50)
        res = {"n_per_class": {}, "auroc": {}}
        for split, name in ((1, "train"), (3, "TEST")):
            for c in HEADS[h]:
                m = (lab == c) & (ctx.split == split)
                res["n_per_class"][f"{c}_{name}"] = int(m.sum())
                if not m.any():
                    continue
                if split == 1:
                    ax.hist(np.clip(score[m], bins[0], bins[-1]), bins=bins, density=True, color=light[c], alpha=0.8,
                            label=f"{c} (train, n={int(m.sum())})")
                else:
                    ax.hist(np.clip(score[m], bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=1.8,
                            color=dark[c], label=f"{c} (TEST, n={int(m.sum())})")
            keep = np.isin(lab, HEADS[h]) & (ctx.split == split)
            res["auroc"][name] = _auroc(score[keep], lab[keep] == HEADS[h][-1])
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"{TITLES[h]}", fontsize=11)
        ax.set_xlabel(f"score (log-odds of {HEADS[h][-1]})")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="best")
        result[h] = res
    fig.tight_layout()
    savefig(fig, out_dir, "fig_head_scores", formats)
    table.to_csv(f"{out_dir}/{SCORES_FILE}", sep="\t", float_format="%.5g")
    return result
