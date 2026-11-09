"""Class balance across CV folds and TEST (notebook cell 14)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..splits import bucket_labels
from .plotting import savefig

FIG_VARS = ["growth_phase", "medium_type", "is_lab_strain"]


def bucket_order(n_folds: int) -> list:
    return [f"fold{k}" for k in range(1, n_folds + 1)] + ["test"]


def _fmt(c) -> str:
    return {True: "lab strain", False: "non-lab", "True": "lab strain", "False": "non-lab",
            "Unlabeled": "Unlabeled"}.get(c, str(c))


def fig_split_balance(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns
    meta = ctx.meta
    order = bucket_order(ctx.cfg.n_folds)
    labels = bucket_labels(meta)
    fig_vars = [v for v in FIG_VARS if v + "_c" in meta.columns]
    fig, axes = plt.subplots(1, len(fig_vars), figsize=(7 * len(fig_vars), 4.5))
    axes = np.atleast_1d(axes)
    result = {}
    for ax, v in zip(axes, fig_vars):
        tab = pd.crosstab(labels, meta[v + "_c"].astype("object").fillna("Unlabeled"), normalize="index")
        tab = tab.reindex([b for b in order if b in tab.index])
        classes = [c for c in tab.columns if c != "Unlabeled"]
        color_map = dict(zip(classes, sns.color_palette("Set2", len(classes))))
        color_map["Unlabeled"] = (0.7, 0.7, 0.7)
        bottom = np.zeros(len(tab))
        for cls in tab.columns:
            ax.bar(tab.index, tab[cls].values, bottom=bottom, label=_fmt(cls), color=color_map[cls])
            bottom += tab[cls].values
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("proportion")
        ax.set_title(v)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=20)
        n_per = labels.value_counts().reindex(tab.index)
        for i, nm in enumerate(tab.index):
            ax.text(i, 1.01, f"n={int(n_per[nm])}", ha="center", fontsize=9)
        result[v] = {"n_per_bucket": {str(k): int(x) for k, x in n_per.items()},
                     "proportions": {str(b): {str(c): float(tab.loc[b, c]) for c in tab.columns} for b in tab.index}}
    fig.tight_layout()
    savefig(fig, out_dir, "fig_split_balance", formats)
    return result
