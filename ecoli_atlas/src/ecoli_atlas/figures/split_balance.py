"""Class balance across the CV folds and TEST: the composition of the training and test data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HEADS
from ..splits import bucket_labels
from .plotting import GREY, UNLABELLED, savefig

TITLES = {"phase": "growth phase", "medium": "medium type", "lab_strain": "lab strain"}


def bucket_order(splits: pd.DataFrame) -> list:
    folds = sorted(int(k) for k in splits.loc[splits["split"] == 1, "cv_fold"].dropna().unique())
    return [f"fold{k}" for k in folds] + ["test"]


def balance_table(labels: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Long table: label, unit (samples / studies), bucket, class, n, proportion. A study counts
    for a class when it has at least one sample of it, and as unlabelled when it has none."""
    bucket = bucket_labels(splits)
    rows = []
    for head, classes in HEADS.items():
        lab = labels[head].astype("object").where(labels[head].notna(), UNLABELLED)
        frame = pd.DataFrame({"bucket": bucket, "cls": lab, "gse": splits["gse"]})
        frame = frame[frame["bucket"].isin(bucket_order(splits))]
        any_label = frame[frame["cls"] != UNLABELLED].groupby("gse").size()
        for b, sub in frame.groupby("bucket"):
            counts = {"samples": sub["cls"].value_counts(),
                      "studies": pd.concat([sub[sub["cls"] != UNLABELLED].drop_duplicates(["gse", "cls"])["cls"],
                                            pd.Series(UNLABELLED, index=[g for g in sub["gse"].unique() if g not in any_label.index])
                                            ]).value_counts()}
            for unit, vc in counts.items():
                total = len(sub) if unit == "samples" else sub["gse"].nunique()
                for cls in list(classes) + [UNLABELLED]:
                    n = int(vc.get(cls, 0))
                    rows.append({"label": head, "unit": unit, "bucket": b, "class": cls, "n": n, "total": total,
                                 "proportion": n / total if total else 0.0})
    return pd.DataFrame(rows)


def fig_split_balance(labels: pd.DataFrame, splits: pd.DataFrame, out_dir, formats) -> pd.DataFrame:
    """Top row: share of samples per class in each fold and TEST. Bottom row: share of studies
    with samples of the class (a study with both classes counts for both, so bars can pass 1)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    table = balance_table(labels, splits)
    order = bucket_order(splits)
    heads = list(HEADS)
    fig, axes = plt.subplots(2, len(heads), figsize=(7 * len(heads), 9), squeeze=False)
    for j, head in enumerate(heads):
        classes = list(HEADS[head]) + [UNLABELLED]
        colors = dict(zip(HEADS[head], sns.color_palette("Set2", len(HEADS[head])))) | {UNLABELLED: GREY}
        for i, unit in enumerate(("samples", "studies")):
            ax = axes[i, j]
            sub = table[(table["label"] == head) & (table["unit"] == unit)]
            bottom = np.zeros(len(order))
            for cls in classes:
                vals = sub[sub["class"] == cls].set_index("bucket")["proportion"].reindex(order).fillna(0).to_numpy()
                ax.bar(order, vals, bottom=bottom, label=cls, color=colors[cls])
                bottom += vals
            totals = sub.drop_duplicates("bucket").set_index("bucket")["total"].reindex(order)
            for k, b in enumerate(order):
                ax.text(k, bottom[k] + 0.01, f"n={int(totals[b])}", ha="center", fontsize=9)
            ax.set_ylim(0, max(1.08, bottom.max() + 0.1))
            ax.set_ylabel("proportion of " + unit)
            ax.set_title(TITLES[head] if i == 0 else "")
            ax.tick_params(axis="x", rotation=20)
            if i == 0:
                ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_split_balance", formats)
    return table
