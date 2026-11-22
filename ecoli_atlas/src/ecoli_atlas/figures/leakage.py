"""Study leakage on TEST: the share of a sample's nearest neighbours that come from its own study,
for the z-scored expression the model is given and for the embedding it makes. The head blocks
alone are in the numbers, not in the figure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HEADS
from ..metrics import same_study_rate
from ..utils import boot_ci, paired_boot_ci
from .plotting import savefig

K = 25
ORDER = ("input", "blocks", "embedding")
PLOTTED = {"input": ("z-scored expression", "0.6"), "embedding": ("atlas embedding", "#2ca02c")}


def leakage_stats(ctx, k: int = K) -> dict:
    pool = ctx.data.te_rows
    if len(pool) < 3:
        raise ValueError("fewer than 3 TEST samples; the leakage figure needs a TEST set")
    kk = int(min(k, len(pool) - 1))
    gse = ctx.data.gse_codes[pool]
    cid, ann = ctx.data.identity["cond_id"][pool], ctx.data.identity["is_annot"][pool]
    sizes = pd.Series(gse).value_counts()
    chance = float(np.mean([(sizes[g] - 1) / max(len(pool) - 1, 1) for g in gse]))
    cands = {"input": ctx.data.X.numpy()[pool], "embedding": ctx.z[pool]}
    head_blocks = [ctx.blocks[h][pool] for h in HEADS if h in ctx.blocks]
    if head_blocks:
        cands["blocks"] = np.concatenate(head_blocks, axis=1)
    data = {n: same_study_rate(E, gse, cid, ann, kk) for n, E in cands.items()}
    stats = {}
    for n in [n for n in ORDER if n in data]:
        lo, hi = boot_ci(data[n], n_boot=2000, seed=0)
        stats[n] = {"mean": float(np.mean(data[n])), "ci": [lo, hi], "n": int(len(data[n]))}
    paired = {}
    for a, b in (("embedding", "blocks"), ("embedding", "input"), ("blocks", "input")):
        if a in data and b in data:
            d, lo, hi = paired_boot_ci(data[a], data[b])
            paired[f"{a}-{b}"] = {"diff": d, "ci": [lo, hi], "includes_zero": bool(lo <= 0 <= hi)}
    return {"n_test": int(len(pool)), "n_studies": int(len(sizes)), "k": kk, "chance": chance, "rates": stats, "paired": paired}


def fig_study_leakage(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    st = leakage_stats(ctx)
    stats, chance = st["rates"], st["chance"]
    plot = [n for n in PLOTTED if n in stats]
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(plot))
    means = [stats[n]["mean"] for n in plot]
    lo = [stats[n]["mean"] - stats[n]["ci"][0] for n in plot]
    hi = [stats[n]["ci"][1] - stats[n]["mean"] for n in plot]
    ax.bar(x, means, color=[PLOTTED[n][1] for n in plot], alpha=0.85, width=0.6, edgecolor="k", linewidth=1.0,
           yerr=[lo, hi], capsize=5, ecolor="k")
    for i, n in enumerate(plot):
        ax.text(i, stats[n]["ci"][1] + 0.006, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(chance, color="k", ls=":", lw=1.4)
    ax.text(0.5, chance, f"chance = {chance:.3f}", va="bottom", ha="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([PLOTTED[n][0] for n in plot], fontsize=11)
    ax.set_ylabel(f"same-study share of {st['k']} neighbours", fontsize=12)
    ax.set_title(f"TEST: {st['n_test']} samples of {st['n_studies']} studies", fontsize=11)
    ax.set_ylim(0, max(max(m + h for m, h in zip(means, hi)), chance) + 0.06)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_study_leakage", formats)
    return st
