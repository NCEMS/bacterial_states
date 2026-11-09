"""UMAP of each raw factor block, coloured by label and by train / TEST (notebook cell 27)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..compare import umap_fit
from .plotting import savefig, tab20_pairs


def fig_factor_umaps(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    meta = ctx.meta
    factors = ctx.factors
    coords, pal_dark, pal_light, result = {}, {}, {}, {}
    fig, axes = plt.subplots(1, len(factors), figsize=(6 * len(factors), 5.5))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, factors):
        xy = umap_fit(ctx.res.blocks[f], ctx.seed).embedding_
        coords[f] = xy
        cats = ctx.classes(f)
        pal_dark[f], pal_light[f] = tab20_pairs(cats)
        lab = ctx.labels_str(f).values
        is_un = pd.isna(lab)
        if is_un.any():
            ax.scatter(xy[is_un, 0], xy[is_un, 1], s=4, c="#d9d9d9", label=f"unlabelled (n={int(is_un.sum())})", zorder=1)
        counts = {}
        for c in cats:
            m = lab == c
            counts[c] = int(m.sum())
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=pal_dark[f][c], label=f"{c} (n={int(m.sum())})", zorder=2)
        ax.set_title(f, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=2, loc="best")
        result[f] = {"n_unlabelled": int(is_un.sum()), "n_per_class": counts, "block_dim": int(ctx.res.blocks[f].shape[1])}
    fig.tight_layout()
    savefig(fig, out_dir, "fig_factor_umaps", formats)

    te_mask = (meta["split"] == 3).values
    tr_mask = (meta["split"] == 1).values
    fig, axes = plt.subplots(1, len(factors), figsize=(6 * len(factors), 5.5))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, factors):
        xy = coords[f]
        lab = ctx.labels_str(f).values
        for c in ctx.classes(f):
            m = (lab == c) & tr_mask
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=6, color=pal_light[f][c], label=f"{c} (train, n={int(m.sum())})", zorder=2)
        for c in ctx.classes(f):
            m = (lab == c) & te_mask
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=34, color=pal_dark[f][c], edgecolor="k", linewidth=0.4,
                           label=f"{c} (TEST, n={int(m.sum())})", zorder=3)
        ax.set_title(f"{f}: train / TEST", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=1.4, loc="best", ncol=2)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_factor_umaps_overlay", formats)

    tab = pd.DataFrame(index=meta.index)
    for f in factors:
        tab[f"{f}_umap1"], tab[f"{f}_umap2"] = coords[f][:, 0], coords[f][:, 1]
    tab["split"] = meta["split"].values
    tab.to_csv(f"{out_dir}/umap_factor_coords.csv")
    return result
