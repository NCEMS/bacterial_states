"""UMAP of each head's block of the embedding, coloured by label and by training pool / TEST."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HEADS
from .plotting import savefig, tab20_pairs
from .split_balance import TITLES

COORDS_FILE = "umap_factor_coords.tsv"


def umap_fit(E: np.ndarray, seed: int = 77, n_neighbors: int = 30, min_dist: float = 0.3, metric: str = "cosine"):
    """UMAP with the settings of the first atlas (30 neighbours, min_dist 0.3, cosine); the number
    of neighbours is lowered for very small inputs. Returns the coordinates."""
    try:
        import umap
    except ImportError as e:  # pragma: no cover
        raise SystemExit("UMAP needs umap-learn: pip install 'ecoli-atlas[umap]'") from e
    n = int(np.asarray(E).shape[0])
    return umap.UMAP(n_neighbors=max(2, min(n_neighbors, n - 1)), min_dist=min_dist, metric=metric, random_state=seed).fit_transform(E)


def fig_factor_umaps(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    heads = [h for h in HEADS if h in ctx.blocks]
    if not heads:
        return {"skipped": "the embedding has no block per head (unstructured fit)"}
    coords, dark, light, result = {}, {}, {}, {}
    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 5.5))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, heads):
        xy = coords[h] = umap_fit(ctx.blocks[h], ctx.seed)
        dark[h], light[h] = tab20_pairs(HEADS[h])
        lab = ctx.labels(h)
        un = pd.isna(lab)
        if un.any():
            ax.scatter(xy[un, 0], xy[un, 1], s=4, c="#d9d9d9", label=f"unlabelled (n={int(un.sum())})", zorder=1)
        counts = {}
        for c in HEADS[h]:
            m = lab == c
            counts[c] = int(m.sum())
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=dark[h][c], label=f"{c} (n={counts[c]})", zorder=2)
        ax.set_title(f"{TITLES[h]} block ({ctx.blocks[h].shape[1]} dimensions)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=2, loc="best")
        result[h] = {"n_unlabelled": int(un.sum()), "n_per_class": counts, "block_dim": int(ctx.blocks[h].shape[1])}
    fig.tight_layout()
    savefig(fig, out_dir, "fig_factor_umaps", formats)

    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 5.5))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, heads):
        xy, lab = coords[h], ctx.labels(h)
        for c in HEADS[h]:
            m = (lab == c) & (ctx.split == 1)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=6, color=light[h][c], label=f"{c} (train, n={int(m.sum())})", zorder=2)
        for c in HEADS[h]:
            m = (lab == c) & (ctx.split == 3)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=34, color=dark[h][c], edgecolor="k", linewidth=0.4,
                           label=f"{c} (TEST, n={int(m.sum())})", zorder=3)
        ax.set_title(f"{TITLES[h]}: train / TEST", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=1.4, loc="best", ncol=2)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_factor_umaps_overlay", formats)

    tab = pd.DataFrame(index=pd.Index(ctx.ids, name="sample"))
    for h in heads:
        tab[f"{h}_umap1"], tab[f"{h}_umap2"] = coords[h][:, 0], coords[h][:, 1]
    tab["split"] = ctx.split
    tab.to_csv(f"{out_dir}/{COORDS_FILE}", sep="\t", float_format="%.5g")
    return result
