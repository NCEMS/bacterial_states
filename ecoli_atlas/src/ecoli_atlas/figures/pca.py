"""Batch structure in the expression data itself, before any model: PCA of the log2 normalised
expression, and the genes that differ most between studies of one strain in one condition."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .plotting import savefig

N_TOP_BATCH = 10
STRAINS = {"MG1655": r"mg[\s_-]?1655", "BW25113": r"bw[\s_-]?25113"}
COORDS_FILE = "pca_log_expression_coords.tsv"


def matched_subsets(ctx) -> dict:
    """MG1655 and BW25113 samples that are exponential, untreated wild type and in a rich medium;
    ``_rich`` is every rich-medium sample."""
    strain = pd.Series(ctx.column("strain_name")).str.lower()
    rich = ctx.labels("medium") == "rich"
    base = (ctx.labels("phase") == "exponential") & rich & (ctx.column("is_wild_type") == "true") & (ctx.column("treated") == "false")
    out = {name: strain.map(lambda s: bool(re.search(pat, s))).to_numpy() & base for name, pat in STRAINS.items()}
    out["_rich"] = rich
    return out


def fit_pca(M: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(M), pca.explained_variance_ratio_ * 100


def _scatter(ax, xy, ev, values, mask, legend_title, title, topn=None, s=20):
    import matplotlib.pyplot as plt
    if not mask.any():
        ax.axis("off")
        return
    vals = np.asarray(values, dtype=object).copy()
    vals[~mask] = "other"
    vals[mask & (vals == "")] = "not stated"
    if topn is not None:
        keep = pd.Series(vals[mask]).value_counts().head(topn).index.tolist()
        vals[mask & ~np.isin(vals, keep)] = "other"
    levels = sorted([c for c in pd.unique(vals) if c != "other"], key=str)
    cmap, ncmax = (plt.cm.tab20, 20) if len(levels) > 10 else (plt.cm.tab10, 10)
    bg = vals == "other"
    if bg.any():
        ax.scatter(xy[bg, 0], xy[bg, 1], s=s * 0.6, alpha=0.15, color="lightgrey", label="other", zorder=1)
    for j, lev in enumerate(levels):
        sub = vals == lev
        ax.scatter(xy[sub, 0], xy[sub, 1], s=s, alpha=0.85, color=cmap(j % ncmax), label=f"{lev} (n={int(sub.sum())})", zorder=2)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.15)
    h, l = ax.get_legend_handles_labels()
    order = [i for i, x in enumerate(l) if x != "other"] + [i for i, x in enumerate(l) if x == "other"]
    ax.legend([h[i] for i in order], [l[i] for i in order], title=legend_title, fontsize=7, title_fontsize=8, framealpha=0.6,
              markerscale=1.3, ncol=2 if len(levels) > 10 else 1, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)


def fig_pca_batch(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    M = ctx.logx
    xy, ev = fit_pca(M, ctx.seed)
    subsets = matched_subsets(ctx)
    result = {"n_samples": int(M.shape[0]), "n_features": int(M.shape[1]),
              "explained_variance_pct": [float(ev[0]), float(ev[1])], "subsets": {}}
    for name, fn in (("MG1655", "fig_pca_mg1655_batch"), ("BW25113", "fig_pca_bw_batch")):
        mask = subsets[name]
        result["subsets"][name] = {"n_samples": int(mask.sum()), "n_studies": int(len(np.unique(ctx.gse[mask])))}
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        _scatter(ax, xy, ev, ctx.gse, mask, "study", f"{name}, exponential, untreated wild type, rich medium, by study", topn=10)
        fig.tight_layout()
        savefig(fig, out_dir, fn, formats)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _scatter(ax, xy, ev, ctx.column("strain_name"), subsets["_rich"], "strain", "Rich-medium samples by strain", topn=8, s=10)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_pca_strain_all_complex", formats)
    result["subsets"]["rich_medium"] = {"n_samples": int(subsets["_rich"].sum())}
    pd.DataFrame(xy, index=pd.Index(ctx.ids, name="sample"), columns=["PC1", "PC2"]).to_csv(f"{out_dir}/{COORDS_FILE}", sep="\t",
                                                                                           float_format="%.5g")
    return result


def top_batch_genes(M: np.ndarray, genes, gse: np.ndarray, mask: np.ndarray, n_top: int = N_TOP_BATCH, min_mean: float = 1.0):
    """Between-study SD of the per-study mean within a matched subset (replicate noise averaged out)."""
    rows = np.where(mask)[0]
    g = gse[rows]
    studies = pd.unique(g)
    if len(rows) < 3 or len(studies) < 2:
        return None, len(rows), len(studies)
    sub = M[rows]
    per_study_mean = np.vstack([sub[g == s].mean(0) for s in studies])
    keep = sub.mean(0) >= min_mean
    score = pd.Series(np.where(keep, per_study_mean.std(0), 0.0), index=list(genes)).sort_values(ascending=False)
    return score.head(n_top), len(rows), len(studies)


def fig_batch_genes(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    subsets = matched_subsets(ctx)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    tops, rows_out, result = {}, [], {}
    for ax, (name, col) in zip(axes, (("MG1655", "#1f77b4"), ("BW25113", "#d62728"))):
        g, n_s, n_st = top_batch_genes(ctx.logx, ctx.gene_names, ctx.gse, subsets[name])
        tops[name] = g
        result[name] = {"n_samples": int(n_s), "n_studies": int(n_st),
                        "top_genes": None if g is None else {k: float(v) for k, v in g.items()}}
        if g is None:
            ax.axis("off")
            continue
        rows_out += [{"subset": name, "gene": gene, "between_study_sd": float(v)} for gene, v in g.items()]
        y = np.arange(len(g))
        ax.barh(y, g.values[::-1], color=col, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(g.index[::-1], fontsize=8)
        ax.set_xlabel("between-study SD of mean log2 expression")
        ax.set_title(f"{name} (n={n_s} samples, {n_st} studies)", fontsize=10)
        ax.grid(alpha=0.15, axis="x")
    fig.suptitle("Most study-variable genes within matched subsets", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_batch_genes", formats)
    pd.DataFrame(rows_out, columns=["subset", "gene", "between_study_sd"]).to_csv(f"{out_dir}/batch_genes.tsv", sep="\t", index=False)
    if all(v is not None for v in tops.values()):
        a, b = set(tops["MG1655"].index), set(tops["BW25113"].index)
        result["overlap"] = {"both": sorted(a & b), "MG1655_only": sorted(a - b), "BW25113_only": sorted(b - a)}
    return result
