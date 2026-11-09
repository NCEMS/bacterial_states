"""Batch structure in raw expression: PCA on log1p(CPM) and the most study-variable genes
(notebook cells 16-17)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..labels import rowkey
from .plotting import savefig

N_TOP_BATCH = 10


def strain_masks(meta: pd.DataFrame) -> dict:
    scols = [c for c in ["strain", "substrain", "strain_background_c"] if c in meta.columns]
    stxt = rowkey(meta, scols, sep=" ").str.lower().values

    def is_(pat):
        return np.array([bool(re.search(pat, s)) for s in stxt])

    return {"mg1655": is_(r"mg[\s_-]?1655"), "bw": is_(r"bw[\s_-]?25113"),
            "control": meta["is_wildtype"].values.astype(bool),
            "exp": (meta["growth_phase_c"] == "exponential").values,
            "complex": (meta["medium_type_c"] == "complex").values}


def matched_subsets(meta: pd.DataFrame) -> dict:
    """The two strain / phase / medium matched subsets used by both batch figures."""
    m = strain_masks(meta)
    base = m["exp"] & m["control"] & m["complex"]
    return {"MG1655": m["mg1655"] & base, "BW25113": m["bw"] & base, "_complex": m["complex"]}


def fit_pca(M: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(M), pca.explained_variance_ratio_ * 100


def _scatter(ax, PCx, ev, meta, mask, color_by, title, topn=None, s=20):
    import matplotlib.pyplot as plt
    if not mask.any():
        ax.axis("off")
        return
    vals = meta[color_by].astype("object").to_numpy(copy=True)
    vals[~mask] = "other"
    if topn is not None:
        keep = pd.Series(vals[mask]).dropna().value_counts().head(topn).index.tolist()
        vals[mask & ~np.isin(vals, keep)] = "other"
    idx = np.arange(len(vals))
    levels = sorted([c for c in pd.unique(vals) if not pd.isna(c) and c != "other"], key=str)
    cmap, ncmax = (plt.cm.tab20, 20) if len(levels) > 10 else (plt.cm.tab10, 10)
    bg = idx[vals == "other"]
    if len(bg):
        ax.scatter(PCx[bg, 0], PCx[bg, 1], s=s * 0.6, alpha=0.15, color="lightgrey", label="other", zorder=1)
    na = idx[pd.isna(pd.Series(vals)).values]
    if len(na):
        ax.scatter(PCx[na, 0], PCx[na, 1], s=s * 0.6, alpha=0.18, color="gainsboro", label="NaN", zorder=1)
    for j, lev in enumerate(levels):
        sub = idx[vals == lev]
        ax.scatter(PCx[sub, 0], PCx[sub, 1], s=s, alpha=0.85, color=cmap(j % ncmax), label=f"{lev} (n={len(sub)})", zorder=2)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.15)
    h, l = ax.get_legend_handles_labels()
    order = [i for i, x in enumerate(l) if x not in ("other", "NaN")] + [i for i, x in enumerate(l) if x in ("other", "NaN")]
    ax.legend([h[i] for i in order], [l[i] for i in order], title=color_by, fontsize=7, framealpha=0.6,
              markerscale=1.3, ncol=2 if len(levels) > 10 else 1, bbox_to_anchor=(1.04, 1), loc="upper left",
              borderaxespad=0)


def fig_pca_batch(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    meta = ctx.meta
    M = ctx.logcpm
    PCx, ev = fit_pca(M, ctx.seed)
    subsets = matched_subsets(meta)
    result = {"n_samples": int(M.shape[0]), "n_genes": int(M.shape[1]),
              "explained_variance_pct": [float(ev[0]), float(ev[1])], "subsets": {}}
    for name, fn in [("MG1655", "fig_pca_mg1655_batch"), ("BW25113", "fig_pca_bw_batch")]:
        mask = subsets[name]
        result["subsets"][name] = {"n_samples": int(mask.sum()), "n_studies": int(meta.loc[mask, "gse"].nunique())}
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        _scatter(ax, PCx, ev, meta, mask, "gse", f"{name}, exponential, untreated, complex medium, by study", topn=10)
        fig.tight_layout()
        savefig(fig, out_dir, fn, formats)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _scatter(ax, PCx, ev, meta, subsets["_complex"], "strain_background_c", "Complex-medium samples by strain background",
             topn=8, s=10)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_pca_strain_all_complex", formats)
    pd.DataFrame(PCx, index=meta.index, columns=["PC1", "PC2"]).to_csv(f"{out_dir}/pca_logcpm_coords.csv")
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
    meta = ctx.meta
    subsets = matched_subsets(meta)
    gse = meta["gse"].astype(str).values
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    tops, rows_out, result = {}, [], {}
    for ax, (name, col) in zip(axes, [("MG1655", "#1f77b4"), ("BW25113", "#d62728")]):
        g, n_s, n_st = top_batch_genes(ctx.logcpm, ctx.genes, gse, subsets[name])
        tops[name] = g
        result[name] = {"n_samples": int(n_s), "n_studies": int(n_st),
                        "top_genes": None if g is None else {k: float(v) for k, v in g.items()}}
        if g is None:
            ax.axis("off")
            continue
        for gene, v in g.items():
            rows_out.append({"subset": name, "gene": gene, "between_study_sd": float(v)})
        y = np.arange(len(g))
        ax.barh(y, g.values[::-1], color=col, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(g.index[::-1], fontsize=8)
        ax.set_xlabel("between-study SD of mean log1p(CPM)")
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
