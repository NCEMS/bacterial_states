"""Figures of gene-set enrichment (``ecoli_atlas.gsea``): the top terms of one ranking, the terms of two
rankings against each other, and the `gsea` panel of `ecoli_atlas figures` (GO terms of the genes that
differ most between studies of MG1655 and of BW25113, the ranking of `batch_genes`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import GREY, savefig

UP, DOWN = "#b2182b", "#2166ac"
FDR = 0.05


def _label(res: pd.DataFrame) -> pd.Series:
    return res["name"].where(res["name"].astype(str) != "", res["term"])


def fig_gsea_bars(res: pd.DataFrame, title: str, out_dir, name: str, formats, top: int = 20, both: bool = False,
                  up_label: str = "top of the ranking", down_label: str = "bottom of the ranking", color: str = UP) -> None:
    """The ``top`` terms with the largest NES (``both``: also the most negative NES, in a second panel).
    Terms with FDR >= 0.05 are drawn pale."""
    import matplotlib.pyplot as plt
    parts = [(res[res["nes"] > 0].sort_values("nes", ascending=False).head(top), up_label, color)]
    if both:
        parts.append((res[res["nes"] < 0].sort_values("nes").head(top), down_label, DOWN))
    n = max(1, max(len(p[0]) for p in parts))
    fig, axes = plt.subplots(1, len(parts), figsize=(8 * len(parts), max(4.0, 0.33 * n + 1.6)), squeeze=False)
    for ax, (sub, sub_title, col) in zip(axes[0], parts):
        y = np.arange(len(sub))[::-1]
        bars = ax.barh(y, sub["nes"].abs(), color=col, height=0.75)
        for bar, f in zip(bars, sub["fdr"]):
            bar.set_alpha(1.0 if f < FDR else 0.35)
        ax.set_yticks(y)
        ax.set_yticklabels(_label(sub), fontsize=10)
        ax.set_xlabel("|NES|" if both else "Normalized Enrichment Score (NES)", fontsize=12)
        ax.set_title(sub_title if both else title, fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
    if both:
        fig.suptitle(title, fontsize=14)
    fig.text(0.99, 0.005, f"pale: FDR ≥ {FDR}", ha="right", va="bottom", fontsize=9, color=GREY)
    fig.tight_layout()
    savefig(fig, out_dir, name, formats)


def fig_gsea_scatter(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str, out_dir, name: str, formats,
                     top: int = 15) -> pd.DataFrame:
    """NES of every term tested in both rankings, ``a`` against ``b``, with the dashed line y = x.
    The ``top`` terms with the largest |NES_a| + |NES_b| among those with FDR < 0.05 in either are
    labelled. Returns the joined table."""
    import matplotlib.pyplot as plt
    c = a[["term", "name", "nes", "fdr"]].merge(b[["term", "nes", "fdr"]], on="term", suffixes=("_a", "_b"))
    sig = (c["fdr_a"] < FDR) | (c["fdr_b"] < FDR)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(c.loc[~sig, "nes_a"], c.loc[~sig, "nes_b"], s=10, color="0.75", alpha=0.6, label=f"FDR ≥ {FDR} in both")
    ax.scatter(c.loc[sig, "nes_a"], c.loc[sig, "nes_b"], s=18, color="#1f77b4", alpha=0.85, label=f"FDR < {FDR} in either")
    lim = float(np.nanmax(np.abs(c[["nes_a", "nes_b"]].to_numpy()))) * 1.08 if len(c) else 1.0
    ax.plot([-lim, lim], [-lim, lim], ls="--", color=GREY, lw=1)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    pick = c[sig].assign(both=lambda d: d["nes_a"].abs() + d["nes_b"].abs()).sort_values("both", ascending=False).head(top)
    placed = []                                   # simple vertical spreading of labels that would overlap
    for _, r in pick.sort_values("nes_b").iterrows():
        yt = r["nes_b"]
        for xp, yp in placed:
            if abs(xp - r["nes_a"]) < 0.25 * lim and abs(yp - yt) < 0.055 * lim:
                yt = yp + 0.055 * lim
        placed.append((r["nes_a"], yt))
        ax.annotate(r["name"] or r["term"], (r["nes_a"], r["nes_b"]), xytext=(r["nes_a"] + 0.02 * lim, yt + 0.01 * lim),
                    fontsize=9, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff3a0", alpha=0.7, ec="none"))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"{label_a} NES", fontsize=12)
    ax.set_ylabel(f"{label_b} NES", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out_dir, name, formats)
    return c


def fig_gsea_batch(ctx, out_dir, formats, permutations: int = 10000, gene_sets=None, threads: int = 4) -> dict:
    """The `gsea` panel: GO enrichment of the between-study variability ranking (`batch_genes`) of
    MG1655 and of BW25113 (exponential, untreated wild type, rich medium). Writes
    `gsea_batch_<strain>.tsv`, `fig_gsea_batch_<strain>` (top 20 terms by NES) and `fig_gsea_batch_scatter`."""
    from ..gsea import run
    from .pca import batch_ranking, matched_subsets
    subsets = matched_subsets(ctx)
    results, summary = {}, {}
    for strain, color in (("MG1655", "#1f77b4"), ("BW25113", "#d62728")):
        ranking, n_s, n_st = batch_ranking(ctx.logx, ctx.gene_names, ctx.gse, subsets[strain])
        if ranking is None:
            summary[strain] = {"n_samples": int(n_s), "n_studies": int(n_st), "skipped": "fewer than 3 samples or 2 studies"}
            continue
        res, info = run(ranking.set_axis([str(g).lower() for g in ranking.index]), gene_sets, permutations=permutations,
                        threads=threads)
        res.to_csv(f"{out_dir}/gsea_batch_{strain.lower()}.tsv", sep="\t", index=False, float_format="%.5g")
        fig_gsea_bars(res, f"{strain}: top enriched GO terms (n={n_s} samples, {n_st} studies)", out_dir,
                      f"fig_gsea_batch_{strain.lower()}", formats, color=color)
        results[strain] = res
        summary[strain] = {"n_samples": int(n_s), "n_studies": int(n_st)} | info
    if len(results) == 2:
        joined = fig_gsea_scatter(results["MG1655"], results["BW25113"], "MG1655", "BW25113", out_dir, "fig_gsea_batch_scatter", formats)
        summary["nes_correlation"] = float(np.corrcoef(joined["nes_a"], joined["nes_b"])[0, 1]) if len(joined) > 2 else None
    return summary
