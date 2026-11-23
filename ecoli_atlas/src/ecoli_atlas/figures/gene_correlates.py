"""Gene correlates of each call: Spearman rho of a gene's expression with the head's probability
of its positive class, over all samples. The text file is the gene-set enrichment input."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HEADS
from .plotting import correlate_barh, savefig, spearman_series
from .split_balance import TITLES

N_EACH = 10
N_LIST = 50
TXT_FILE = "head_gene_correlations.txt"
TSV_FILE = "head_gene_correlations.tsv"


def head_gene_correlations(ctx) -> dict:
    """head -> Spearman rho of every gene's log2 expression with P(positive class), ascending."""
    return {h: spearman_series(ctx.logx, ctx.probs[h], ctx.gene_names).sort_values() for h in HEADS}


def write_correlation_txt(corrs: dict, path: Path, n_list: int = N_LIST) -> None:
    """The top ``n_list`` genes each way per head, as plain text (input for gene-set enrichment tools)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Gene correlations with the head probabilities\n")
        fh.write("=" * 50 + "\n\n")
        for h, corr in corrs.items():
            neg, pos = HEADS[h]
            fh.write(f"HEAD: {h}\n")
            fh.write(f"Correlation target: P({pos})\n")
            fh.write("-" * 30 + "\n")
            fh.write(f"Top {n_list} Positive Correlates (toward {pos}):\n")
            for gene, val in corr.sort_values(ascending=False).head(n_list).items():
                fh.write(f"  {gene:10s} {val:8.4f}\n")
            fh.write("\n")
            fh.write(f"Top {n_list} Negative Correlates (toward {neg}):\n")
            for gene, val in corr.sort_values(ascending=True).head(n_list).items():
                fh.write(f"  {gene:10s} {val:8.4f}\n")
            fh.write("\n" + "=" * 50 + "\n\n")


def fig_head_gene_correlates(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    corrs = head_gene_correlations(ctx)
    fig, axes = plt.subplots(1, len(corrs), figsize=(6 * len(corrs), 6.5))
    axes = np.atleast_1d(axes)
    result = {}
    for ax, (h, corr) in zip(axes, corrs.items()):
        neg, pos = HEADS[h]
        correlate_barh(ax, corr, N_EACH, f"{TITLES[h]}: gene rho vs P({pos})", "Spearman rho")
        result[h] = {"target": f"P({pos})", f"toward_{pos}": list(corr.tail(N_EACH).iloc[::-1].index),
                     f"toward_{neg}": list(corr.head(N_EACH).index), "n_genes": int(len(corr))}
    fig.tight_layout()
    savefig(fig, out_dir, "fig_head_gene_correlates", formats)
    out = Path(out_dir)
    write_correlation_txt(corrs, out / TXT_FILE)
    pd.concat([pd.DataFrame({"head": h, "gene": c.index, "feature": pd.Series(ctx.data.fs.ids, index=ctx.gene_names).reindex(c.index).to_numpy(),
                             "spearman_rho": c.values}) for h, c in corrs.items()], ignore_index=True)\
        .to_csv(out / TSV_FILE, sep="\t", index=False, float_format="%.6f")
    return result
