"""Gene correlates of each factor axis (notebook cells 31 and 33)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import correlate_barh, savefig, spearman_series

N_EACH = 10
N_LIST = 50
TXT_FILE = "factor_gene_correlations.txt"
TSV_FILE = "factor_gene_correlations.tsv"


def factor_gene_correlations(ctx) -> dict:
    """factor -> (Spearman rho of every gene's CPM with P(positive class), ascending)."""
    out = {}
    for f in ctx.factors:
        corr = spearman_series(ctx.data.cpm, ctx.proba_positive(f), ctx.genes).sort_values()
        out[f] = corr
    return out


def write_correlation_txt(corrs: dict, ctx, path: Path, n_list: int = N_LIST) -> None:
    """Byte-for-byte the layout the notebook wrote (used downstream for gene-set enrichment)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Gene Correlations for Disentangled Factors\n")
        fh.write("=" * 50 + "\n\n")
        for f, corr in corrs.items():
            cats, pos = ctx.classes(f), ctx.positive_index(f)
            fh.write(f"FACTOR: {f}\n")
            fh.write(f"Correlation target: P({cats[pos]})\n")
            fh.write("-" * 30 + "\n")
            fh.write(f"Top {n_list} Positive Correlates (toward {cats[pos]}):\n")
            for gene, val in corr.sort_values(ascending=False).head(n_list).items():
                fh.write(f"  {gene:10s} {val:8.4f}\n")
            fh.write("\n")
            fh.write(f"Top {n_list} Negative Correlates (toward {cats[1 - pos]}):\n")
            for gene, val in corr.sort_values(ascending=True).head(n_list).items():
                fh.write(f"  {gene:10s} {val:8.4f}\n")
            fh.write("\n" + "=" * 50 + "\n\n")


def fig_factor_gene_correlates(ctx, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    corrs = factor_gene_correlations(ctx)
    factors = list(corrs)
    fig, axes = plt.subplots(1, len(factors), figsize=(6 * len(factors), 6.5))
    axes = np.atleast_1d(axes)
    result = {}
    for ax, f in zip(axes, factors):
        cats, pos = ctx.classes(f), ctx.positive_index(f)
        corr = corrs[f]
        correlate_barh(ax, corr, N_EACH, f"{f}: gene rho vs P({cats[pos]})", "Spearman rho")
        result[f] = {"target": f"P({cats[pos]})",
                     f"toward_{cats[pos]}": list(corr.tail(N_EACH).iloc[::-1].index),
                     f"toward_{cats[1 - pos]}": list(corr.head(N_EACH).index),
                     "n_genes": int(corr.notna().sum())}
    fig.tight_layout()
    savefig(fig, out_dir, "fig_factor_gene_correlates", formats)
    out = Path(out_dir)
    write_correlation_txt(corrs, ctx, out / TXT_FILE)
    long = pd.concat([pd.DataFrame({"factor": f, "gene": c.index, "spearman_rho": c.values}) for f, c in corrs.items()],
                     ignore_index=True)
    long.to_csv(out / TSV_FILE, sep="\t", index=False, float_format="%.6f")
    return result
