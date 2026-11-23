"""The `batch_tracking` panel: how the most study-variable RegulonDB sets of the matched subsets
(``pca.matched_subsets``: exponential, untreated wild type, LB) follow the growth rate of each study.

The sets are the ``TOP`` RegulonDB sets with the largest NES summed over the two strains in the batch GSEA of the
module table (`gsea_batch_modules_<strain>.tsv` of the `gsea` panel; the sets labelled in
`fig_gsea_batch_scatter_regulondb`), or ``DEFAULT_SETS`` without those tables. One point per study (MG1655 and
BW25113): the set's mean log2 expression against that of the ribosomal proteins (a growth-rate marker: higher = faster
growth). Panels are ordered by |NES| in MG1655 plus |NES| in BW25113 (the order of the scatter's labels), and every
panel spans the same number of log2 units, so the spreads compare directly. Each panel gives the R² of one line for
both strains and the residual SD around it (log2 units): how much of the between-study variation the growth rate
explains, and how much scatter is left. Sets that
change with the growth slowdown in LB (RpoS, Crp, Fnr, ArcA) follow the ribosomal proteins; the flagellar sets (FlhDC,
FliA), which differ between lab stocks (on or off), do not."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pca import matched_subsets
from .plotting import savefig

TABLE_FILE = "batch_tracking.tsv"
TOP = 9
DEFAULT_SETS = ("regulondb:RpoS+", "regulondb:Crp+", "regulondb:Fnr+", "regulondb:ArcA-", "regulondb:FliA+")
RIBOSOME = r"^rp[lsm][A-Z]$"
COLORS = {"MG1655": "#1f77b4", "BW25113": "#d62728"}


def top_sets(out_dir, database: str = "RegulonDB", n: int = TOP) -> pd.DataFrame | None:
    """The ``n`` sets of ``database`` with FDR < 0.05 in either strain and the largest |NES| summed over both, from the
    `gsea_batch_modules_<strain>.tsv` tables in ``out_dir``, in that order: term, nes_MG1655, nes_BW25113, nes_sum; None
    when the tables are missing."""
    paths = [Path(out_dir) / f"gsea_batch_modules_{s.lower()}.tsv" for s in COLORS]
    if not all(p.exists() for p in paths):
        return None
    a, b = (pd.read_csv(p, sep="\t", keep_default_na=False) for p in paths)
    j = a[a["database"] == database].merge(b[b["database"] == database], on="term", suffixes=("_a", "_b"))
    for c in ("nes_a", "nes_b", "fdr_a", "fdr_b"):
        j[c] = j[c].astype(float)
    j = j[(j["fdr_a"] < 0.05) | (j["fdr_b"] < 0.05)]
    j = j.assign(nes_sum=j["nes_a"].abs() + j["nes_b"].abs()).sort_values("nes_sum", ascending=False).head(n)
    return j.rename(columns={"nes_a": "nes_MG1655", "nes_b": "nes_BW25113"})[["term", "nes_MG1655", "nes_BW25113", "nes_sum"]]


def study_table(ctx, modules, terms) -> tuple[pd.DataFrame, dict]:
    """One row per study of the matched subsets: strain, gse, n and the mean log2 expression of every set of ``terms``
    and of the ribosomal proteins; and {term: set name}."""
    from ..modules import read_module_table
    mod = read_module_table(modules).set_index("term")
    sym = pd.Index([str(x) for x in ctx.gene_names])
    groups = {t: np.where(sym.isin(mod.loc[t, "genes"].split(",")))[0] for t in terms if t in mod.index}
    names = {t: str(mod.loc[t, "name"]) for t in groups}
    groups["ribosome"] = np.where(sym.str.match(RIBOSOME))[0]
    subsets = matched_subsets(ctx)
    rows = []
    for strain in COLORS:
        idx = np.where(subsets[strain])[0]
        for gse in pd.unique(ctx.gse[idx]):
            r = idx[ctx.gse[idx] == gse]
            rows.append({"strain": strain, "gse": gse, "n": len(r),
                         **{k: float(ctx.logx[np.ix_(r, c)].mean()) for k, c in groups.items() if len(c)}})
    return pd.DataFrame(rows), names


def _size(n):
    return 25 + 12 * np.asarray(n, dtype=float)


def spreads(t: pd.DataFrame, cols) -> pd.DataFrame:
    """Per set: the SD of the study means within each strain and their sum; sorted by the sum, largest first."""
    sd = t.groupby("strain")[list(cols)].std().T
    sd["sum"] = sd.sum(axis=1)
    return sd.sort_values("sum", ascending=False)


def plot_batch_tracking(t: pd.DataFrame, out_dir, formats, names: dict | None = None, nes: pd.DataFrame | None = None) -> dict:
    """The figure from the table of ``study_table`` (also `batch_tracking.tsv`): one panel per set column (every column
    but strain, gse, n and ribosome), in the order of ``nes`` (``top_sets``; its NES in the panel titles), else by
    ``spreads``."""
    import matplotlib.pyplot as plt
    names = names or {}
    cols = [c for c in t.columns if c not in ("strain", "gse", "n", "ribosome")]
    sp = spreads(t, cols)
    nes = nes.set_index("term") if nes is not None else None
    order = [c for c in nes.index if c in cols] if nes is not None else list(sp.index)
    x = t["ribosome"].to_numpy()
    span = max(t[c].max() - t[c].min() for c in order) + 1.0
    ncol = 3
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.3 * nrow + 0.8), squeeze=False)
    axes = axes.ravel()
    result = {"n_studies": int(len(t)), "y_span_log2": float(span), "sets": {}}
    for ax, col in zip(axes, order):
        y = t[col].to_numpy()
        for strain, color in COLORS.items():
            m = (t["strain"] == strain).to_numpy()
            ax.scatter(x[m], y[m], s=_size(t["n"][m]), color=color, alpha=0.75, edgecolors="black", linewidths=0.5)
        r = float(np.corrcoef(x, y)[0, 1])
        k, b = np.polyfit(x, y, 1)
        resid = y - (k * x + b)
        r2 = 1 - float(resid.var()) / float(y.var())                   # one line for both strains
        rsd = float(np.sqrt((resid ** 2).sum() / max(len(y) - 2, 1)))    # scatter around it, log2 units
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, k * xs + b, color="0.5", lw=1, ls="--")
        mid = (y.min() + y.max()) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
        s = sp.loc[col]
        ax.set_title(f"{names.get(col, col)}\nR\u00b2 = {r2:.2f};  residual SD = {rsd:.2f} log2", fontsize=11)
        ax.set_ylabel("mean log2 expression", fontsize=10)
        ax.set_xlabel("ribosomal proteins, mean log2 expression\n(slower growth  <-  ->  faster growth)", fontsize=9.5)
        result["sets"][col] = {"name": names.get(col, col), "r_with_ribosome": r, "r2": r2, "residual_sd": rsd,
                               "sd_between_studies": {k_: float(v) for k_, v in s.items()}}
    for ax in axes[len(order):]:
        ax.axis("off")
    strains = [plt.scatter([], [], s=_size(3), color=c, alpha=0.75, edgecolors="black", linewidths=0.5, label=s_)
               for s_, c in COLORS.items()]
    nmax = int(t["n"].max())
    counts = [plt.scatter([], [], s=_size(n), color="0.6", alpha=0.75, edgecolors="black", linewidths=0.5,
                          label=f"{n} sample{'s' if n > 1 else ''}") for n in sorted({n for n in (1, 3, 8, nmax) if n <= nmax})]
    fig.legend(handles=strains + counts, loc="lower center", ncol=len(strains) + len(counts), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0), title="strain; samples in the study (point size)", title_fontsize=10)
    fig.suptitle(f"{len(t)} LB 'mid-exponential' studies, one point each; the most study-variable RegulonDB sets against the "
                 f"growth rate (one line for both strains; every panel spans {span:.0f} log2 units)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, out_dir, "fig_batch_tracking", formats)
    return result


def fig_batch_tracking(ctx, out_dir, formats, modules=None) -> dict:
    """Writes `fig_batch_tracking` and `batch_tracking.tsv`; needs the module table (``modules``). Run after the `gsea`
    panel to take its top RegulonDB sets."""
    if not modules:
        return {"skipped": "needs the module table (figures --modules, from `ecoli_atlas download modules`)"}
    found = top_sets(out_dir)
    terms = list(found["term"]) if found is not None else list(DEFAULT_SETS)
    t, names = study_table(ctx, modules, terms)
    if len(t) < 3:
        return {"skipped": f"{len(t)} studies in the matched subsets"}
    t.to_csv(f"{out_dir}/{TABLE_FILE}", sep="\t", index=False, float_format="%.4f")
    res = plot_batch_tracking(t, out_dir, formats, names, found)
    sets = (found if found is not None else pd.DataFrame({"term": terms})).assign(name=lambda d: d["term"].map(names))
    fit = pd.DataFrame({k: {"r2": v["r2"], "residual_sd": v["residual_sd"]} for k, v in res["sets"].items()}).T
    sets.merge(fit, left_on="term", right_index=True).to_csv(f"{out_dir}/batch_tracking_sets.tsv", sep="\t", index=False,
                                                              float_format="%.4f")
    return res | {"from_gsea": found is not None}
