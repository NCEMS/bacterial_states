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
    """``database:name`` (e.g. ``iModulon:UC-4``, ``KEGG:Ribosome``, ``GO:translation``); the term id without a name."""
    name = res["name"].where(res["name"].astype(str) != "", res["term"])
    if "database" not in res.columns:
        return name
    db = res["database"].fillna("").astype(str)
    return (db + ":" + name.astype(str)).where(db != "", name)


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


ARROW_ON, ARROW_OFF = "#c0392b", "#b0b0b0"


def _regulon_arrows(sub: pd.DataFrame) -> list | None:
    """For RegulonDB sets split by direction (terms ``regulondb:<regulator>+`` / ``-``): per row (regulator name, the
    set's direction "+"/"-", whether the regulator also has a set in the other direction); None for other tables."""
    terms = sub["term"].astype(str)
    if not len(sub) or not (terms.str.startswith("regulondb:") & terms.str[-1].isin(["+", "-"])).all():
        return None
    cat = sub["category"].fillna("").astype(str) if "category" in sub.columns else pd.Series("", index=sub.index)
    names = sub["name"].astype(str).str.replace(r" \((activated|repressed)\)$", "", regex=True)
    return [(n, t[-1], c == "") for n, t, c in zip(names, terms, cat)]


def _draw_arrows(ax, y, info, fontsize: float = 10) -> None:
    """Up and down arrows right of the y tick labels: the set's own direction red, the regulator's other direction grey
    when it has one (absent for a regulator that acts in one direction only)."""
    from matplotlib.transforms import blended_transform_factory, offset_copy
    base = blended_transform_factory(ax.transAxes, ax.transData)
    for yy, (_, eff, both) in zip(y, info):
        for arrow, sign, dx in (("↑", "+", -22), ("↓", "-", -10)):
            if sign != eff and not both:
                continue
            ax.text(0, yy, arrow, transform=offset_copy(base, fig=ax.figure, x=dx, y=0, units="points"), ha="center",
                    va="center", fontsize=fontsize + 1, color=ARROW_ON if sign == eff else ARROW_OFF, fontweight="bold")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")


def fig_gsea_by_database(res: pd.DataFrame, title: str, out_dir, name: str, formats, top: int = 10,
                         up_label: str = "top of the ranking", down_label: str = "bottom of the ranking", databases=None) -> list:
    """One figure per database (``databases`` gives the order; default: as they come), ``<name>_<database>``: the
    ``top`` terms with the largest positive NES above and the most negative below, the term names as labels
    (with the term's category in brackets when the table has one, e.g. KEGG ``Ribosome [Translation]``). The width
    follows the longest label. Terms with FDR >= 0.05 are drawn pale. Returns the figure names written."""
    import matplotlib.pyplot as plt
    dbs = [d for d in (databases or list(dict.fromkeys(res["database"]))) if (res["database"] == d).any()]
    written = []
    for d in dbs:
        sub_all = res[res["database"] == d]
        parts = [(sub_all[sub_all["nes"] > 0].sort_values("nes", ascending=False).head(top), up_label, UP),
                 (sub_all[sub_all["nes"] < 0].sort_values("nes").head(top), down_label, DOWN)]
        labels, arrows = [], []
        for sub, _, _ in parts:
            reg = _regulon_arrows(sub)
            arrows.append(reg)
            if reg is not None:                          # RegulonDB: the regulator's name, its directions as arrows
                labels.append([r[0] for r in reg])
                continue
            names = sub["name"].where(sub["name"].astype(str) != "", sub["term"]).astype(str)
            if "category" in sub.columns:
                cat = sub["category"].fillna("").astype(str)
                names = (names + " [" + cat + "]").where(cat != "", names)
            labels.append(list(names))
        n = max(1, max(len(p[0]) for p in parts))
        longest = max([len(x) for lab in labels for x in lab] or [10])
        width = 5.5 + 0.085 * longest                    # inches: the bars plus the labels at fontsize 10
        fig, axes = plt.subplots(2, 1, figsize=(width, 2 * (0.33 * n + 1.2) + 0.6), squeeze=False)
        xmax = float(sub_all["nes"].abs().max()) * 1.05
        for ax, (sub, side, col), lab, arr in zip(axes[:, 0], parts, labels, arrows):
            y = np.arange(len(sub))[::-1]
            bars = ax.barh(y, sub["nes"].abs(), color=col, height=0.75)
            for bar, f in zip(bars, sub["fdr"]):
                bar.set_alpha(1.0 if f < FDR else 0.35)
            ax.set_yticks(y)
            ax.set_yticklabels(lab, fontsize=10)
            if arr is not None:
                ax.tick_params(axis="y", pad=32, length=0)   # room for the arrows between the names and the axis
                _draw_arrows(ax, y, arr)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlim(0, xmax)
            ax.set_xlabel("|NES|", fontsize=11)
            ax.set_title(side, fontsize=12)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{d}: {title}", fontsize=13)
        note = f"pale: FDR ≥ {FDR}"
        if any(a is not None for a in arrows):
            note = "↑ activated, ↓ repressed: red = this set, grey = the regulator's other direction (none: one direction only);  " + note
        fig.text(0.99, 0.005, note, ha="right", va="bottom", fontsize=9, color=GREY)
        fig.tight_layout()
        fname = f"{name}_{_slug(d)}"
        savefig(fig, out_dir, fname, formats)
        written.append(fname)
    return written


def fig_gsea_scatter(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str, out_dir, name: str, formats,
                     top: int = 9, off_diagonal: int = 0, title: str | None = None) -> pd.DataFrame:
    """NES of every term tested in both rankings, ``a`` against ``b``, with the dashed line y = x.
    The ``top`` terms with the largest |NES_a| + |NES_b| among those with FDR < 0.05 in either are
    labelled (default 9; for RegulonDB the sets of `fig_batch_tracking`), and optionally the ``off_diagonal`` terms with FDR < 0.05 in either on which the two rankings
    disagree most (largest |NES_a - NES_b|; with two-sided NES only those of opposite signs; default none). Returns the
    joined table."""
    import matplotlib.pyplot as plt
    c = a[["term", "name", "nes", "fdr"]].merge(b[["term", "nes", "fdr"]], on="term", suffixes=("_a", "_b"))
    sig = (c["fdr_a"] < FDR) | (c["fdr_b"] < FDR)
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.subplots_adjust(left=0.07, right=0.6)
    ax.scatter(c.loc[~sig, "nes_a"], c.loc[~sig, "nes_b"], s=10, color="0.75", alpha=0.6, label=f"FDR ≥ {FDR} in both")
    ax.scatter(c.loc[sig, "nes_a"], c.loc[sig, "nes_b"], s=18, color="#1f77b4", alpha=0.85, label=f"FDR < {FDR} in either")
    lim = float(np.nanmax(np.abs(c[["nes_a", "nes_b"]].to_numpy()))) * 1.08 if len(c) else 1.0
    low = 0.0 if len(c) and (c[["nes_a", "nes_b"]].to_numpy() >= 0).all() else -lim      # one-sided NES: from 0
    ax.plot([low, lim], [low, lim], ls="--", color=GREY, lw=1)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    pick = c[sig].assign(both=lambda d: d["nes_a"].abs() + d["nes_b"].abs()).sort_values("both", ascending=False).head(top)
    one_sided = len(c) > 0 and bool((c[["nes_a", "nes_b"]].to_numpy() >= 0).all())
    # off the diagonal, where the rankings disagree: two-sided, NES of opposite signs; one-sided, the largest differences
    disagree = c[sig] if one_sided else c[sig & (np.sign(c["nes_a"]) != np.sign(c["nes_b"]))]
    off = disagree.assign(gap=(disagree["nes_a"] - disagree["nes_b"]).abs()).sort_values("gap", ascending=False)
    off = off[~off["term"].isin(pick["term"])].head(off_diagonal)
    pick = pd.concat([pick, off])
    # labels in a column right of the plot, each at its point's height, pushed apart only as far as needed to keep a
    # minimum gap (and kept inside the plot), each joined to its point by a thin line
    pick = pick.sort_values("nes_b")
    n_lab = len(pick)
    if n_lab:
        span = lim - low
        gap = min(span / 16, 0.94 * span / max(n_lab - 1, 1))
        ys = pick["nes_b"].to_numpy(dtype=float).clip(low + 0.03 * span, lim - 0.03 * span)
        for i in range(1, n_lab):                       # push up
            ys[i] = max(ys[i], ys[i - 1] + gap)
        over = ys[-1] - (lim - 0.03 * span)
        if over > 0:                                    # too high: shift down, then keep the gaps from the top
            ys[-1] -= over
            for i in range(n_lab - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        for (_, r), yt in zip(pick.iterrows(), ys):
            ax.annotate(r["name"] or r["term"], (r["nes_a"], r["nes_b"]), xytext=(lim * 1.04, yt), textcoords="data",
                        fontsize=9, ha="left", va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color="0.65", lw=0.5, shrinkA=0, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.2", fc="#fff3a0", alpha=0.8, ec="none"))
    ax.set_xlim(low, lim)
    ax.set_ylim(low, lim)
    ax.set_xlabel(f"{label_a} NES", fontsize=12)
    ax.set_ylabel(f"{label_b} NES", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    if title:
        ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out_dir, name, formats)
    return c


BATCH_MODULE_DATABASES = ("RegulonDB", "KEGG pathway", "KEGG module")


def fig_gsea_batch(ctx, out_dir, formats, permutations: int = 10000, gene_sets=None, threads: int = 4, sides: str = "one",
                   modules=None, module_databases=BATCH_MODULE_DATABASES) -> dict:
    """The `gsea` panel: GO enrichment of the between-study variability ranking (`batch_genes`) of
    MG1655 and of BW25113 (exponential, untreated wild type, LB: ``pca.matched_subsets``). Writes
    `gsea_batch_<strain>.tsv`, `fig_gsea_batch_<strain>` (top 20 terms by NES) and `fig_gsea_batch_scatter`.
    ``sides="one"`` (default): which processes are the most study-variable; "two" also
    gives the least variable ones, with negative NES. With ``modules`` (a module table of `ecoli_atlas download
    modules`) the same test of its sets: `gsea_batch_modules_<strain>.tsv` and one scatter per database of
    ``module_databases``, `fig_gsea_batch_scatter_<database>`."""
    from ..gsea import run
    from .pca import batch_ranking, matched_subsets
    subsets = matched_subsets(ctx)
    results, mod_results, summary = {}, {}, {}
    for strain, color in (("MG1655", "#1f77b4"), ("BW25113", "#d62728")):
        ranking, n_s, n_st = batch_ranking(ctx.logx, ctx.gene_names, ctx.gse, subsets[strain])
        if ranking is None:
            summary[strain] = {"n_samples": int(n_s), "n_studies": int(n_st), "skipped": "fewer than 3 samples or 2 studies"}
            continue
        res, info = run(ranking.set_axis([str(g).lower() for g in ranking.index]), gene_sets, permutations=permutations,
                        threads=threads, sides=sides)
        res.to_csv(f"{out_dir}/gsea_batch_{strain.lower()}.tsv", sep="\t", index=False, float_format="%.5g")
        fig_gsea_bars(res, f"{strain}: top enriched GO terms (n={n_s} samples, {n_st} studies)", out_dir,
                      f"fig_gsea_batch_{strain.lower()}", formats, color=color)
        results[strain] = res
        summary[strain] = {"n_samples": int(n_s), "n_studies": int(n_st), "sides": sides} | info
        if modules:
            mres, minfo = run(ranking.set_axis([str(g).lower() for g in ranking.index]), modules, permutations=permutations,
                              threads=threads, sides=sides)
            mres.to_csv(f"{out_dir}/gsea_batch_modules_{strain.lower()}.tsv", sep="\t", index=False, float_format="%.5g")
            mod_results[strain] = mres
            summary[strain]["modules"] = minfo
    if len(results) == 2:
        joined = fig_gsea_scatter(results["MG1655"], results["BW25113"], "MG1655", "BW25113", out_dir, "fig_gsea_batch_scatter", formats)
        summary["nes_correlation"] = float(np.corrcoef(joined["nes_a"], joined["nes_b"])[0, 1]) if len(joined) > 2 else None
    if len(mod_results) == 2:
        summary["module_nes_correlation"] = {}
        for d in module_databases:
            a, b = (mod_results[s][mod_results[s]["database"] == d] for s in ("MG1655", "BW25113"))
            if a.empty or b.empty:
                continue
            j = fig_gsea_scatter(a, b, "MG1655", "BW25113", out_dir, f"fig_gsea_batch_scatter_{_slug(d)}", formats,
                                 title=f"{d}: the most study-variable sets")
            summary["module_nes_correlation"][d] = float(np.corrcoef(j["nes_a"], j["nes_b"])[0, 1]) if len(j) > 2 else None
    return summary
