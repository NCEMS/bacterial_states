"""Figures of gene-set enrichment (``ecoli_atlas.gsea``): the top terms of one ranking (one figure, or one per
database), and the `gsea` panel of `ecoli_atlas figures` (the sets of the genes that differ most between the LB
exponential studies of the laboratory strains)."""
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
                         up_label: str = "top of the ranking", down_label: str = "bottom of the ranking", databases=None,
                         down: bool = True) -> list:
    """One figure per database (``databases`` gives the order; default: as they come), ``<name>_<database>``: the
    ``top`` terms with the largest positive NES above and the most negative below, the term names as labels
    (with the term's category in brackets when the table has one, e.g. KEGG ``Ribosome [Translation]``). The width
    follows the longest label. Terms with FDR >= 0.05 are drawn pale. ``down=False`` (one-sided tests): the top panel
    only. Returns the figure names written."""
    import matplotlib.pyplot as plt
    dbs = [d for d in (databases or list(dict.fromkeys(res["database"]))) if (res["database"] == d).any()]
    written = []
    for d in dbs:
        sub_all = res[res["database"] == d]
        parts = [(sub_all[sub_all["nes"] > 0].sort_values("nes", ascending=False).head(top), up_label, UP),
                 (sub_all[sub_all["nes"] < 0].sort_values("nes").head(top), down_label, DOWN)][:2 if down else 1]
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
        fig, axes = plt.subplots(len(parts), 1, figsize=(width, len(parts) * (0.33 * n + 1.2) + 0.6), squeeze=False)
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
            ax.set_xlabel("|NES|" if down else "NES", fontsize=11)
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


BATCH_MODULE_DATABASES = ("RegulonDB", "KEGG pathway", "KEGG module")


def fig_gsea_batch(ctx, out_dir, formats, permutations: int = 10000, gene_sets=None, threads: int = 4, sides: str = "one",
                   modules=None, module_databases=BATCH_MODULE_DATABASES, top: int = 20) -> dict:
    """The `gsea` panel: enrichment of the between-study variability ranking of every LB exponential study of the
    laboratory strains (``pca.lab_subsets``: MG1655, BW25113 and the other K-12, B, C and W strains; exponential,
    untreated wild type, LB), pooled: genes by the SD of their per-study means (each strain of a study apart). A gene called missing from a sample's
    strain is left out of that sample's study mean (a strain's deletions would otherwise top the ranking). Writes `gsea_batch.tsv` and
    `fig_gsea_batch_go` (the ``top`` GO terms by NES). ``sides="one"`` (default): which processes are the most
    study-variable; "two" also gives the least variable ones, with negative NES, in a second panel. With ``modules``
    (a module table of `ecoli_atlas download modules`) the same test of its sets: `gsea_batch_modules.tsv` and one bar
    figure per database of ``module_databases``, `fig_gsea_batch_<database>`. RegulonDB is tested as whole regulons
    (``modules.whole_regulons``: activated and repressed targets together), since the ranking has no direction."""
    from ..gsea import run
    from .pca import batch_ranking, lab_subsets
    subsets = lab_subsets(ctx)
    mask = np.logical_or.reduce(list(subsets.values()))
    missing = ctx.data.M.numpy() if ctx.data.M is not None else None
    unit = np.asarray([f"{g}|{n}" for g, n in zip(ctx.gse, ctx.column("strain_name"))], dtype=object)   # a study's strains apart
    ranking, n_s, n_st = batch_ranking(ctx.logx, ctx.gene_names, unit, mask, missing=missing)
    per_group = {g: {"n_samples": int(m.sum()), "n_studies": int(len(pd.unique(ctx.gse[m])))} for g, m in subsets.items()}
    summary = {"n_samples": int(n_s), "n_studies": int(n_st), "groups": per_group, "sides": sides,
               "genes_called_missing": int(missing[mask].any(0).sum()) if missing is not None else 0}
    if ranking is None:
        return summary | {"skipped": "fewer than 3 samples or 2 studies"}
    ranking = ranking.set_axis([str(g).lower() for g in ranking.index])
    both = sides == "two"
    title = f"the most study-variable sets ({n_st} LB exponential studies × strains, lab strains, {n_s} samples)"
    try:
        res, info = run(ranking, gene_sets, permutations=permutations, threads=threads, sides=sides)
    except ValueError as e:                              # e.g. too few ranked genes for any set (small test tables)
        return summary | {"skipped": str(e)}
    res.to_csv(f"{out_dir}/gsea_batch.tsv", sep="\t", index=False, float_format="%.5g")
    fig_gsea_by_database(res.assign(database="GO"), title, out_dir, "fig_gsea_batch", formats, top=top,
                         up_label="most study-variable", down_label="least study-variable", down=both)
    summary["go"] = info
    if modules:
        import tempfile
        from ..modules import read_module_table, whole_regulons
        with tempfile.TemporaryDirectory() as tmp:          # whole regulons: the ranking has no sign, so no direction
            path = f"{tmp}/modules_whole_regulons.tsv"
            whole_regulons(read_module_table(modules)).to_csv(path, sep="\t", index=False)
            mres, minfo = run(ranking, path, permutations=permutations, threads=threads, sides=sides)
        minfo["gene_sets"] = f"{modules} (RegulonDB: whole regulons)"
        mres.to_csv(f"{out_dir}/gsea_batch_modules.tsv", sep="\t", index=False, float_format="%.5g")
        summary["modules"] = minfo
        summary["module_figures"] = fig_gsea_by_database(mres, title, out_dir, "fig_gsea_batch", formats, top=top,
                                                         up_label="most study-variable", down_label="least study-variable",
                                                         databases=list(module_databases), down=both)
    return summary
