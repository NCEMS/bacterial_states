"""The `batch_tracking` panel: how the most study-variable RegulonDB sets of LB exponential cultures follow the growth
state of each study.

The sets are the ``TOP`` RegulonDB regulons (whole: activated and repressed targets together) with the largest NES (FDR < 0.05) in the pooled batch GSEA of the module
table (`gsea_batch_modules.tsv` of the `gsea` panel, `fig_gsea_batch_regulondb`), or ``DEFAULT_SETS`` without it.
The studies are those of ``pca.lab_subsets``: exponential, untreated wild type, LB; MG1655, BW25113 and, as a third
group, the other laboratory strains (``pca.LAB_GROUPS``). One point per study: the set's mean log2 expression against the
mean log2 expression of the ribosomal proteins (``x="ribosome"``, the default; a growth-rate marker) or the study's
mean phase ordinal score (``x="phase"``: the growth-phase axis of the embedding, early exponential = 0, stationary =
1, fitted as `ecoli_atlas axes` does in the atlas_axes example and scored without the sample's study; within
exponential cultures it explains less of the sets' variation than the ribosomal proteins). Panels are in the order
of the NES, and every panel spans the same number of log2 units, so the spreads compare directly. Each panel gives
the R² of one line for all studies and the residual SD around it (log2 units): how much of the between-study
variation the growth state explains, and how much scatter is left. Sets that change with the growth slowdown in LB
(RpoS, Crp, Fnr, ArcA) follow the ribosomal proteins; the flagellar sets (FlhDC, FliA), which differ between lab
stocks (on or off), do not."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pca import OTHER_LAB as OTHER, lab_subsets
from .plotting import savefig

TABLE_FILE = "batch_tracking.tsv"
TOP = 9
DEFAULT_SETS = ("regulondb:RpoS", "regulondb:Crp", "regulondb:Fnr", "regulondb:ArcA", "regulondb:FliA")
RIBOSOME = r"^rp[lsm][A-Z]$"
GSEA_STRAINS = ("MG1655", "BW25113")       # the strains whose within-strain spreads order the panels without GSEA
COLORS = {"MG1655": "#1f77b4", "BW25113": "#d62728", OTHER: "#2ca02c"}
PHASE_AXIS = "phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary"
PHASE_HOLDOUT = ("lag", "late_stationary", "long_term_stationary")
PHASE_WHERE = {"treated": ["false"], "standard_rnaseq": ["true"]}
X_AXES = {"phase": "phase ordinal score, study mean\n(early exponential = 0, stationary = 1)",
          "ribosome": "ribosomal proteins, mean log2 expression\n(slower growth  <-  ->  faster growth)"}
NOT_SETS = ("strain", "strain_name", "gse", "n", "ribosome", "phase")


def top_sets(out_dir, database: str = "RegulonDB", n: int = TOP) -> pd.DataFrame | None:
    """The ``n`` sets of ``database`` with FDR < 0.05 and the largest NES in the pooled batch GSEA
    (`gsea_batch_modules.tsv` in ``out_dir``), in that order: term, nes; None when the table is missing."""
    path = Path(out_dir) / "gsea_batch_modules.tsv"
    if not path.exists():
        return None
    a = pd.read_csv(path, sep="\t", keep_default_na=False)
    a = a[a["database"] == database].astype({"nes": float, "fdr": float})
    return a[a["fdr"] < 0.05].sort_values("nes", ascending=False).head(n)[["term", "nes"]].reset_index(drop=True)


def phase_scores(ctx) -> np.ndarray:
    """Every sample's phase ordinal score on the embedding (``PHASE_AXIS``, as in the atlas_axes example: untreated
    standard RNA-seq, lag and the late stationary levels held out; each sample scored by a model that has not seen its
    study)."""
    from ..axes import AxisSpec, axis_scores, category_frame
    spec = AxisSpec.parse(PHASE_AXIS, holdout=PHASE_HOLDOUT)
    frame = category_frame(ctx.meta, ctx.data.splits, spec, PHASE_WHERE)
    return axis_scores(ctx.z, frame, spec, seed=ctx.seed)["scores"]["ordinal"].to_numpy()


def study_table(ctx, modules, terms) -> tuple[pd.DataFrame, dict]:
    """One row per study of ``pca.lab_subsets``: strain, strain_name (the most common one), gse, n, the mean phase
    ordinal score, and the mean log2 expression of the ribosomal proteins and of every set of ``terms``; and
    {term: set name}."""
    from ..modules import read_module_table, whole_regulons
    mod = whole_regulons(read_module_table(modules)).set_index("term")
    sym = pd.Index([str(x) for x in ctx.gene_names])
    groups = {t: np.where(sym.isin(mod.loc[t, "genes"].split(",")))[0] for t in terms if t in mod.index}
    names = {t: str(mod.loc[t, "name"]) for t in groups}
    rib = np.where(sym.str.match(RIBOSOME))[0]
    phase = phase_scores(ctx)
    strain_name = ctx.column("strain_name")
    rows = []
    for strain, mask in lab_subsets(ctx).items():
        idx = np.where(mask)[0]
        for gse in pd.unique(ctx.gse[idx]):
            r = idx[ctx.gse[idx] == gse]
            rows.append({"strain": strain, "strain_name": (pd.Series(strain_name[r]).replace("", np.nan).mode().tolist() or [ctx.column("strain_group")[r[0]]])[0], "gse": gse, "n": len(r),
                         "phase": float(np.nanmean(phase[r])), "ribosome": float(ctx.logx[np.ix_(r, rib)].mean()),
                         **{k: float(ctx.logx[np.ix_(r, c)].mean()) for k, c in groups.items() if len(c)}})
    return pd.DataFrame(rows), names


def _size(n):
    return 25 + 12 * np.asarray(n, dtype=float)


def spreads(t: pd.DataFrame, cols) -> pd.DataFrame:
    """Per set: the SD of the study means within MG1655 and within BW25113 and their sum; sorted by the sum, largest first."""
    sd = t[t["strain"].isin(GSEA_STRAINS)].groupby("strain")[list(cols)].std().T
    sd["sum"] = sd.sum(axis=1)
    return sd.sort_values("sum", ascending=False)


def plot_batch_tracking(t: pd.DataFrame, out_dir, formats, names: dict | None = None, nes: pd.DataFrame | None = None,
                        x: str = "ribosome") -> dict:
    """The figure from the table of ``study_table`` (also `batch_tracking.tsv`): one panel per set column (every column
    but ``NOT_SETS``) against column ``x`` (``X_AXES``), in the order of ``nes`` (``top_sets``), else by ``spreads``.
    """
    import matplotlib.pyplot as plt
    names = names or {}
    cols = [c for c in t.columns if c not in NOT_SETS]
    sp = spreads(t, cols)
    nes = nes.set_index("term") if nes is not None else None
    order = [c for c in nes.index if c in cols] if nes is not None else list(sp.index)
    t = t[t[x].notna()].reset_index(drop=True)
    xv = t[x].to_numpy(dtype=float)
    span = max(t[c].max() - t[c].min() for c in order) + 1.0
    ncol = 3
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.3 * nrow + 0.8), squeeze=False)
    axes = axes.ravel()
    groups = [g for g in COLORS if (t["strain"] == g).any()]
    result = {"x": x, "n_studies": int(len(t)), "studies": {g: int((t["strain"] == g).sum()) for g in groups},
              "y_span_log2": float(span), "sets": {}}
    for ax, col in zip(axes, order):
        y = t[col].to_numpy()
        for strain in groups:
            m = (t["strain"] == strain).to_numpy()
            ax.scatter(xv[m], y[m], s=_size(t["n"][m]), color=COLORS[strain], alpha=0.75, edgecolors="black", linewidths=0.5)
        r = float(np.corrcoef(xv, y)[0, 1])
        k, b = np.polyfit(xv, y, 1)
        resid = y - (k * xv + b)
        r2 = 1 - float(resid.var()) / float(y.var())                   # one line for all studies
        rsd = float(np.sqrt((resid ** 2).sum() / max(len(y) - 2, 1)))    # scatter around it, log2 units
        xs = np.linspace(xv.min(), xv.max(), 10)
        ax.plot(xs, k * xs + b, color="0.5", lw=1, ls="--")
        mid = (y.min() + y.max()) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
        ax.set_title(f"{names.get(col, col)}\nR\u00b2 = {r2:.2f};  residual SD = {rsd:.2f} log2", fontsize=11)
        ax.set_ylabel("mean log2 expression", fontsize=10)
        ax.set_xlabel(X_AXES[x], fontsize=9.5)
        result["sets"][col] = {"name": names.get(col, col), "r": r, "r2": r2, "residual_sd": rsd,
                               "sd_between_studies": {k_: float(v) for k_, v in sp.loc[col].items()}}
    for ax in axes[len(order):]:
        ax.axis("off")
    strains = [plt.scatter([], [], s=_size(3), color=COLORS[g], alpha=0.75, edgecolors="black", linewidths=0.5, label=g)
               for g in groups]
    nmax = int(t["n"].max())
    counts = [plt.scatter([], [], s=_size(n), color="0.6", alpha=0.75, edgecolors="black", linewidths=0.5,
                          label=f"{n} sample{'s' if n > 1 else ''}") for n in sorted({n for n in (1, 3, 8, nmax) if n <= nmax})]
    fig.legend(handles=strains + counts, loc="lower center", ncol=len(strains) + len(counts), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0), title="strain; samples in the study (point size)", title_fontsize=10)
    what = "the phase ordinal score" if x == "phase" else "the ribosomal proteins"
    fig.suptitle(f"{len(t)} LB 'exponential' studies of lab strains, one point each; the most study-variable RegulonDB sets "
                 f"against {what}\n(one line for all studies; every panel spans {span:.0f} log2 units)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, out_dir, "fig_batch_tracking", formats)
    return result


def fig_batch_tracking(ctx, out_dir, formats, modules=None, x: str = "ribosome") -> dict:
    """Writes `fig_batch_tracking` and `batch_tracking.tsv`; needs the module table (``modules``). Run after the `gsea`
    panel to take its top RegulonDB sets."""
    if not modules:
        return {"skipped": "needs the module table (figures --modules, from `ecoli_atlas download modules`)"}
    found = top_sets(out_dir)
    terms = list(found["term"]) if found is not None else list(DEFAULT_SETS)
    t, names = study_table(ctx, modules, terms)
    if len(t) < 3:
        return {"skipped": f"{len(t)} studies in the lab-strain subsets"}
    t.to_csv(f"{out_dir}/{TABLE_FILE}", sep="\t", index=False, float_format="%.4f")
    res = plot_batch_tracking(t, out_dir, formats, names, found, x=x)
    sets = (found if found is not None else pd.DataFrame({"term": terms})).assign(name=lambda d: d["term"].map(names))
    fit = pd.DataFrame({k: {"r": v["r"], "r2": v["r2"], "residual_sd": v["residual_sd"]} for k, v in res["sets"].items()}).T
    sets.merge(fit, left_on="term", right_index=True).to_csv(f"{out_dir}/batch_tracking_sets.tsv", sep="\t", index=False,
                                                              float_format="%.4f")
    return res | {"from_gsea": found is not None}
