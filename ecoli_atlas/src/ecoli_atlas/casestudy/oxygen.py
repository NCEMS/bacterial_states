"""Low oxygen vs aerobic across two withheld studies and two strains (notebook cell 47): one LDA axis
on the full embedding, per-study AUROC, and the genes along the axis (mean within-study rho)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..labels import oxygen_class
from ..figures.plotting import BLUE, RED, correlate_barh, savefig, spearman_series

OXYGEN_CASE_STUDIES = ("GSE189154", "GSE78756")
N_TOP = 10


def oxygen_annotation(meta: pd.DataFrame, log=None) -> pd.Series:
    """'low' / 'aerobic' / None per sample; unrecognised spellings are reported, never corrected."""
    if "oxygen" not in meta.columns:
        return pd.Series(None, index=meta.index, dtype=object)
    raw = meta["oxygen"]
    oxc = raw.map(oxygen_class)
    if log is not None:
        unknown = raw[raw.notna() & oxc.isna()].astype(str).str.strip().str.lower().value_counts()
        if len(unknown):
            log("[oxygen] annotations not recognised as aerobic / low (left unlabelled): "
                + ", ".join(f"{k!r} x{v}" for k, v in unknown.items()))
    return oxc


def run_oxygen(ctx, out_dir, formats=("pdf", "png"), studies=OXYGEN_CASE_STUDIES, log=None) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from .axis import oriented_lda_score
    log = log or ctx.log
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = ctx.meta
    studies = list(studies)
    oxc = oxygen_annotation(meta, log)
    gse_all = meta["gse"].astype(str).values
    rows = np.concatenate([np.where((gse_all == g) & oxc.notna().values & ~meta["is_stressed"].values.astype(bool))[0]
                           for g in studies])
    if len(rows) == 0:
        raise ValueError(f"no oxygen-annotated, unstressed samples in {studies}")
    y = (oxc.values[rows] == "low").astype(int)
    study = gse_all[rows]
    for g in studies:
        m = study == g
        if len(np.unique(y[m])) != 2:
            raise ValueError(f"{g} does not contribute both oxygen classes")
    if 1 in set(meta["split"].values[rows]):
        raise ValueError("case-study samples are in TRAIN; the analysis would be circular")

    E = ctx.embedding[rows].astype(float)
    score = oriented_lda_score(E, y)
    auroc = float(roc_auc_score(y, score))
    log(f"[oxygen] one discriminant, {len(rows)} samples, {E.shape[1]}-dim embedding, {len(studies)} studies; "
        f"pooled AUROC = {auroc:.3f}")
    strain_of = {g: str(meta["strain_background"].map(str).values[rows][study == g][0]) for g in studies}
    per_study = {}
    for g in studies:
        m = study == g
        low_name = [v for v in pd.unique(meta["oxygen"].map(str).values[rows][m]) if oxygen_class(v) == "low"]
        per_study[g] = {"strain_background": strain_of[g], "n": int(m.sum()), "n_aerobic": int((y[m] == 0).sum()),
                        "n_low": int(y[m].sum()), "low_label": low_name[0] if low_name else None,
                        "auroc_shared_axis": float(roc_auc_score(y[m], score[m]))}
        log(f"  {g:11s} {strain_of[g]:20s} n={int(m.sum()):3d}  aerobic {per_study[g]['n_aerobic']} vs "
            f"{per_study[g]['low_label']} {per_study[g]['n_low']}  AUROC on the shared axis = {per_study[g]['auroc_shared_axis']:.3f}")
    rho = pd.concat([spearman_series(ctx.data.cpm[rows][study == g], score[study == g], ctx.genes) for g in studies],
                    axis=1).mean(axis=1).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    ypos = {g: i for i, g in enumerate(studies)}
    markers = ["o", "s", "^", "D", "v", "P"]
    mark = {g: markers[i % len(markers)] for i, g in enumerate(studies)}
    rng = np.random.RandomState(0)
    span = np.abs(score).max() * 1.2
    for g in studies:
        for grp, color in [(0, BLUE), (1, RED)]:
            m = (study == g) & (y == grp)
            if not m.any():
                continue
            ax.scatter(score[m], ypos[g] + rng.uniform(-0.1, 0.1, size=int(m.sum())), s=60, c=color, marker=mark[g],
                       edgecolor="k", linewidth=.5, alpha=0.8, zorder=3)
            ax.plot([score[m].mean()] * 2, [ypos[g] - 0.26, ypos[g] + 0.26], color=color, lw=3, zorder=4)
    ax.axvline(0, color="grey", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.75, len(studies) - 0.25)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"{g}\n{strain_of[g]}" for g in studies], fontsize=9)
    ax.set_xlabel("LDA discriminant score (single axis fitted on both studies)")
    ax.annotate("", xy=(0.99, -0.155), xytext=(0.01, -0.155), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="0.35"), annotation_clip=False)
    ax.text(0.01, -0.235, "aerobic", transform=ax.transAxes, fontsize=11, color=BLUE, weight="bold", ha="left")
    ax.text(0.99, -0.235, "low oxygen", transform=ax.transAxes, fontsize=11, color=RED, weight="bold", ha="right")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec="k", ms=9, label="aerobic"),
               plt.Line2D([], [], marker="o", ls="", mfc=RED, mec="k", ms=9, label="low oxygen")]
    handles += [plt.Line2D([], [], marker=mark[g], ls="", mfc="0.7", mec="k", ms=9, label=g) for g in studies]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2 + len(studies), fontsize=9, frameon=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"One oxygen axis across {len(studies)} studies and {len(set(strain_of.values()))} strains (AUROC={auroc:.2f})",
                 fontsize=11, pad=40)
    fig.tight_layout()
    savefig(fig, out, "fig_oxygen_lda_projection", formats)

    fig, ax = plt.subplots(figsize=(8, 5))
    correlate_barh(ax, rho, N_TOP,
                   f"Genes along the shared oxygen axis\n{len(rows)} samples, {len(studies)} studies, "
                   f"{len(set(strain_of.values()))} strains",
                   "mean within-study Spearman rho    (aerobic  <-->  low oxygen)", fontsize=9, xlim=(-1, 1))
    fig.tight_layout()
    savefig(fig, out, "fig_oxygen_axis_genes", formats)

    pd.DataFrame({"gene": rho.index, "mean_within_study_rho": rho.values}).sort_values("mean_within_study_rho", ascending=False)\
        .to_csv(out / "oxygen_axis_genes.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame({"sample": meta.index.values[rows], "gse": study, "oxygen": meta["oxygen"].values[rows],
                  "oxygen_class": np.where(y == 1, "low", "aerobic"), "lda_score": score, "split": meta["split"].values[rows]})\
        .to_csv(out / "oxygen_samples.tsv", sep="\t", index=False)
    result = {"studies": studies, "n_samples": int(len(rows)), "embedding_dim": int(E.shape[1]), "pooled_auroc": auroc,
              "per_study": per_study, "toward_low_oxygen": list(rho.tail(N_TOP).iloc[::-1].index),
              "toward_aerobic": list(rho.head(N_TOP).index),
              "n_oxygen_annotated": int(oxc.notna().sum()),
              "n_oxygen_unrecognised": int((meta["oxygen"].notna() & oxc.isna()).sum()) if "oxygen" in meta.columns else 0}
    (out / "oxygen_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result
