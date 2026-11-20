"""Low oxygen against aerobic growth in studies the model was not trained on: one discriminant
axis on the atlas embedding, its AUROC per study, and the genes along it (mean within-study rho).

The axis is fitted on the samples it is drawn for, so ``pooled_auroc`` is an in-sample number;
``auroc_leave_one_out`` scores every sample with an axis fitted without it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..figures.plotting import BLUE, RED, correlate_barh, savefig, spearman_series

OXYGEN_CASE_STUDIES = ("GSE189154",)
LOW = ("anaerobic", "microaerobic")
N_TOP = 10


def run_oxygen(ctx, out_dir, formats=("pdf", "png"), studies=OXYGEN_CASE_STUDIES, log=None) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from .axis import lopo_axis, oriented_lda_score
    log = log or ctx.log
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    studies = list(studies)
    oxygen = ctx.column("oxygen")
    known = np.isin(oxygen, ("aerobic",) + LOW) & (ctx.column("treated") == "false")
    rows = np.concatenate([np.where((ctx.gse == g) & known)[0] for g in studies])
    if not len(rows):
        raise ValueError(f"no untreated sample with a stated oxygen condition in {studies}")
    y = np.isin(oxygen[rows], LOW).astype(int)
    study = ctx.gse[rows]
    for g in studies:
        if len(np.unique(y[study == g])) != 2:
            raise ValueError(f"{g} does not have both oxygen classes among its untreated samples")
    if (ctx.split[rows] == 1).any():
        raise ValueError("case-study samples are in the training pool; the analysis would be circular")

    E = ctx.z[rows].astype(float)
    score = oriented_lda_score(E, y)
    auroc = float(roc_auc_score(y, score))
    loo = lopo_axis(E, y, np.arange(len(rows)))
    auroc_loo = float(roc_auc_score(y, loo))
    log(f"[oxygen] one discriminant, {len(rows)} samples, {E.shape[1]}-dim embedding, {len(studies)} studies; "
        f"AUROC {auroc:.3f} in sample, {auroc_loo:.3f} leaving each sample out")
    strain_of = {g: ctx.column("strain_name")[rows][study == g][0] for g in studies}
    per_study = {}
    for g in studies:
        m = study == g
        per_study[g] = {"strain": strain_of[g], "n": int(m.sum()), "n_aerobic": int((y[m] == 0).sum()), "n_low": int(y[m].sum()),
                        "low_label": sorted(set(oxygen[rows][m][y[m] == 1])), "auroc_shared_axis": float(roc_auc_score(y[m], score[m])),
                        "auroc_leave_one_out": float(roc_auc_score(y[m], loo[m]))}
    rho = pd.concat([spearman_series(ctx.logx[rows][study == g], score[study == g], ctx.gene_names) for g in studies],
                    axis=1).mean(axis=1).sort_values()

    fig, ax = plt.subplots(figsize=(10, 3.2 + 0.9 * len(studies)))
    ypos = {g: i for i, g in enumerate(studies)}
    markers = ["o", "s", "^", "D", "v", "P"]
    mark = {g: markers[i % len(markers)] for i, g in enumerate(studies)}
    rng = np.random.RandomState(0)
    span = np.abs(score).max() * 1.2
    for g in studies:
        for cls, color in ((0, BLUE), (1, RED)):
            m = (study == g) & (y == cls)
            ax.scatter(score[m], ypos[g] + rng.uniform(-0.1, 0.1, size=int(m.sum())), s=60, c=color, marker=mark[g],
                       edgecolor="k", linewidth=.5, alpha=0.8, zorder=3)
            ax.plot([score[m].mean()] * 2, [ypos[g] - 0.26, ypos[g] + 0.26], color=color, lw=3, zorder=4)
    ax.axvline(0, color="grey", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.75, len(studies) - 0.25)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"{g}\n{strain_of[g]}" for g in studies], fontsize=10)
    ax.set_xlabel("LDA discriminant score" + (f" (one axis fitted on {len(studies)} studies)" if len(studies) > 1 else ""), fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec="k", ms=9, label="aerobic"),
               plt.Line2D([], [], marker="o", ls="", mfc=RED, mec="k", ms=9, label="low oxygen")]
    if len(studies) > 1:
        handles += [plt.Line2D([], [], marker=mark[g], ls="", mfc="0.7", mec="k", ms=9, label=g) for g in studies]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), fontsize=10, frameon=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Oxygen axis, {len(rows)} samples of {len(studies)} " + ("study" if len(studies) == 1 else "studies")
                 + f" (AUROC {auroc_loo:.2f} leaving each sample out)", fontsize=12, pad=34)
    fig.tight_layout()
    savefig(fig, out, "fig_oxygen_lda_projection", formats)

    fig, ax = plt.subplots(figsize=(8, 5))
    correlate_barh(ax, rho, N_TOP, f"Genes along the oxygen axis\n{len(rows)} samples, {len(studies)} "
                   + ("study" if len(studies) == 1 else "studies"),
                   "mean within-study Spearman rho    (aerobic  <-->  low oxygen)", fontsize=9, xlim=(-1, 1))
    fig.tight_layout()
    savefig(fig, out, "fig_oxygen_axis_genes", formats)

    pd.DataFrame({"gene": rho.index, "mean_within_study_rho": rho.values}).sort_values("mean_within_study_rho", ascending=False)\
        .to_csv(out / "oxygen_axis_genes.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame({"sample": np.asarray(ctx.ids)[rows], "gse": study, "oxygen": oxygen[rows], "oxygen_class": np.where(y == 1, "low", "aerobic"),
                  "lda_score": score, "lda_score_leave_one_out": loo, "split": ctx.split[rows]})\
        .to_csv(out / "oxygen_samples.tsv", sep="\t", index=False, float_format="%.6g")
    result = {"studies": studies, "n_samples": int(len(rows)), "embedding_dim": int(E.shape[1]), "pooled_auroc": auroc,
              "auroc_leave_one_out": auroc_loo, "per_study": per_study, "toward_low_oxygen": list(rho.tail(N_TOP).iloc[::-1].index),
              "toward_aerobic": list(rho.head(N_TOP).index)}
    (out / "oxygen_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result
