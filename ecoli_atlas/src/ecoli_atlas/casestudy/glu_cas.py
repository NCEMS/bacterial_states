"""Glucose (GLU) against casamino acids (CAS) in the nutrient-stress time course of Contreras et
al. (GSE271069), on the atlas embedding. CAS samples are those with ``amino_acids_supplied`` true;
samples of the two media in the same growth phase form a pair group.

The study is part of the atlas tables and a forced TEST study (``TrainConfig.test_studies``); the
metrics say which split its samples are in, so a bundle fitted on them shows it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..figures.plotting import BLUE, RED, correlate_barh, savefig, spearman_series

GLU_CAS_STUDY = "GSE271069"
POS, NEG = "GLU", "CAS"
N_GENES_PAIRED = 15
N_TOP = 10
SPLIT_NAMES = {1: "training_pool", 3: "test", 0: "projected"}


def glu_cas_design(ctx, study: str = GLU_CAS_STUDY) -> pd.DataFrame:
    rows = np.where(ctx.gse == study)[0]
    if not len(rows):
        raise ValueError(f"no sample of {study} in the tables")
    amino = ctx.column("amino_acids_supplied")[rows]
    d = pd.DataFrame({"sample": np.asarray(ctx.ids)[rows], "row": rows, "treatment": np.where(amino == "true", NEG, np.where(amino == "false", POS, "")),
                      "pair_group": ctx.column("growth_phase")[rows], "split": ctx.split[rows]})
    d = d[d["treatment"] != ""]
    if d["treatment"].nunique() < 2:
        raise ValueError(f"{study}: both media are needed (amino_acids_supplied true and false)")
    return d.reset_index(drop=True)


def run_glu_cas(ctx, out_dir, formats=("pdf", "png"), study: str = GLU_CAS_STUDY, n_perm: int = 1000, log=None) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from .axis import center_by_group, oriented_lda_score, paired_axis_test, perm_p_text
    log = log or ctx.log
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d_all = glu_cas_design(ctx, study)
    in_split = {SPLIT_NAMES[int(k)]: int(v) for k, v in d_all["split"].value_counts().items()}
    result = {"gse": study, "n_samples": int(len(d_all)), "samples_by_split": in_split,
              "fitted_by_the_model": bool((d_all["split"] == 1).any())}
    if result["fitted_by_the_model"]:
        log(f"[glu-cas] {study} is in the training pool: the embedding was fitted on these samples (not an outside test)")

    # paired within growth phase: leave-one-group-out axis and its permutation null
    n_media = d_all[d_all["pair_group"] != ""].groupby("pair_group")["treatment"].nunique()
    d = d_all[d_all["pair_group"].isin(n_media.index[n_media == 2])].copy()
    if d["pair_group"].nunique() < 2:
        result["paired"] = {"skipped": "fewer than two growth phases with both media"}
    else:
        grp, y = d["pair_group"].to_numpy(), (d["treatment"].to_numpy() == POS).astype(int)
        test = paired_axis_test(center_by_group(ctx.z[d["row"].to_numpy()].astype(float), grp), y, grp, n_perm=n_perm, seed=ctx.seed)
        log(f"[glu-cas] {test['n_pairs']} growth phases with both media, {len(d)} samples; leave-one-phase-out: {POS} above "
            f"{NEG} in {test['correct']}/{test['n_pairs']} (sign p={test['p_sign']:.4f}, permutation p={test['p_perm']:.4f})")
        d["axis_oof"] = test["axis_oof"]
        corr = spearman_series(ctx.logx[d["row"].to_numpy()], d["axis_oof"].to_numpy(), ctx.gene_names).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        correlate_barh(ax, corr, N_GENES_PAIRED,
                       f"Glucose (red) vs Casamino Acids (blue)\nn={len(d)}, {test['n_pairs']} growth phases | "
                       f"{test['correct']}/{test['n_pairs']}, permutation {perm_p_text(test)}",
                       "Spearman rho vs out-of-fold condition axis")
        fig.tight_layout()
        savefig(fig, out, "fig_glu_cas_axis", formats)
        d.drop(columns=["row"]).to_csv(out / "glu_cas_paired_samples.tsv", sep="\t", index=False, float_format="%.6g")
        result["paired"] = {"n_samples": int(len(d)), "groups": sorted(d["pair_group"].unique()),
                            **{k: v for k, v in test.items() if k != "axis_oof"},
                            f"toward_{POS}": list(corr.tail(N_GENES_PAIRED).iloc[::-1].index),
                            f"toward_{NEG}": list(corr.head(N_GENES_PAIRED).index)}

    # one discriminant on every sample of the two media
    rows = d_all["row"].to_numpy()
    y = (d_all["treatment"].to_numpy() == POS).astype(int)
    score = oriented_lda_score(ctx.z[rows].astype(float), y)
    auroc = float(roc_auc_score(y, score))
    rho = spearman_series(ctx.logx[rows], score, ctx.gene_names).sort_values()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    rng = np.random.RandomState(0)
    span = np.abs(score).max() * 1.25
    for cls, color, label in ((0, BLUE, NEG), (1, RED, POS)):
        m = y == cls
        ax.scatter(score[m], rng.uniform(-0.04, 0.04, size=int(m.sum())), s=85, c=color, marker="o", edgecolor="k", linewidth=.5,
                   alpha=0.9, label=label, zorder=3)
        ax.plot([score[m].mean()] * 2, [-0.12, 0.15], color=color, lw=3, zorder=4)
    ax.axvline(0, color="grey", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([0])
    ax.set_yticklabels([f"{len(rows)} samples"], fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel(f"LDA discriminant score    (Casamino Acids, {NEG}  <-->  Glucose, {POS})", fontsize=12)
    ax.set_title(f"{POS} vs {NEG} discriminant projection (in-sample AUROC={auroc:.2f})", fontsize=12, pad=30)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    savefig(fig, out, "fig_glu_cas_lda_projection", formats)

    fig, ax = plt.subplots(figsize=(8, 5))
    correlate_barh(ax, rho, N_TOP, f"Genes along the {POS} vs {NEG} axis", f"Spearman rho ({NEG} <--> {POS})", fontsize=9, xlim=(-1.05, 1.05))
    fig.tight_layout()
    savefig(fig, out, "fig_glu_cas_axis_genes", formats)

    pd.DataFrame({"gene": rho.index, "spearman_rho": rho.values}).sort_values("spearman_rho", ascending=False)\
        .to_csv(out / "glu_cas_gene_correlations.tsv", sep="\t", index=False, float_format="%.6f")
    d_all.drop(columns=["row"]).assign(lda_score=score).to_csv(out / "glu_cas_lda_scores.tsv", sep="\t", index=False, float_format="%.6g")
    result["lda"] = {"n_samples": int(len(rows)), "embedding_dim": int(ctx.z.shape[1]), "pooled_auroc": auroc,
                     "treatments": {str(k): int(v) for k, v in d_all["treatment"].value_counts().items()},
                     f"toward_{POS}": list(rho.tail(N_TOP).iloc[::-1].index), f"toward_{NEG}": list(rho.head(N_TOP).index)}
    (out / "glu_cas_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result
