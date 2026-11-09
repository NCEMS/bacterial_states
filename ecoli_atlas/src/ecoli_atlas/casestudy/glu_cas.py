"""Glucose (GLU) vs casamino acids (CAS) in the Contreras et al. count table (inference notebook
cells 19-27). The samples are not in the atlas: they are embedded with the bundle first. The
metadata layout is the SRA run table of that study (``metadata_paper1.csv``): index = SRR run,
columns ``treatment`` ("GLU condition. timepoint S2") and ``BioProject``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..embed import embed_counts
from ..preprocess import GeneMap, clean_counts, cpm
from ..figures.plotting import BLUE, RED, correlate_barh, savefig, spearman_series

POS, NEG = "GLU", "CAS"
N_GENES_PAIRED = 15
N_TOP = 10
TREATMENT_COL, PROJECT_COL = "treatment", "BioProject"


def glu_cas_design(meta: pd.DataFrame) -> pd.DataFrame:
    if TREATMENT_COL not in meta.columns:
        raise ValueError(f"case metadata lacks the {TREATMENT_COL!r} column (metadata_paper1.csv layout expected)")
    first = meta[TREATMENT_COL].astype(str).str.split(" ").str[0]
    d = meta[first.isin([POS, NEG]).fillna(False)].copy()
    if d.empty:
        raise ValueError(f"no {POS}/{NEG} samples in the case metadata")
    d["short_treatment"] = d[TREATMENT_COL].astype(str).str.split(" ").str[0]
    project = d[PROJECT_COL].astype(str) if PROJECT_COL in d.columns else pd.Series("project", index=d.index)
    d["pair_group"] = project + "_" + d[TREATMENT_COL].astype(str).str.split(" ").str[-1]
    return d


def run_glu_cas(counts: pd.DataFrame, case_meta: pd.DataFrame, bundle, feat: pd.DataFrame | None, out_dir,
                formats=("pdf", "png"), seed: int = 77, n_perm: int = 1000, log=None) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from .axis import center_by_group, oriented_lda_score, paired_axis_test
    log = log or (lambda *_: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gm = GeneMap.from_feature_table(feat) if feat is not None else None
    res = embed_counts(counts, bundle, gm)
    cov = res.coverage
    log(f"[glu-cas] embedded {len(res.embedding)} samples; gene coverage {cov.n_present}/{cov.n_model_genes}")
    E_all = res.embedding
    cpm_all = cpm(clean_counts(counts, gm) if gm is not None else counts.T.copy())
    cpm_all.index = cpm_all.index.astype(str)
    meta = case_meta.copy()
    meta.index = meta.index.astype(str).str.strip()
    missing = [s for s in meta.index if s not in E_all.index]
    if missing:
        log(f"[glu-cas] {len(missing)} metadata samples not in the count table (ignored), e.g. {missing[:3]}")
    meta = meta.loc[[s for s in meta.index if s in E_all.index]]
    d_all = glu_cas_design(meta)
    result = {"n_samples_embedded": int(len(E_all)), "gene_coverage": cov.fraction}

    # --- paired within (BioProject, timepoint): leave-one-group-out axis + permutation null (cell 23)
    pair_counts = d_all.groupby("pair_group")["short_treatment"].nunique()
    d = d_all[d_all["pair_group"].isin(pair_counts[pair_counts == 2].index)].copy()
    if d.empty or d["pair_group"].nunique() < 2:
        log(f"[glu-cas] not enough paired {POS}/{NEG} groups for the paired analysis; treatments: "
            f"{meta[TREATMENT_COL].value_counts().to_dict()}")
        result["paired"] = {"skipped": "fewer than two paired groups"}
    else:
        grp = d["pair_group"].values
        y = (d["short_treatment"].values == POS).astype(int)
        log(f"[glu-cas] {d['pair_group'].nunique()} paired groups, {len(d)} samples, "
            f"{d['short_treatment'].value_counts().to_dict()}")
        Ec = center_by_group(E_all.loc[d.index].values.astype(float), grp)
        test = paired_axis_test(Ec, y, grp, n_perm=n_perm, seed=seed)
        log(f"[glu-cas] leave-one-group-out: {POS} above its own {NEG} in {test['correct']}/{test['n_pairs']} groups "
            f"(sign p={test['p_sign']:.4f}, Wilcoxon p={test['p_wilcoxon']:.4f}); permutation null median "
            f"{test['null_median']:.1f}, p={test['p_perm']:.4f}")
        d["axis_oof"] = test["axis_oof"]
        corr = spearman_series(cpm_all.loc[d.index].values, d["axis_oof"].values, cpm_all.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        correlate_barh(ax, corr, N_GENES_PAIRED,
                       f"Glucose (red) vs Casamino Acids (blue)\nn={len(d)}, {test['n_pairs']} groups | "
                       f"{test['correct']}/{test['n_pairs']}, permutation p={test['p_perm']:.4f}",
                       "Spearman rho vs out-of-fold condition axis")
        fig.tight_layout()
        savefig(fig, out, "fig_glu_cas_axis", formats)
        d.drop(columns=[c for c in d.columns if c not in (TREATMENT_COL, PROJECT_COL, "short_treatment", "pair_group", "axis_oof")])\
            .to_csv(out / "glu_cas_paired_samples.tsv", sep="\t")
        result["paired"] = {"n_samples": int(len(d)), **{k: v for k, v in test.items() if k != "axis_oof"},
                            f"toward_{POS}": list(corr.tail(N_GENES_PAIRED).iloc[::-1].index),
                            f"toward_{NEG}": list(corr.head(N_GENES_PAIRED).index)}

    # --- one LDA axis on every GLU / CAS sample (cells 24-27)
    rows = d_all.index
    y = (d_all["short_treatment"].values == POS).astype(int)
    E = E_all.loc[rows].values.astype(float)
    score = oriented_lda_score(E, y)
    auroc = float(roc_auc_score(y, score))
    log(f"[glu-cas] one discriminant, {len(rows)} samples, {E.shape[1]}-dim embedding; pooled AUROC = {auroc:.3f}")
    rho = spearman_series(cpm_all.loc[rows].values, score, cpm_all.columns).sort_values()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    rng = np.random.RandomState(0)
    span = np.abs(score).max() * 1.25
    for grp_, color, label in [(0, BLUE, NEG), (1, RED, POS)]:
        m = y == grp_
        ax.scatter(score[m], [0] * int(m.sum()) + rng.uniform(-0.04, 0.04, size=int(m.sum())), s=85, c=color, marker="o",
                   edgecolor="k", linewidth=.5, alpha=0.9, label=label, zorder=3)
        ax.plot([score[m].mean()] * 2, [-0.12, 0.15], color=color, lw=3, zorder=4)
    ax.axvline(0, color="grey", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([0])
    ax.set_yticklabels(["All Samples"], fontsize=10)
    ax.set_xlabel("LDA discriminant score (shared treatment axis)", labelpad=15)
    ax.set_title(f"{POS} vs {NEG} Discriminant Projection (AUROC={auroc:.2f})", pad=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
    ax.text(0.01, -0.25, f"Casamino Acids ({NEG})", transform=ax.transAxes, fontsize=11, color=BLUE, weight="bold", ha="left")
    ax.text(0.99, -0.25, f"Glucose ({POS})", transform=ax.transAxes, fontsize=11, color=RED, weight="bold", ha="right")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    savefig(fig, out, "fig_glu_cas_lda_projection", formats)

    fig, ax = plt.subplots(figsize=(8, 5))
    correlate_barh(ax, rho, N_TOP, f"Genes along the shared {POS} vs {NEG} axis", f"Spearman rho ({NEG} <--> {POS})",
                   fontsize=9, xlim=(-1.05, 1.05))
    fig.tight_layout()
    savefig(fig, out, "fig_glu_cas_axis_genes", formats)

    gene_corr = pd.DataFrame({"Spearman_Rho": rho.values}, index=pd.Index(rho.index, name="Gene_Symbol"))
    gene_corr.sort_values("Spearman_Rho", ascending=False).to_csv(out / "glu_cas_gene_correlations.csv")
    pd.DataFrame({"Sample_ID": rows, "Treatment": d_all["short_treatment"].values,
                  "Detailed_Treatment": d_all[TREATMENT_COL].values, "LDA_Discriminant_Score": score})\
        .to_csv(out / "glu_cas_lda_scores.csv", index=False)
    result["lda"] = {"n_samples": int(len(rows)), "embedding_dim": int(E.shape[1]), "pooled_auroc": auroc,
                     "treatments": {str(k): int(v) for k, v in d_all["short_treatment"].value_counts().items()},
                     f"toward_{POS}": list(rho.tail(N_TOP).iloc[::-1].index), f"toward_{NEG}": list(rho.head(N_TOP).index)}
    (out / "glu_cas_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result
