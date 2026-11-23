"""Urinary tract infection against urine culture, paired by patient isolate (GSE128997), on the
atlas embedding. The isolate is the metadata's ``strain_name``; ``in_host`` tells the sample taken
from the patient (UTI) from the same isolate grown in urine (UR)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..figures.plotting import spearman_series
from .pathways import pathway_analysis

UTI_STUDY = "GSE128997"
POS, NEG = "UTI", "UR"
N_GENES_SHOWN = 15


def uti_design(ctx, study: str = UTI_STUDY) -> pd.DataFrame:
    """One row per sample of the study whose isolate has both conditions (``row`` = position in the tables)."""
    rows = np.where(ctx.gse == study)[0]
    if not len(rows):
        raise ValueError(f"no sample of {study} in the tables")
    if (ctx.split[rows] == 1).any():
        raise ValueError(f"{study} is in the training pool; the analysis would be circular")
    d = pd.DataFrame({"sample": np.asarray(ctx.ids)[rows], "row": rows, "patient": ctx.column("strain_name")[rows],
                      "cond": np.where(ctx.column("in_host")[rows] == "true", POS, NEG), "split": ctx.split[rows]})
    d = d[d["patient"] != ""]
    per = d.groupby("patient")["cond"].agg(["nunique", "size"])
    d = d[d["patient"].isin(per.index[(per["nunique"] == 2) & (per["size"] == 2)])]
    if d["patient"].nunique() < 3:
        raise ValueError(f"{study}: fewer than three isolates with one sample of each condition")
    return d.sort_values(["patient", "cond"]).reset_index(drop=True)


def run_uti(ctx, out_dir, formats=("pdf", "png"), study: str = UTI_STUDY, n_perm: int = 1000, log=None, gsea: bool = False,
            modules=None, gene_sets=None, permutations: int = 10000, sides: str = "two") -> dict:
    from .axis import center_by_group, paired_axis_test
    log = log or ctx.log
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = uti_design(ctx, study)
    rows, grp = d["row"].to_numpy(), d["patient"].to_numpy()
    y = (d["cond"].to_numpy() == POS).astype(int)
    n_all = int((ctx.gse == study).sum())
    log(f"[uti] {study}: {len(d)} of {n_all} samples in {d['patient'].nunique()} paired isolates")
    test = paired_axis_test(center_by_group(ctx.z[rows].astype(float), grp), y, grp, n_perm=n_perm, seed=ctx.seed)
    log(f"[uti] leave-one-patient-out: {POS} above its own {NEG} in {test['correct']}/{test['n_pairs']} patients "
        f"(sign p={test['p_sign']:.4f}, Wilcoxon p={test['p_wilcoxon']:.4f}); permutation null median "
        f"{test['null_median']:.1f}, p={test['p_perm']:.4f}")
    d["axis_oof"] = test["axis_oof"]
    corr = spearman_series(ctx.logx[rows], d["axis_oof"].to_numpy(), ctx.gene_names).sort_values()

    pd.DataFrame({"gene": corr.index, "spearman_rho": corr.values}).sort_values("spearman_rho", ascending=False)\
        .to_csv(out / "uti_axis_genes.tsv", sep="\t", index=False, float_format="%.6f")
    d.drop(columns=["row"]).to_csv(out / "uti_samples.tsv", sep="\t", index=False, float_format="%.6g")
    result = {"gse": study, "n_samples_in_study": n_all, "n_samples": int(len(d)), "n_patients": int(d["patient"].nunique()),
              "conditions": {str(k): int(v) for k, v in d["cond"].value_counts().items()},
              **{k: v for k, v in test.items() if k != "axis_oof"},
              f"toward_{POS}": list(corr.tail(N_GENES_SHOWN).iloc[::-1].index), f"toward_{NEG}": list(corr.head(N_GENES_SHOWN).index)}
    result |= pathway_analysis(corr, out, "uti", POS, f"{NEG} (urine culture)", formats, gsea, modules, gene_sets, permutations, log,
                               axis_label=f"{POS} vs urine culture", sides=sides)
    (out / "uti_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result
