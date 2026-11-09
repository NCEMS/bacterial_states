"""UTI vs urine (UR), paired within patient, on the full atlas embedding (notebook cell 46).

The samples carry no condition annotation in the metadata; identity comes from the curated GSM list
and titles below (GSE128997). Sequencing runs of one GSM are averaged before the analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..labels import MULTIRUN_POLICY, agg_rows, rows_of_gsm
from ..figures.plotting import correlate_barh, savefig, spearman_series

UTI_GSM_RANGE = (3689802, 3689829)
UTI_GSMS = tuple(f"GSM{n}" for n in range(UTI_GSM_RANGE[0], UTI_GSM_RANGE[1] + 1))
UTI_TITLES = ("HM01_UTI", "HM03_UTI", "HM06_UTI", "HM07_UTI", "HM14_UTI", "HM17_UTI", "HM43_UTI",
              "HM54_UTI", "HM56_UTI", "HM57_UTI", "HM60_UTI", "HM66_UTI", "HM68_UTI", "HM86_UTI",
              "HM01_UR", "HM03_UR", "HM06_UR", "HM07_UR", "HM14_UR", "HM17_UR", "HM43_UR",
              "HM54_UR", "HM56_UR", "HM57_UR", "HM60_UR", "HM66_UR", "HM68_UR", "HM86_UR")
N_GENES_SHOWN = 15


def uti_design(meta: pd.DataFrame, gsms=UTI_GSMS, titles=UTI_TITLES, pos_label="UTI", neg_label="UR") -> pd.DataFrame:
    """One row per GSM found in ``meta`` with patient / condition parsed from the title; keeps only
    patients with both conditions."""
    if len(gsms) != len(titles):
        raise ValueError("gsms and titles must have the same length")
    rows_of = rows_of_gsm(meta.index)
    d = pd.DataFrame({"gsm": list(gsms), "title": list(titles)})
    d["rows"] = d["gsm"].map(lambda g: rows_of.get(g, []))
    d = d[d["rows"].str.len() > 0].copy()
    if d.empty:
        raise ValueError("none of the case-study GSM accessions is present in the metadata")
    d["row"] = d["rows"].str[0]
    d["patient"] = d["title"].str.extract(r"^(.+?)_[^_]+$")[0]
    d["cond"] = d["title"].str.extract(rf"_({pos_label}|{neg_label})$")[0]
    if d[["patient", "cond"]].isna().any().any():
        raise ValueError("every title must look like <patient>_<condition>")
    if not all(len(set(meta["gse"].values[r])) == 1 for r in d["rows"]):
        raise ValueError("a GSM spans several studies")
    gses = set(meta["gse"].astype(str).values[d["row"].to_numpy()])
    if len(gses) != 1:
        raise ValueError(f"case-study samples span several studies: {sorted(gses)}")
    if 1 in set(meta["split"].values[d["row"].to_numpy()]):
        raise ValueError("case-study samples are in TRAIN; the analysis would be circular")
    paired = d.groupby("patient")["cond"].nunique() == 2
    d = d[d["patient"].isin(paired[paired].index)].sort_values(["patient", "cond"]).reset_index(drop=True)
    d.attrs["gse"] = next(iter(gses))
    return d


def run_uti(ctx, out_dir, formats=("pdf", "png"), gsms=UTI_GSMS, titles=UTI_TITLES, pos_label="UTI", neg_label="UR",
            n_perm: int = 1000, log=None) -> dict:
    import matplotlib.pyplot as plt
    from .axis import center_by_group, paired_axis_test
    log = log or ctx.log
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = ctx.meta
    d = uti_design(meta, gsms, titles, pos_label, neg_label)
    patients = sorted(d["patient"].unique())
    log(f"[uti] {d.attrs['gse']}: {len(d)} samples, {len(patients)} paired patients, {d['cond'].value_counts().to_dict()} "
        f"(runs per GSM averaged: {MULTIRUN_POLICY})")
    grp = d["patient"].values
    y = (d["cond"].values == pos_label).astype(int)
    row_lists = list(d["rows"])
    Ec = center_by_group(agg_rows(ctx.embedding, row_lists).astype(float), grp)
    test = paired_axis_test(Ec, y, grp, n_perm=n_perm, seed=ctx.seed)
    log(f"[uti] leave-one-patient-out: {pos_label} above its own {neg_label} in {test['correct']}/{test['n_pairs']} patients "
        f"(sign p={test['p_sign']:.4f}, Wilcoxon p={test['p_wilcoxon']:.4f}); permutation null median "
        f"{test['null_median']:.1f}, p={test['p_perm']:.4f}")
    d["axis_oof"] = test["axis_oof"]
    cpm_case = agg_rows(ctx.data.cpm, row_lists)
    corr = spearman_series(cpm_case, d["axis_oof"].values, ctx.genes).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    correlate_barh(ax, corr, N_GENES_SHOWN,
                   f"{pos_label} (red) vs {neg_label} (blue)\nn={len(d)}, {len(patients)} patients | "
                   f"{test['correct']}/{test['n_pairs']}, permutation p={test['p_perm']:.4f}",
                   "Spearman rho vs out-of-fold condition axis")
    fig.tight_layout()
    savefig(fig, out, "fig_uti_axis", formats)

    pd.DataFrame({"gene": corr.index, "spearman_rho": corr.values}).sort_values("spearman_rho", ascending=False)\
        .to_csv(out / "uti_axis_genes.tsv", sep="\t", index=False, float_format="%.6f")
    d.drop(columns=["rows"]).to_csv(out / "uti_samples.tsv", sep="\t", index=False)
    result = {"gse": d.attrs["gse"], "n_samples": int(len(d)), "n_patients": len(patients),
              "conditions": {str(k): int(v) for k, v in d["cond"].value_counts().items()},
              **{k: v for k, v in test.items() if k != "axis_oof"},
              f"toward_{pos_label}": list(corr.tail(N_GENES_SHOWN).iloc[::-1].index),
              f"toward_{neg_label}": list(corr.head(N_GENES_SHOWN).index)}
    (out / "uti_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result
