"""Lineage from the genotypes, associations within lineage, and the IS1H state, on the tables of scripts/collect.py.

Many top associations with the FlhDC residual are BW25113-lineage markers, and some samples labelled MG1655 carry
them. So the lineage is read from the genotypes:
  markers        mutation keys called in >= 80% of the covered BW25113-labelled samples and in <= 10% of the covered
                 MG1655-labelled ones (non-mixed samples, >= 3 / >= 5 covered)
  lineage        K-12 samples (and lab strains without a strain name): BW25113-like where >= 70% of >= 3 covered markers are called, MG1655-like where
                 <= 20% are, otherwise `unclear` (`no_info` with < 3 covered); other strains keep their strain name
Writes into --results:
  lineage.tsv                  per sample: label, markers covered / called, lineage
  lineage_markers.tsv          the markers
  associations_by_lineage.tsv  the tests of collect.py repeated within each lineage (MG1655-like, BW25113-like)
Samples of excluded studies (sample_qc `excluded`) get a lineage but are left out of the markers and the tests.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

from common import RESULTS
from common import SETS
INFO = ["type", "position", "gene_name", "gene_position", "mutation_category", "aa_ref_seq", "aa_position",
        "aa_new_seq", "n_called", "n_covered"]


def lineage(geno, qc):
    mixed = set(qc.gsm[qc.mixed]) | set(qc.gsm[qc.excluded.fillna("") != ""])
    lab = qc.set_index("gsm").strain_name.fillna("")
    bw = [g for g in geno.columns if lab.get(g) == "BW25113" and g not in mixed]
    mg = [g for g in geno.columns if lab.get(g) == "MG1655" and g not in mixed]
    def share(cols):
        v = geno[cols]
        return (v >= 0.5).sum(axis=1) / v.notna().sum(axis=1).replace(0, np.nan), v.notna().sum(axis=1)
    fb, nb = share(bw); fm, nm = share(mg)
    markers = geno.index[(nb >= 3) & (fb >= 0.8) & (nm >= 5) & (fm <= 0.1)]
    v = geno.loc[markers]
    cov, called = v.notna().sum(), (v >= 0.5).sum()
    out = pd.DataFrame({"gsm": geno.columns, "strain_name": [lab.get(g, "") for g in geno.columns],
                        "markers_covered": cov.values, "markers_called": called.values})
    out["marker_share"] = out.markers_called / out.markers_covered.replace(0, np.nan)
    strain = qc.set_index("gsm").strain
    def call(r):
        if strain.get(r.gsm) not in ("MG1655", "BW25113") and r.strain_name not in ("BW27786", "", "nan"):
            return r.strain_name
        if r.markers_covered < 3:
            return "no_info"
        return "BW25113-like" if r.marker_share >= 0.7 else "MG1655-like" if r.marker_share <= 0.2 else "unclear"
    out["lineage"] = out.apply(call, axis=1)
    out["mixed"] = out.gsm.isin(set(qc.gsm[qc.mixed]))
    out["excluded"] = out.gsm.isin(set(qc.gsm[qc.excluded.fillna("") != ""]))
    mk = pd.DataFrame({"share_BW25113": fb[markers], "n_BW25113": nb[markers], "share_MG1655": fm[markers],
                       "n_MG1655": nm[markers]})
    return out, mk


def tests(geno, res, gse_of, samples, label):
    r = res.set_index("gsm").loc[samples]
    rows = []
    for key, v in geno[samples].iterrows():
        carrier, non = v[v >= 0.5].index, v[v == 0].index
        if len(carrier) < 2 or len(non) < 5:
            continue
        row = {"lineage": label, "key": key, "n_carriers": len(carrier), "n_noncarriers": len(non),
               "carrier_studies": ",".join(sorted({gse_of[g] for g in carrier}))}
        row["n_carrier_studies"] = len(row["carrier_studies"].split(","))
        for k in ["ribosome"] + SETS:
            col = k if k == "ribosome" else f"res_{k}"
            a, b = r.loc[carrier, col], r.loc[non, col]
            row[f"diff_{k}"] = a.mean() - b.mean()
            row[f"p_{k}"] = stats.mannwhitneyu(a, b).pvalue
            st = pd.DataFrame({"gse": [gse_of[g] for g in list(carrier) + list(non)],
                               "c": [1] * len(a) + [0] * len(b), "y": list(a) + list(b)}).groupby("gse").mean()
            sa, sb = st[st.c >= 0.5].y, st[st.c < 0.5].y
            row[f"pstudy_{k}"] = stats.mannwhitneyu(sa, sb).pvalue if len(sa) > 1 and len(sb) > 1 else np.nan
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(RESULTS))
    a = p.parse_args()
    geno = pd.read_csv(f"{a.results}/genotypes.tsv", sep="\t", index_col=0)
    geno = geno.drop(columns=[c for c in INFO if c in geno.columns])
    qc = pd.read_csv(f"{a.results}/sample_qc.tsv", sep="\t")
    res = pd.read_csv(f"{a.results}/residuals.tsv", sep="\t")
    gse_of = dict(zip(qc.gsm, qc.gse))
    lin, mk = lineage(geno, qc)
    lin.to_csv(f"{a.results}/lineage.tsv", sep="\t", index=False, float_format="%.2f")
    info = pd.read_csv(f"{a.results}/genotypes.tsv", sep="\t", index_col=0, usecols=["key"] + INFO[:8])
    info.join(mk, how="inner").to_csv(f"{a.results}/lineage_markers.tsv", sep="\t", float_format="%.2f")
    print(f"{len(mk)} BW25113-lineage markers")
    print(pd.crosstab(lin.strain_name, lin.lineage).to_string())
    rows = []
    for label in ["MG1655-like", "BW25113-like"]:
        s = [g for g in lin.gsm[(lin.lineage == label) & ~lin.mixed & ~lin.excluded] if g in set(res.gsm)]
        rows += tests(geno, res, gse_of, s, label)
    asc = pd.DataFrame(rows)
    if len(asc):
        asc["lineage_marker"] = asc.key.isin(mk.index)
        asc = asc.sort_values("p_FlhDC")
    asc.to_csv(f"{a.results}/associations_by_lineage.tsv", sep="\t", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
