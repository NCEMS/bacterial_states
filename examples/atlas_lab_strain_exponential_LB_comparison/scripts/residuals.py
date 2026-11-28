"""results/residuals.tsv: each sample's residual from the line through the study means of every regulon against the
ribosomal proteins (one line for all studies, as in fig_batch_tracking; the studies of common.EXCLUDE_STUDIES are
left out of the line but get residuals).
    python scripts/residuals.py"""
import numpy as np, pandas as pd
from common import SAMPLES, RESULTS, SETS, EXCLUDE_STUDIES


def residuals(s):
    t = s[~s.gse.isin(EXCLUDE_STUDIES)].groupby(["strain", "gse"])[["ribosome"] + SETS].mean()
    out = s[["gsm", "gse", "strain", "strain_name", "ribosome"]].copy()
    for k in SETS:
        b, a = np.polyfit(t["ribosome"], t[k], 1)
        out[f"res_{k}"] = s[k] - (a + b * s["ribosome"])
        out[k] = s[k]
    return out


def trend(set_name):
    """slope, intercept of the line of `residuals` for one set"""
    s = pd.read_csv(SAMPLES, sep="\t")
    t = s[~s.gse.isin(EXCLUDE_STUDIES)].groupby(["strain", "gse"])[["ribosome", set_name]].mean()
    return np.polyfit(t["ribosome"], t[set_name], 1)


def main():
    RESULTS.mkdir(exist_ok=True)
    s = pd.read_csv(SAMPLES, sep="\t")
    residuals(s).to_csv(RESULTS / "residuals.tsv", sep="\t", index=False, float_format="%.3f")
    print(f"residuals.tsv: {len(s)} samples")


if __name__ == "__main__":
    main()
