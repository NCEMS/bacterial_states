"""Write the GC content of every feature of an inGEST counting annotation (package data of ecoli_atlas).

    python ecoli_atlas/scripts/feature_gc.py \
        --features resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv \
        --genome resources/annotation/GCF_000005845.2_ASM584v2_genomic.fna \
        --out ecoli_atlas/src/ecoli_atlas/reference/feature_gc_counting_v2.tsv

Columns Feature_ID, Length (as in the features table, to check that a bundle was trained on the same
annotation), GC (share of G + C over the feature's span, Start..End, 1-based inclusive) and the window
profile w0.20 .. w0.80: the share of the feature's 300 bp windows (centred every 50 bp along the feature,
clipped to the contig) whose GC falls in each 0.01 bin (`technical.GC_BINS`; windows outside 0.20-0.80 go
to the end bins). The inGEST GC fit (gc_bias.py) uses 300 bp windows, so the offset of a feature is the
mean of the fitted curve over its windows (`technical.gc_offset`).
"""
import argparse

import numpy as np
import pandas as pd

WINDOW, STEP = 300, 50
BINS = np.round(np.arange(0.20, 0.8001, 0.01), 2)


def read_fasta(path: str) -> dict:
    seqs, name, parts = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(parts).upper()
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        seqs[name] = "".join(parts).upper()
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    feat = pd.read_csv(a.features, sep="\t")
    genome = read_fasta(a.genome)
    cums = {}
    for name, seq in genome.items():
        b = np.frombuffer(seq.encode(), dtype=np.uint8)
        cums[name] = np.concatenate([[0.0], np.cumsum((b == ord("G")) | (b == ord("C")))])
    gc, prof = [], []
    for r in feat.itertuples():
        cs = cums[r.Chr]
        n = len(cs) - 1
        lo0, hi0 = int(r.Start) - 1, int(r.End)
        gc.append((cs[hi0] - cs[lo0]) / (hi0 - lo0))
        centre = np.arange(lo0, hi0, STEP) + STEP // 2
        centre = centre[centre < hi0] if (centre < hi0).any() else np.array([(lo0 + hi0) // 2])
        lo, hi = np.clip(centre - WINDOW // 2, 0, n), np.clip(centre + WINDOW // 2, 0, n)
        w = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
        k = np.clip(np.rint((w - BINS[0]) / 0.01).astype(int), 0, len(BINS) - 1)
        prof.append(np.bincount(k, minlength=len(BINS)) / len(k))
    out = pd.DataFrame({"Feature_ID": feat["Feature_ID"], "Length": feat["Length"], "GC": gc})
    out = pd.concat([out, pd.DataFrame(np.array(prof), columns=[f"w{b:.2f}" for b in BINS])], axis=1)
    out["GC"] = out["GC"].map(lambda v: f"{v:.5f}")
    out.to_csv(a.out, sep="\t", index=False, float_format="%.4g")
    print(f"wrote {len(out)} features to {a.out} (mean GC {np.mean(gc):.4f})")


if __name__ == "__main__":
    main()
