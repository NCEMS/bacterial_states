"""Write the GC content of every feature of an inGEST counting annotation (package data of ecoli_atlas).

    python ecoli_atlas/scripts/feature_gc.py \
        --features resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv \
        --genome resources/annotation/GCF_000005845.2_ASM584v2_genomic.fna \
        --out ecoli_atlas/src/ecoli_atlas/reference/feature_gc_counting_v2.tsv

Columns Feature_ID, Length (as in the features table, to check that a bundle was trained on the same
annotation) and GC (share of G + C over the feature's span, Start..End, 1-based inclusive). Used by
`embed --corrected-counts` / `--decoded-counts` for the GC offset of every feature (`technical.py`).
"""
import argparse

import pandas as pd


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
    gc = []
    for r in feat.itertuples():
        seq = genome[r.Chr][int(r.Start) - 1:int(r.End)]
        gc.append((seq.count("G") + seq.count("C")) / len(seq))
    out = pd.DataFrame({"Feature_ID": feat["Feature_ID"], "Length": feat["Length"], "GC": gc})
    out.to_csv(a.out, sep="\t", index=False, float_format="%.5f")
    print(f"wrote {len(out)} features to {a.out} (mean GC {out['GC'].mean():.4f})")


if __name__ == "__main__":
    main()
