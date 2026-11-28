"""One row per sample with every state found, from the tables of scripts/collect.py and scripts/stratify.py and the
breseq BAMs: sample_states.tsv in the example folder (tracked; the input of the figures with results/residuals.tsv).

Columns
  gsm, gse, strain, strain_name, excluded       sample labels; `excluded`: reason the study is left out of the analysis
  n_calls, n_clustered, mixed, junction_k50     breseq calls (sample_qc.tsv)
  lineage, markers_covered, markers_called      lineage from the BW25113 markers (stratify.py)
  IS1H_state, IS1H_reads, depth_flhD, upTSS_frac           flhDC upstream IS1H (loci_states.tsv)
  IS1I_state, IS1I_reads, depth_crl                        IS1I in crl: absent = crl intact
  IS5U_state, IS5U_reads, depth_oppA                       IS5U between ychE and oppA
  fim_state, fim_on_frac, fim_on_reads, fim_off_reads, depth_fimA   fim switch (reference orientation = OFF)
  glpR_3560455_insG, gatC_2173363_del2          stock alleles that split MG1655: 1 called, 0 covered and not called
                                                (reference allele), empty without coverage
  mg1655_stock                                  MG1655-like samples, from the informative loci among crl, IS5U, glpR
                                                and gatC (Freddolino et al. 2012, J Bacteriol 194:303): CGSC7740-type
                                                (crl::IS1 = IS1I present, IS5U present, the glpR and gatC frameshifts; the state of
                                                ATCC 700926 / CGSC 7740 and of the reference NC_000913.3) or
                                                ATCC47076-type (crl intact, no IS5U, glpR +G and gatC 2-bp deletion against
                                                NC_000913.3: the older isolate ATCC 47076 / CGSC 6300, and the state of
                                                BW25113, MG1693); `conflict` when
                                                they disagree. IS1H (flhDC)
                                                is not used: it differs between stocks on its own
  rpoS_codon33, rpoS_codon33_frac               main codon 33 in the reads (reference CAG, Gln) and its share
  rpoS_inside_over_nlpD                         mean depth inside rpoS (2866600-2867500) / nlpD (2867700-2868700)
  rpoS_status                                   deleted (ratio < 0.15 with nlpD >= 10x; ΔrpoS 0.01-0.2, wild type >= 0.28), frameshift / nonsense (a call
                                                in rpoS), intact, no_info (nlpD < 10x)
  target_variants                               calls in the target genes (or 150 bp upstream), clustered calls left
                                                out: gene:change; ...
    python scripts/states.py [--results results] [--breseq $SCR/breseq]
"""
import argparse, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

from common import SCR, RESULTS, STATES
C33 = (2867453, 2867455)                  # rpoS codon 33 (minus strand)
RPOS_IN, NLPD = (2866600, 2867500), (2867700, 2868700)
STOCK = {"glpR_3560455_insG": "INS:3560455:C>G", "gatC_2173363_del2": "DEL:2173363:2"}
BED = None
rc = lambda s: s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def rpos(bam):
    """codon 33 counts from reads aligned without gaps over it, and depth inside rpoS / nlpD"""
    codons = {}
    for line in run(["samtools", "view", bam, f"NC_000913:{C33[0]}-{C33[1]}"]).splitlines():
        f = line.split("\t")
        if not f[5].rstrip("M").isdigit():
            continue
        i = C33[0] - int(f[3])
        if 0 <= i and i + 3 <= len(f[9]):
            c = rc(f[9][i:i + 3])
            codons[c] = codons.get(c, 0) + 1
    lines = run(["samtools", "bedcov", BED, bam]).splitlines()
    cov = {name: int(l.split("\t")[3]) / (b - a + 1) for l, (name, (a, b)) in
           zip(lines, (("in", RPOS_IN), ("nlpD", NLPD)))}
    return codons, cov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--breseq", default=str(SCR / "breseq"))
    a = p.parse_args()
    R = a.results
    qc = pd.read_csv(f"{R}/sample_qc.tsv", sep="\t", keep_default_na=False)
    lin = pd.read_csv(f"{R}/lineage.tsv", sep="\t", keep_default_na=False)
    loci = pd.read_csv(f"{R}/loci_states.tsv", sep="\t")
    geno = pd.read_csv(f"{R}/genotypes.tsv", sep="\t", index_col=0)
    tg = pd.read_csv(f"{R}/target_genes.tsv", sep="\t")

    t = qc[["gsm", "gse", "strain", "strain_name", "excluded", "n_calls", "n_clustered", "mixed", "junction_k50"]]
    t = t.merge(lin[["gsm", "lineage", "markers_covered", "markers_called"]], on="gsm", how="left")
    t = t.merge(loci[["gsm", "IS1H_state", "IS1H_reads", "depth_flhD", "upTSS_frac", "IS1I_state", "IS1I_reads",
                      "depth_crl", "IS5U_state", "IS5U_reads", "depth_oppA", "fim_state", "fim_on_frac", "fim_on_reads", "fim_off_reads", "depth_fimA"]],
                on="gsm", how="left")
    for col, key in STOCK.items():
        t[col] = t.gsm.map(geno.loc[key]) if key in geno.index else np.nan

    global BED
    BED = f"{R}/_rpos.bed"
    with open(BED, "w") as f:
        for lo, hi in (RPOS_IN, NLPD):
            f.write(f"NC_000913\t{lo - 1}\t{hi}\n")
    def stock(r):
        if r.lineage != "MG1655-like":
            return ""
        votes = [{"present": "CGSC7740-type", "absent": "ATCC47076-type"}.get(r[c]) for c in ("IS1I_state", "IS5U_state")]
        votes += [{0: "CGSC7740-type", 1: "ATCC47076-type"}.get(r[c]) if pd.notna(r[c]) else None for c in STOCK]
        votes = {v for v in votes if v}
        return "no_info" if not votes else votes.pop() if len(votes) == 1 else "conflict"
    t["mg1655_stock"] = t.apply(stock, axis=1)
    bams = {g: f"{a.breseq}/{g}/data/reference.bam" for g in t.gsm}
    with ThreadPoolExecutor(16) as ex:
        rp = dict(zip(bams, ex.map(rpos, bams.values())))
    os.remove(BED)
    calls = {}
    for _, row in tg[tg.gene == "rpoS"].iterrows():
        for g in t.gsm:
            v = str(row.get(g, ""))
            calls[g] = [x for x in v.split("; ") if x and x != "." and not x.startswith(("nd(", "[cl]"))]
    c33, c33f, ratio, status = [], [], [], []
    for g in t.gsm:
        codons, cov = rp[g]
        n = sum(codons.values())
        top = max(codons, key=codons.get) if n else ""
        c33.append(top); c33f.append(codons[top] / n if n else np.nan)
        r = (cov["in"] + 1) / (cov["nlpD"] + 1)
        ratio.append(r)
        cl = calls.get(g, [])
        if cov["nlpD"] >= 10 and r < 0.15:
            status.append("deleted")
        elif any(x.startswith(("+", "Δ")) for x in cl):
            status.append("frameshift")
        elif any("*" in x for x in cl):
            status.append("nonsense")
        elif cov["nlpD"] < 10 and cov["in"] < 10:
            status.append("no_info")
        else:
            status.append("intact")
    t["rpoS_codon33"], t["rpoS_codon33_frac"] = c33, c33f
    t["rpoS_inside_over_nlpD"], t["rpoS_status"] = ratio, status

    variants = {g: [] for g in t.gsm}
    for _, row in tg.iterrows():
        for g in t.gsm:
            v = str(row.get(g, ""))
            for x in v.split("; "):
                if x and x != "." and x != "nan" and not x.startswith(("nd(", "[cl]")):
                    variants[g].append(f"{row.gene}:{x}")
    t["target_variants"] = t.gsm.map(lambda g: "; ".join(dict.fromkeys(variants[g])))
    t.to_csv(STATES, sep="\t", index=False, float_format="%.3f")
    print(f"sample_states.tsv: {len(t)} samples")
    for c in ["lineage", "mg1655_stock", "IS1H_state", "IS1I_state", "IS5U_state", "fim_state", "rpoS_status"]:
        print(c, t[c].value_counts().to_dict())


if __name__ == "__main__":
    main()
