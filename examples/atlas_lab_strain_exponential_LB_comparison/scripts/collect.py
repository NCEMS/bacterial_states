"""Collect the breseq runs of the LB-exponential lab-strain samples into tables and test which variants go with a
sample's distance from the trend lines of fig_batch_tracking (regulon mean against the ribosomal proteins).

Writes into --out:
  mutations.tsv           every predicted mutation (gdtools ANNOTATE), one row per sample x mutation, with the
                          sample's study, strain and a key that is the same for the same change in every sample
  genotypes.tsv           mutation key x sample: frequency where called, 0 where the site is covered (>= MIN_DEPTH
                          reads) and not called, NA where it is not covered (no RNA there: no information)
  target_genes.tsv        the target genes (flagella, chemotaxis, flhDC regulators, cAMP-Crp, stress / growth
                          regulators, known lab-stock loci) x sample: mean read depth and the changes called in the
                          gene or its 150 bp upstream
  target_junctions.tsv    breseq junctions (JC) with a side in a target gene region, not resolved into a mutation
  loci_states.tsv         IS1H (flhDC upstream), IS1I (in crl), IS5U (ychE-oppA) and the fim switch (scripts/loci_states.py on the
                          scripts/loci_bam.sh BAMs) with the sample table and residuals; loci_junctions.tsv: the
                          clipped reads behind them
  sample_qc.tsv           calls per sample and calls in clusters (another call within 1 kb: reads of a second
                          E. coli in the library where the K-12 genes are silent); `mixed` samples (>= 20 clustered
                          calls) are left out of the associations; `junction_k50`: breseq run with the junction-candidate
                          alignment capped at 50 hits per read instead of 2000 (scripts/breseq.sh, reads.tsv)
  residuals.tsv           each sample's residual from the study-level line of every set against the ribosome
                          (scripts/residuals.py)
  associations.tsv        per variable mutation (K-12 samples): carriers vs covered non-carriers, residual
                          difference per set, at the sample and at the study level
Then scripts/stratify.py (lineage.tsv, lineage_markers.tsv, associations_by_lineage.tsv) and scripts/states.py
(sample_states.tsv in the example folder: one row per sample with every state; tracked, the input of the figures).
Studies in EXCLUDE_STUDIES stay in the per-sample tables (`excluded`) but are left out of the trend-line fit, the
lineage markers and the associations.
"""
import argparse, csv, glob, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from scipy import stats

from common import SCR, RESULTS, GBK, FASTA, GENOME_INDEX, FEATURES, SAMPLES, READS, SETS, EXCLUDE_STUDIES
from residuals import residuals

HERE = os.path.dirname(os.path.abspath(__file__))
REF = [str(GBK), str(FASTA)]        # the GenBank file has a CONTIG line: gdtools takes the sequence from the FASTA
MIN_DEPTH, MIN_FREQ = 5, 0.25
CLUSTER_BP, MIXED_CLUSTERED = 1000, 20   # clustered calls (foreign-strain reads); a sample with >= 20 is `mixed`
UPSTREAM = 150
THREADS = 16
TARGET_GROUPS = {
    "flagella_chemotaxis": r"^(flh[A-Z]|fli[A-Z]|flg[A-Z]|mot[AB]|che[ABRWYZ]|tar|tap|tsr|trg|aer|ycgR|pdeH|rflP|flk)$",
    "flhDC_regulators": r"^(hdfR|lrhA|ompR|envZ|rcs[ABCDF]|qse[BC]|ecpR|csrA|csrB|csrC|ihf[AB]|hns|stpA|fis|clp[XP]|lon|"
                        r"uvrY|barA|bglJ|yjjQ|iscR|rflM|hfq)$",
    "cAMP_Crp": r"^(crp|cyaA|cpdA|crr|ptsG|ptsH|ptsI|mlc|sgrR|sgrS|glk|ptsN)$",
    "stress_growth": r"^(rpoS|nlpD|rssB|ira[PMD]|relA|spoT|dksA|rpo[ABCDZ]|crl|arcA|arcB|fnr|fur|nar[LXPQ]|lrp|rpoE|rseA|"
                     r"nusG|rho|cydA|cyoA|ndh|sdhC|aceE|gltA|ppsA|pykF|pgi|zwf)$",
    "lab_stock_loci": r"^(rph|pyrE|ilvG|glpR|gatC|fhuA|lacI|lacZ|araC|rhaS|mreB|nmpC|trkH|cyaR|ompF|ompC|fimB|fimE)$",
}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def sample_table(samples):
    s = pd.read_csv(samples, sep="\t")
    jobs = pd.read_csv(READS, sep="\t", quoting=csv.QUOTE_NONE)
    ctrl = jobs[~jobs.gsm.isin(s.gsm)][["gsm", "gse"]].assign(strain="control", strain_name="control")
    s = pd.concat([s, ctrl], ignore_index=True)
    s["excluded"] = s.gse.map(EXCLUDE_STUDIES).fillna("")
    return s


def annotate(gds, out):
    tsv = f"{out}/annotated_raw.tsv"
    args = ["gdtools", "ANNOTATE", "-f", "TSV", "-o", tsv]
    for r in REF:
        args += ["-r", r]
    run(args + gds)
    m = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    m = m.rename(columns={"title": "gsm"})
    m["frequency"] = pd.to_numeric(m["frequency"], errors="coerce").fillna(1.0) if "frequency" in m else 1.0   # consensus mode: none
    m["position"] = m["position"].astype(int)
    # The same change has the same key in every sample
    def key(r):
        if r["type"] == "MOB":
            return f"MOB:{r['position']}:{r.get('repeat_name', '')}:{r.get('strand', '')}:{r.get('duplication_size', '')}"
        if r["type"] == "DEL":
            return f"DEL:{r['position']}:{r.get('size', '')}"
        return f"{r['type']}:{r['position']}:{r.get('ref_seq', '')}>{r.get('new_seq', '')}"
    m["key"] = m.apply(key, axis=1)
    os.remove(tsv)
    return m


def positions_depth(bams, positions, out):
    """Depth at each position (1-based) in every BAM: DataFrame positions x gsm."""
    bed = f"{out}/_positions.bed"
    with open(bed, "w") as f:
        for p in sorted(set(positions)):
            f.write(f"NC_000913\t{p - 1}\t{p}\n")
    def one(bam):
        txt = run(["samtools", "depth", "-a", "-b", bed, "-Q", "0", bam])
        return {int(l.split("\t")[1]): int(l.split("\t")[2]) for l in txt.splitlines()}
    with ThreadPoolExecutor(THREADS) as ex:
        cols = dict(zip(bams, ex.map(one, bams.values())))
    os.remove(bed)
    return pd.DataFrame(cols).reindex(sorted(set(positions))).fillna(0)


def gene_depth(bams, genes, out):
    bed = f"{out}/_genes.bed"
    genes[["chrom", "start0", "End", "Symbol"]].to_csv(bed, sep="\t", header=False, index=False)
    one = lambda bam: [int(l.split("\t")[4]) for l in run(["samtools", "bedcov", bed, bam]).splitlines()]
    with ThreadPoolExecutor(THREADS) as ex:
        cols = dict(zip(bams, ex.map(one, bams.values())))
    os.remove(bed)
    d = pd.DataFrame(cols, index=genes["Symbol"].values)
    return d.div((genes["End"] - genes["start0"]).values, axis=0)


def target_genes():
    f = pd.read_csv(FEATURES, sep="\t")
    f = f[f["Type"].isin(["CDS", "ncRNA", "pseudo"]) & f["Symbol"].notna()].copy()
    f["group"] = ""
    for g, rx in TARGET_GROUPS.items():
        hit = f["Symbol"].str.match(rx) & (f["group"] == "")
        f.loc[hit, "group"] = g
    f = f[f["group"] != ""].drop_duplicates("Symbol").copy()
    f["chrom"], f["start0"] = "NC_000913", f["Start"] - 1
    up = np.where(f["Strand"] == "+", f["Start"] - UPSTREAM, f["Start"])
    dn = np.where(f["Strand"] == "+", f["End"], f["End"] + UPSTREAM)
    f["reg_start"], f["reg_end"] = up, dn
    return f.sort_values("Start")


def short(r):
    t = r["type"]
    if t == "SNP":
        d = f"{r['aa_ref_seq']}{r['aa_position']}{r['aa_new_seq']}" if r.get("aa_position") else f"{r['ref_seq']}>{r['new_seq']}"
    elif t == "MOB":
        d = f"{r.get('repeat_name', 'IS')}"
    elif t == "DEL":
        d = f"Δ{r.get('size', '')}bp"
    elif t == "INS":
        d = f"+{r.get('new_seq', '')}"
    else:
        d = t
    d += f"@{r['position']}"
    if r["frequency"] < 0.8:
        d += f"({r['frequency']:.2f})"
    return f"[cl]{d}" if r.get("cluster") else d


def junctions(gds, regions):
    rows = []
    for gd in gds:
        gsm = gd.split("/")[-3]
        for line in open(gd):
            if not line.startswith("JC\t"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != ".":          # used by a mutation
                continue
            kv = dict(x.split("=", 1) for x in f[9:] if "=" in x)
            n = int(kv.get("new_junction_read_count", 0) or 0)
            seq = kv.get("unique_read_sequence", "")
            if n < 3 or "AGATCGG" in seq or "CCGATCT" in seq:   # adapter read-through
                continue
            p1, p2 = int(f[4]), int(f[7])
            for _, g in regions.iterrows():
                if g.reg_start - 200 <= p1 <= g.reg_end + 200 or g.reg_start - 200 <= p2 <= g.reg_end + 200:
                    rows.append({"gsm": gsm, "gene": g.Symbol, "group": g.group, "side1": p1, "strand1": f[5],
                                 "side2": p2, "strand2": f[8], "reads": n, "frequency": kv.get("frequency", ""),
                                 "side1_annot": kv.get("side_1_annotate_key", ""), "unique_seq": seq})
    return pd.DataFrame(rows)


def associations(geno, res, strain_of, gse_of, mixed):
    k12 = [g for g in geno.columns if strain_of.get(g) in ("MG1655", "BW25113", "other lab strain") and g not in mixed
           and "B DSM" not in str(res.set_index("gsm").loc[g, "strain_name"])]
    r = res.set_index("gsm").loc[k12]
    rows = []
    for key, v in geno[k12].iterrows():
        carrier = v >= 0.5
        noncarrier = v == 0
        if carrier.sum() < 2 or noncarrier.sum() < 5:
            continue
        row = {"key": key, "n_carriers": int(carrier.sum()), "n_noncarriers": int(noncarrier.sum()),
               "carrier_studies": ",".join(sorted(set(gse_of[g] for g in carrier[carrier].index)))}
        row["n_carrier_studies"] = len(row["carrier_studies"].split(","))
        for k in ["ribosome"] + SETS:
            col = k if k == "ribosome" else f"res_{k}"
            a, b = r.loc[carrier[carrier].index, col], r.loc[noncarrier[noncarrier].index, col]
            row[f"diff_{k}"] = a.mean() - b.mean()
            row[f"p_{k}"] = stats.mannwhitneyu(a, b).pvalue if len(a) > 1 else np.nan
            # study level: one value per study (mean), carrier study = most covered samples carry it
            st = pd.DataFrame({"gse": [gse_of[g] for g in a.index.tolist() + b.index.tolist()],
                               "c": [1] * len(a) + [0] * len(b), "y": list(a) + list(b)}).groupby("gse").mean()
            sa, sb = st[st.c >= 0.5].y, st[st.c < 0.5].y
            row[f"pstudy_{k}"] = stats.mannwhitneyu(sa, sb).pvalue if len(sa) > 1 and len(sb) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=str(SAMPLES))
    p.add_argument("--breseq", default=str(SCR / "breseq"), help="<GSM>/output/output.gd and <GSM>/data/reference.bam")
    p.add_argument("--loci-bams", default=str(SCR / "loci"), help="scripts/loci_bam.sh output, <GSM>.bam")
    p.add_argument("--out", default=str(RESULTS))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = sample_table(a.samples)
    gds = sorted(g for g in glob.glob(f"{a.breseq}/*/output/output.gd") if g.split("/")[-3] in set(s.gsm))
    done = [g.split("/")[-3] for g in gds]
    bams = {g: f"{a.breseq}/{g}/data/reference.bam" for g in done if os.path.exists(f"{a.breseq}/{g}/data/reference.bam")}
    print(f"{len(gds)} breseq runs, {len(bams)} BAMs, {len(s)} samples")
    strain_of = dict(zip(s.gsm, s.strain)); gse_of = dict(zip(s.gsm, s.gse))

    res = residuals(s[s.strain != "control"])      # scripts/residuals.py
    res.to_csv(f"{a.out}/residuals.tsv", sep="\t", index=False, float_format="%.3f")

    m = annotate(gds, a.out)
    m.insert(1, "gse", m.gsm.map(gse_of)); m.insert(2, "strain", m.gsm.map(strain_of))
    m.insert(3, "strain_name", m.gsm.map(dict(zip(s.gsm, s.strain_name))))
    keep = ["gsm", "gse", "strain", "strain_name", "key", "type", "position", "frequency", "gene_name", "gene_position",
            "gene_product", "mutation_category", "snp_type", "aa_ref_seq", "aa_position", "aa_new_seq", "ref_seq",
            "new_seq", "size", "repeat_name", "genes_inactivated", "genes_promoter", "new_read_count", "ref_read_count", "cluster"]
    m = m[[c for c in keep if c in m.columns]]
    # Foreign-strain reads: in regions where the K-12 genes are silent, a few reads of another E. coli in the library
    # (contamination, index hopping) give clusters of consensus SNPs. A call with another call within CLUSTER_BP in
    # the same sample is flagged and left out of the genotypes; a sample with >= MIXED_CLUSTERED of them is `mixed`.
    m = m.sort_values(["gsm", "position"])
    gap = lambda d: np.minimum(d.position.diff().abs().fillna(1e9), (-d.position.diff(-1)).abs().fillna(1e9))
    m["cluster"] = m.groupby("gsm", group_keys=False).apply(lambda d: gap(d) <= CLUSTER_BP, include_groups=False) \
        & ~m.type.isin(["MOB", "AMP", "CON", "INV"])
    q = m.groupby("gsm").agg(n_calls=("key", "size"), n_clustered=("cluster", "sum"))
    q = s[s.gsm.isin(done)][["gsm", "gse", "strain", "strain_name", "excluded"]].merge(q, left_on="gsm", right_index=True, how="left")
    q[["n_calls", "n_clustered"]] = q[["n_calls", "n_clustered"]].fillna(0).astype(int)   # a sample without calls
    q["mixed"] = q.n_clustered >= MIXED_CLUSTERED
    # runs resumed with the junction-candidate alignment capped at 50 hits per read (scripts/breseq_resume.sh)
    q["junction_k50"] = [os.path.exists(f"{a.breseq}/{g}/JUNCTION_K50") for g in q.gsm]
    q.to_csv(f"{a.out}/sample_qc.tsv", sep="\t", index=False)
    mixed = set(q.gsm[q.mixed])
    left_out = mixed | set(q.gsm[q.excluded != ""])     # not in the associations
    m = m[m.frequency >= MIN_FREQ]
    m.to_csv(f"{a.out}/mutations.tsv", sep="\t", index=False)

    # genotype matrix with coverage: called / covered-not-called / not covered
    mg = m[~m.cluster]
    first = mg.groupby("key")["position"].first()
    depth = positions_depth(bams, first.values, a.out)
    freq = mg.pivot_table(index="key", columns="gsm", values="frequency", aggfunc="max")
    geno = pd.DataFrame(np.nan, index=first.index, columns=list(bams))
    for g in bams:
        d = depth[g].reindex(first.values).values
        geno[g] = np.where(d >= MIN_DEPTH, 0.0, np.nan)
        if g in freq.columns:
            f = freq[g].reindex(first.index)
            geno.loc[f.notna(), g] = f[f.notna()]
    info = mg.drop_duplicates("key").set_index("key")[["type", "position", "gene_name", "gene_position", "mutation_category",
                                                       "aa_ref_seq", "aa_position", "aa_new_seq"]]
    geno = geno.reindex(columns=[g for g in s.gsm if g in geno.columns])
    called = (geno >= 0.5).sum(axis=1); covered = geno.notna().sum(axis=1)
    info["n_called"], info["n_covered"] = called, covered
    info.join(geno).sort_values("position").to_csv(f"{a.out}/genotypes.tsv", sep="\t", float_format="%.2f")

    # target genes
    tg = target_genes()
    gd = gene_depth(bams, tg, a.out)
    cells = {}
    for _, g in tg.iterrows():
        hit = m[(m.position >= g.reg_start) & (m.position <= g.reg_end)]
        cells[g.Symbol] = hit.groupby("gsm").apply(lambda h: "; ".join(short(r) for _, r in h.iterrows()), include_groups=False)
    rows = []
    for _, g in tg.iterrows():
        r = {"group": g.group, "gene": g.Symbol, "start": g.Start, "end": g.End, "strand": g.Strand}
        for gsm in geno.columns:
            dep = gd.loc[g.Symbol, gsm] if gsm in gd.columns else np.nan
            mut = cells[g.Symbol].get(gsm, "")
            r[gsm] = mut if mut else ("." if dep >= MIN_DEPTH else f"nd({dep:.0f}x)")
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{a.out}/target_genes.tsv", sep="\t", index=False)
    gd.to_csv(f"{a.out}/target_gene_depth.tsv", sep="\t", float_format="%.1f")
    junctions(gds, tg).to_csv(f"{a.out}/target_junctions.tsv", sep="\t", index=False)

    # IS1H, IS1I and fim switch states
    lbams = sorted(b for b in glob.glob(f"{a.loci_bams}/*.bam") if os.path.basename(b)[:-4] in set(s.gsm))
    if lbams:
        run([sys.executable, f"{HERE}/loci_states.py", "--index", str(GENOME_INDEX), "--gbk", REF[0],
             "--out", f"{a.out}/_loci.tsv", "--junctions", f"{a.out}/loci_junctions.tsv"] + lbams)
        ld = pd.read_csv(f"{a.out}/_loci.tsv", sep="\t")
        os.remove(f"{a.out}/_loci.tsv")
        s[["gsm", "gse", "strain", "strain_name", "excluded"]].merge(ld, on="gsm", how="right") \
            .merge(res[["gsm"] + [f"res_{k}" for k in SETS]], on="gsm", how="left") \
            .to_csv(f"{a.out}/loci_states.tsv", sep="\t", index=False, float_format="%.3f")

    asc = associations(geno, res, strain_of, gse_of, left_out)
    (asc.sort_values("p_FlhDC") if len(asc) else asc).to_csv(
        f"{a.out}/associations.tsv", sep="\t", index=False, float_format="%.4g")

    # lineage from the genotypes and associations within lineage; then one table of every per-sample state
    print(run([sys.executable, f"{HERE}/stratify.py", "--results", a.out]))
    print(run([sys.executable, f"{HERE}/states.py", "--results", a.out, "--breseq", a.breseq]))

    print("wrote", a.out)


if __name__ == "__main__":
    main()
