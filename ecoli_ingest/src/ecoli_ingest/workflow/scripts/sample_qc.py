#!/usr/bin/env python
"""Per-sample QC record of an inGEST run (docs/atlas_qc_inputs.md).

Alignments are large and are not kept for the atlas, so everything that needs them is taken here:

  sample_qc.py bam       stranded coverage in 100 bp bins (<sample>.cov100.npz), fragment-size
                         histogram, mapped reads                              -> <sample>_bam_qc.json
  sample_qc.py versions  versions of the tools on PATH                        -> tool_versions.json
  sample_qc.py collect   everything about one sample in one file              -> <sample>.qc.json

The record keeps distributions (read lengths, fragment sizes, coverage, gene-body profile), not only
summary values, so that covariates can be redefined later without running the sample again.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from urllib.parse import unquote

BIN = 100
TLEN_MAX = 1000
FAMILIES = {"16S": r"^rrs", "23S": r"^rrl", "5S": r"^rrf", "ssrA": r"^ssrA$", "rnpB": r"^rnpB$", "ssrS": r"^ssrS$",
            "ffs": r"^ffs$"}
STABLE_TYPES = ("rRNA", "tRNA")
VERSION_COMMANDS = {
    "fastp": "fastp --version", "fastqc": "fastqc --version", "centrifuge": "centrifuge --version",
    "vg": "vg version", "samtools": "samtools --version", "bedtools": "bedtools --version",
    "featureCounts": "featureCounts -v", "rseqc": "infer_experiment.py --version", "multiqc": "multiqc --version",
    "snakemake": "snakemake --version", "R": "R --version",
}


def run(cmd: str) -> str:
    res = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise SystemExit(f"sample_qc: command failed ({res.returncode}): {cmd}\n{res.stderr[-400:]}")
    return res.stdout


# ------------------------------------------------------------------------------ alignments

def strand_expressions(call: str, layout: str) -> tuple[str | None, str | None]:
    """samtools filter expressions selecting the reads whose transcript lies on the + and on the
    - strand, given the library's strand call. Unstranded: (None, None)."""
    if layout == "PE":
        own, opp = ("((flag.read1 && !flag.reverse) || (flag.read2 && flag.reverse))",
                    "((flag.read1 && flag.reverse) || (flag.read2 && !flag.reverse))")
    else:
        own, opp = "!flag.reverse", "flag.reverse"
    if call == "forward":
        return own, opp
    if call == "reverse":
        return opp, own
    return None, None


def binned_depth(bam: str, expr: str | None) -> dict:
    """{contig: [summed per-base depth in each 100 bp bin]} for the reads selected by `expr`."""
    select = f"-e '{expr}'" if expr else ""
    text = run(f"samtools view -u -F 4 {select} '{bam}' | samtools depth -a - | "
               f"awk -F'\\t' '{{b=int(($2-1)/{BIN}); k=$1 \"\\t\" b; s[k]+=$3}} END{{for (k in s) print k \"\\t\" s[k]}}'")
    out: dict[str, dict[int, int]] = {}
    for line in text.splitlines():
        contig, b, depth = line.split("\t")
        out.setdefault(contig, {})[int(b)] = int(depth)
    return {c: [d.get(i, 0) for i in range(max(d) + 1)] for c, d in out.items()}


def feature_bins(bed: str) -> dict:
    """{contig: set of 100 bp bins that overlap a feature on either strand}."""
    out: dict[str, set] = {}
    with open(bed, encoding="utf-8") as fh:
        for line in fh:
            c = line.split("\t")
            if len(c) >= 3:
                out.setdefault(c[0], set()).update(range(int(c[1]) // BIN, (int(c[2]) - 1) // BIN + 1))
    return out


def quantile(values: list, q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return float(v[min(len(v) - 1, int(q * len(v)))])


def bam_qc(bam: str, call: str, layout: str, bed: str, npz_path: str) -> dict:
    """Coverage track (written to `npz_path`), its background levels, fragment sizes and mapped reads."""
    import numpy as np
    plus_expr, minus_expr = strand_expressions(call, layout)
    tracks = {"plus": binned_depth(bam, plus_expr), "minus": binned_depth(bam, minus_expr)} if plus_expr else \
        {"both": binned_depth(bam, None)}
    arrays = {}
    for strand, contigs in tracks.items():
        for contig, bins in contigs.items():
            arrays[f"{contig}__{strand}"] = np.asarray(bins, dtype=np.uint32)
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez_compressed(npz_path, bin_size=np.asarray(BIN), strand_call=np.asarray(call), **arrays)

    in_feature = feature_bins(bed)
    out = {"bin_size": BIN, "bam_mapped": int(run(f"samtools view -c -F 4 '{bam}'").strip()),
           "genome_length": sum(int(x.split("\t")[2].split(":")[1]) for x in
                                run(f"samtools view -H '{bam}' | grep '^@SQ'").splitlines())}
    for strand, contigs in tracks.items():
        depths = [d / BIN for bins in contigs.values() for d in bins]
        between = [d / BIN for contig, bins in contigs.items() for i, d in enumerate(bins)
                   if i not in in_feature.get(contig, ())]
        out[f"floor_cov_{strand}"] = quantile(depths, 0.10)            # mean depth per base, 10 % quantile of bins
        out[f"intergenic_cov_{strand}"] = quantile(between, 0.50)      # median over bins that touch no feature
        out[f"n_intergenic_bins_{strand}"] = len(between)
    if layout == "PE":
        text = run(f"samtools view -f 66 -F 4 '{bam}' | awk -F'\\t' '{{t=($9<0)?-$9:$9; if (t>{TLEN_MAX}) t={TLEN_MAX + 1}; "
                   "h[t]++} END{for (k in h) print k \"\\t\" h[k]}'")
        hist = sorted((int(a), int(b)) for a, b in (ln.split("\t") for ln in text.splitlines()))
        out["tlen_hist"] = [list(x) for x in hist]                   # [size, pairs]; last bin is > TLEN_MAX
    return out


# ------------------------------------------------------------------------------ small files

def read_fastp(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    keys = ("total_reads", "total_bases", "q20_rate", "q30_rate", "gc_content", "read1_mean_length", "read2_mean_length")
    pick = lambda block: {k: block[k] for k in keys if k in block}  # noqa: E731
    cut = d.get("adapter_cutting", {})
    ins = d.get("insert_size", {})
    return {
        "fastp_version": d.get("summary", {}).get("fastp_version"),
        "sequencing": d.get("summary", {}).get("sequencing"),
        "cycles": [d[k]["total_cycles"] for k in ("read1_before_filtering", "read2_before_filtering")
                   if k in d and "total_cycles" in d[k]],
        "before": pick(d["summary"]["before_filtering"]),
        "after": pick(d["summary"]["after_filtering"]),
        "filtering": d.get("filtering_result", {}),
        "duplication_rate": d.get("duplication", {}).get("rate"),
        "adapter": {"trimmed_reads": cut.get("adapter_trimmed_reads"), "trimmed_bases": cut.get("adapter_trimmed_bases"),
                    "read1_sequence": cut.get("read1_adapter_sequence"), "read2_sequence": cut.get("read2_adapter_sequence")},
        "fastp_insert": {"peak": ins.get("peak"), "unknown": ins.get("unknown"), "histogram": ins.get("histogram")},
    }


def read_fastqc(zip_path: str, top: int = 10) -> dict:
    """Read-length histogram, most frequent sequences and deduplicated share from a FastQC zip."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("fastqc_data.txt"))
        text = z.read(name).decode("utf-8", errors="replace")
    out = {"length_hist": [], "overrepresented": [], "deduplicated_pct": None}
    module = None
    for line in text.splitlines():
        if line.startswith(">>END_MODULE"):
            module = None
        elif line.startswith(">>"):
            module = line[2:].split("\t")[0]
        elif line.startswith("#Total Deduplicated Percentage"):
            out["deduplicated_pct"] = float(line.split("\t")[1])
        elif line.startswith("#") or not module:
            continue
        elif module == "Sequence Length Distribution":
            length, count = line.split("\t")[:2]                   # FastQC writes ranges ("30-34") for long reads
            out["length_hist"].append([int(length) if length.isdigit() else length, int(float(count))])
        elif module == "Overrepresented sequences" and len(out["overrepresented"]) < top:
            seq, count, pct, source = (line.split("\t") + ["", "", "", ""])[:4]
            out["overrepresented"].append([seq, int(float(count)), round(float(pct), 3), source])
    return out


def length_quantiles(hist: list, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    """Quantiles of a [[length or 'a-b', count], ...] histogram (the lower bound of a range is used)."""
    pairs = sorted((int(str(length).split("-")[0]), n) for length, n in hist)
    total = sum(n for _, n in pairs)
    out, acc, i = {}, 0, 0
    for q in qs:
        while i < len(pairs) and acc + pairs[i][1] < q * total:
            acc += pairs[i][1]
            i += 1
        out[f"q{int(q * 100):02d}"] = pairs[min(i, len(pairs) - 1)][0] if pairs else None
    return out


_VG_LINES = {
    "Total alignments": "total", "Total primary": "primary", "Total secondary": "secondary", "Total aligned": "aligned",
    "Total perfect": "perfect", "Total gapless (softclips allowed)": "gapless", "Total paired": "paired",
    "Total properly paired": "properly_paired",
}


def read_vg_stats(path: str) -> dict:
    """Every value of `vg stats -a`. The file ends with two placeholder lines (time, speed) written by
    the workflow for MultiQC; the first occurrence of a key is the real one."""
    out: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            key, _, rest = line.strip().partition(":")
            rest = rest.strip()
            if key in _VG_LINES:
                out.setdefault(_VG_LINES[key], int(rest))
            elif key in ("Alignment score", "Mapping quality"):
                name = "score" if key.startswith("Alignment") else "mapq"
                for stat, value in re.findall(r"(mean|median|stdev|max) ([0-9.eE+-]+)", rest):
                    out.setdefault(f"{name}_{stat}", float(value))
            elif key in ("Insertions", "Deletions", "Substitutions", "Softclips"):
                m = re.match(r"(\d+) bp in (\d+) read events", rest)
                if m:
                    out.setdefault(f"{key.lower().rstrip('s')}_bp", int(m.group(1)))
                    out.setdefault(f"{key.lower().rstrip('s')}_events", int(m.group(2)))
            elif key in ("Total time", "Speed"):
                out.setdefault("time_seconds" if key == "Total time" else "reads_per_second", float(rest.split()[0]))
    return out


def read_gene_body(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    return [float(x) for x in lines[1][1:]] if len(lines) > 1 else []


def gene_body_slope(profile: list) -> float | None:
    """Least-squares slope of the profile (scaled to mean 1) from the 5' to the 3' end, per unit of gene length."""
    n = len(profile)
    mean = sum(profile) / n if n else 0
    if n < 2 or mean <= 0:
        return None
    xs = [(i + 0.5) / n for i in range(n)]
    ys = [v / mean for v in profile]
    mx = sum(xs) / n
    return round(sum((x - mx) * (y - 1) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs), 4)


def read_strand(path: str, sample: str) -> dict:
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["sample"] == sample:
                return {"layout": row["layout"], "fraction_forward": float(row["fraction_forward"]),
                        "fraction_reverse": float(row["fraction_reverse"]),
                        "fraction_undetermined": float(row["fraction_undetermined"]), "call": row["call"],
                        "call_source": row.get("call_source", "")}
    return {}


# ------------------------------------------------------------------------------ counts

def sample_column(header: list, sample: str) -> int:
    """Index of the sample's column in a featureCounts table (columns are BAM paths)."""
    hits = [i for i, h in enumerate(header) if re.search(rf"(^|/){re.escape(sample)}_sort\.bam$", h) or h == sample]
    if len(hits) != 1:
        raise SystemExit(f"sample_qc: cannot find the column of {sample} among {header[-3:]}")
    return hits[0]


def read_count_column(path: str, sample: str) -> dict:
    """feature id -> (count, length) for one sample of a featureCounts table."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        rows = (ln.rstrip("\n").split("\t") for ln in fh if not ln.startswith("#"))
        header = next(rows)
        col, length = sample_column(header, sample), header.index("Length")
        for r in rows:
            out[r[0]] = (int(float(r[col])), int(r[length]))
    return out


def read_summary_column(path: str, sample: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    col = sample_column(rows[0], sample)
    return {r[0]: int(r[col]) for r in rows[1:]}


def read_feature_types(gff: str) -> dict:
    """feature id -> (type, symbol) from the counting annotation."""
    out = {}
    with open(gff, encoding="utf-8") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(c) != 9:
                continue
            a = dict(x.split("=", 1) for x in c[8].split(";") if "=" in x)
            group = "Rfam" if a.get("source") == "Rfam" else a.get("feature_type", "")
            out[unquote(a["ID"])] = (group, unquote(a.get("symbol", "")))
    return out


def composition(sense: dict, features: dict) -> dict:
    by_type: dict[str, int] = {}
    by_family = {k: 0 for k in list(FAMILIES) + ["tRNA"]}
    for fid, (count, _) in sense.items():
        group, symbol = features.get(fid, ("", ""))
        by_type[group] = by_type.get(group, 0) + count
        if group == "tRNA":
            by_family["tRNA"] += count
        for family, pattern in FAMILIES.items():
            if re.search(pattern, symbol) and group in ("rRNA", "ncRNA"):
                by_family[family] += count
    return {"by_type": by_type, "by_family": by_family}


def robust_line(x: list, y: list, max_pairs: int = 200_000, seed: int = 1) -> tuple[float | None, float | None]:
    """Theil-Sen line y = a + b x: b is the median slope over (a random subset of) point pairs, a the
    median of y - b x. Returns (a, b)."""
    import random
    n = len(x)
    if n < 10:
        return None, None
    rng = random.Random(seed)
    slopes = []
    for _ in range(min(max_pairs, n * (n - 1) // 2)):
        i, j = rng.randrange(n), rng.randrange(n)
        if x[i] != x[j]:
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return None, None
    slopes.sort()
    b = slopes[len(slopes) // 2]
    resid = sorted(yi - b * xi for xi, yi in zip(x, y))
    return resid[len(resid) // 2], b


def dna_estimates(sense: dict, antisense: dict, features: dict, assignment: dict, antisense_assignment: dict,
                  genome_length: int | None) -> dict:
    """Values for judging genomic DNA in the library (docs/atlas_qc_inputs.md, item 15). Counts are in
    the unit featureCounts used (reads, or fragments for paired-end runs); rRNA and tRNA genes are left
    out of the fit."""
    assigned, nofeat = assignment.get("Assigned", 0), assignment.get("Unassigned_NoFeatures", 0)
    anti_assigned = antisense_assignment.get("Assigned", 0)
    xs, ys = [], []
    for fid, (count, length) in sense.items():
        if features.get(fid, ("",))[0] in STABLE_TYPES or fid not in antisense or length <= 0:
            continue
        xs.append(count / length)
        ys.append(antisense[fid][0] / length)
    a, b = robust_line(xs, ys)
    aligned = assigned + nofeat
    bases = sum(length for fid, (_, length) in sense.items() if features.get(fid, ("",))[0] not in STABLE_TYPES)
    anti_total = sum(antisense[fid][0] for fid in sense if fid in antisense
                     and features.get(fid, ("",))[0] not in STABLE_TYPES)
    return {
        "intergenic_share": round(nofeat / aligned, 6) if aligned else None,
        "antisense_share": round(anti_assigned / (assigned + anti_assigned), 6) if assigned + anti_assigned else None,
        "floor_a": a, "leakage_b": b, "n_genes_fit": len(xs),
        # antisense reads per base over all genes outside rRNA / tRNA: an upper bound on the even
        # background that stays informative at low depth, where the median-based fit gives 0
        "antisense_per_base": round(anti_total / bases, 8) if bases else None,
        "dna_fraction_est": (max(0.0, 2 * a * genome_length / aligned) if a is not None and genome_length and aligned
                             else None),
        "aligned_units": aligned,
    }


# ------------------------------------------------------------------------------ assembly

def read_run_metadata(path: str | None, runs: list) -> list:
    """Rows of a run table (e.g. the atlas manifest's runs.tsv) for the sample's runs."""
    wanted = {r["run"]: dict(r) for r in runs}
    keep = ("library_strategy", "library_source", "library_selection", "library_layout", "instrument_platform",
            "instrument_model", "read_count", "base_count", "first_public", "GSM", "GSE")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row.get("run_accession") in wanted:
                    wanted[row["run_accession"]].update({k: row[k] for k in keep if row.get(k)})
    return list(wanted.values())


def collect(args) -> dict:
    s = args.sample
    features = read_feature_types(args.gff)
    sense = read_count_column(args.counts, s)
    antisense = read_count_column(args.antisense, s)
    assignment = read_summary_column(args.counts + ".summary", s)
    anti_assignment = read_summary_column(args.antisense + ".summary", s)
    with open(args.bam_qc, encoding="utf-8") as fh:
        bam = json.load(fh)
    vg = read_vg_stats(args.vg_stats)
    reads = read_fastp(args.fastp)
    fastp_insert = reads.pop("fastp_insert")
    reads["length_hist"], reads["length_quantiles"], reads["overrepresented"], reads["deduplicated_pct"] = {}, {}, {}, {}
    for mate, path in (("r1", args.fastqc_r1), ("r2", args.fastqc_r2)):
        if path:
            fq = read_fastqc(path)
            reads["length_hist"][mate] = fq["length_hist"]
            reads["length_quantiles"][mate] = length_quantiles(fq["length_hist"])
            reads["overrepresented"][mate] = fq["overrepresented"]
            reads["deduplicated_pct"][mate] = fq["deduplicated_pct"]
    comp = composition(sense, features)
    comp["antisense_assigned"] = anti_assignment.get("Assigned", 0)
    stable = sum(comp["by_type"].get(t, 0) for t in STABLE_TYPES)
    aligned, mapped = vg.get("aligned"), bam.get("bam_mapped")
    profile = read_gene_body(args.gene_body) if args.gene_body else []
    dna = dna_estimates(sense, antisense, features, assignment, anti_assignment, bam.get("genome_length"))
    dna.update({k: v for k, v in bam.items() if k.startswith(("floor_cov_", "intergenic_cov_"))})
    versions = {}
    if args.versions and os.path.exists(args.versions):
        with open(args.versions, encoding="utf-8") as fh:
            versions = json.load(fh)
    species = {}
    if args.species and os.path.exists(args.species):
        with open(args.species, encoding="utf-8") as fh:
            species = json.load(fh)
    return {
        "sample": s,
        "pipeline": versions,
        "runs": read_run_metadata(args.run_metadata, json.loads(args.runs) if args.runs else []),
        "reads": reads,
        "fragment": {"tlen_hist": bam.get("tlen_hist", []), "fastp_insert_hist": fastp_insert["histogram"],
                     "fastp_insert_peak": fastp_insert["peak"], "fastp_insert_unknown": fastp_insert["unknown"]},
        "alignment": {"vg_stats": vg, "bam_mapped": mapped,
                      # reads that align to the graph but are not placed on the reference path
                      "off_reference_fraction": round((aligned - mapped) / aligned, 6) if aligned and mapped is not None
                      else None,
                      "substitution_rate": round(vg["substitution_bp"] / reads["after"]["total_bases"], 6)
                      if vg.get("substitution_bp") is not None and reads["after"].get("total_bases") else None},
        "strand": read_strand(args.strand, s),
        "assignment": assignment,
        "antisense_assignment": anti_assignment,
        # read ladder, in the unit each tool reports (fastp and vg: reads; featureCounts: reads or fragments)
        "ladder": {"reads_in": reads["before"].get("total_reads"), "reads_kept": reads["after"].get("total_reads"),
                   "reads_aligned": aligned, "reads_on_reference": mapped, "assigned": assignment.get("Assigned"),
                   "assigned_not_stable_rna": assignment.get("Assigned", 0) - stable if assignment.get("Assigned") else None,
                   "counting_unit": "fragments" if reads["after"].get("read2_mean_length") else "reads"},
        "composition": comp,
        "dna": dna,
        "gene_body": profile,
        "gene_body_slope": gene_body_slope(profile),
        "coverage": {"file": os.path.basename(args.coverage) if args.coverage else None, "bin_size": bam.get("bin_size")},
        "species": species,
    }


def tool_versions() -> dict:
    out = {}
    for tool, cmd in VERSION_COMMANDS.items():
        res = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        m = re.search(r"\d+\.\d+(\.\d+)*[\w.-]*", res.stdout)
        out[tool] = m.group(0) if m and res.returncode in (0, 1) else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bam")
    b.add_argument("--bam", required=True)
    b.add_argument("--strand", required=True, help="the sample's strandedness table")
    b.add_argument("--sample", required=True)
    b.add_argument("--bed", required=True, help="counting annotation BED")
    b.add_argument("--coverage", required=True, help="output .npz")
    b.add_argument("--out", required=True)
    v = sub.add_parser("versions")
    v.add_argument("--commit", default="")
    v.add_argument("--package-version", default="")
    v.add_argument("--annotation", default="")
    v.add_argument("--out", required=True)
    c = sub.add_parser("collect")
    for name in ("sample", "fastp", "vg-stats", "strand", "counts", "antisense", "gff", "bam-qc", "out"):
        c.add_argument("--" + name, required=True)
    for name in ("fastqc-r1", "fastqc-r2", "gene-body", "species", "versions", "coverage", "runs", "run-metadata"):
        c.add_argument("--" + name, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "bam":
        strand = read_strand(args.strand, args.sample)
        out = bam_qc(args.bam, strand.get("call", "unstranded"), strand.get("layout", "SE"), args.bed, args.coverage)
    elif args.cmd == "versions":
        out = {"commit": args.commit, "package_version": args.package_version, "annotation": args.annotation,
               "versions": tool_versions()}
    else:
        out = collect(args)
    with open(args.out + ".part", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(args.out + ".part", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
