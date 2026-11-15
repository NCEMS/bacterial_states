#!/usr/bin/env python
"""Build the manifest of GEO E. coli RNA-seq samples that can go into the atlas, from ENA alone.

Replaces sample_info.R (GEOmetadb + one GEOquery call per sample). ENA records of GEO submissions
carry the GEO ids themselves (study_alias = GSE, experiment_alias = GSM), so one ENA search gives
runs, library fields and the GEO mapping. Every run gets a status and the reason it was excluded.

Steps
  1. ENA read_run search: tax_tree(562) (E. coli and every strain below it), library_strategy RNA-Seq.
  2. For the GEO studies found, list all their runs (any organism, any strategy) to mark studies
     that also hold non-E. coli samples. Their E. coli samples are kept.
  3. Apply the rules below, group runs by GSM.

A run is a candidate unless (first rule that applies is its status; all that apply are in `flags`):
  no_geo_ids            no GSE study alias or no GSM alias (submitted directly to SRA/ENA/DDBJ)
  not_illumina          instrument platform other than ILLUMINA
  not_transcriptomic    library_source other than TRANSCRIPTOMIC
  no_fastq              ENA has no FASTQ files for the run
  library_type          the experiment or sample title names a library type that is not
                        whole-transcriptome RNA-seq (see LIBRARY_TYPE_PATTERNS; the matched class is
                        in `library_type_match`)
  low_reads             fewer than --min-reads reads

Two things do not exclude a run but are noted in `review`:
  mixed_organism_study  its GEO study also has runs from another organism (co-culture or infection
                        samples would be registered under the other organism or a metagenome taxon)
  study_title:<class>   only the study title names a non-standard library type; such studies
                        usually also hold ordinary RNA-seq samples

Strand specificity cannot be read from metadata; it is decided later from the alignments.

Outputs (in --out-dir)
  ena_runs.tsv, ena_study_runs.tsv   the ENA downloads, unmodified (reused unless --refresh)
  runs.tsv                           one row per run
  samples.tsv                        one row per GSM (runs merged)
  studies.tsv                        one row per GSE
  summary.txt
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

ENA_SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
RUN_FIELDS = ["run_accession", "experiment_accession", "sample_accession", "study_accession", "study_alias",
              "experiment_alias", "sample_alias", "run_alias", "tax_id", "scientific_name", "library_strategy",
              "library_source", "library_selection", "library_layout", "instrument_platform", "instrument_model",
              "read_count", "base_count", "first_public", "study_title", "experiment_title", "sample_title",
              "fastq_ftp", "fastq_bytes"]
STUDY_RUN_FIELDS = ["run_accession", "study_accession", "tax_id", "scientific_name", "library_strategy"]
ECOLI_QUERY = 'tax_tree(562) AND library_strategy="RNA-Seq"'

# Library types that are labelled RNA-Seq but do not measure whole-transcript abundance.
# class -> regular expression searched in the titles, case-insensitive except for the acronyms
# inside (?-i:...), which would otherwise match gene names such as dms or rip.
LIBRARY_TYPE_PATTERNS = {
    "ribosome_profiling": r"ribo-?seq|ribosom\w* (profil|footprint)|ribosome[- ]protected|\bRPFs?\b|\bdisome",
    "five_prime_ends": r"\bdRNA-?seq|differential RNA-?seq|cappable|(?-i:\bTSS\b)|transcription start site|\bTEX[+-]|5['′][ -]?ends?\b|\b5['′]?-?RACE",
    "three_prime_ends": r"term-?seq|3['′][ -]?ends?\b|\bPRO-?seq|\bNET-?seq|nascent",
    "rna_interactome": r"RIL-?seq|(?-i:\bCLASH\b|\bCLIP\b|\bRIP\b)|CLIP-?seq|RIP-?seq|grad-?seq|co-?IP\b|pull-?down|immunoprecipitat",
    "rna_structure": r"(?-i:\bSHAPE\b|\bDMS\b|\bPARS\b)|structure-?seq",
    "small_rna_fraction": r"tRNA-?seq|small RNA-?seq|sRNA-?seq|\bmiRNA",
    "single_cell": r"single[- ]cell|\bscRNA|PETRI-?seq|microSPLiT|BacDrop|\bM3-?seq",
    "not_expression": r"\bTn-?seq|TraDIS|amplicon|\bSELEX|in vitro transcri|polysome|\bHi-?C\b|ChIP|\bDNA-?seq",
}
_LIBRARY_TYPE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in LIBRARY_TYPE_PATTERNS.items()}
STATUS_ORDER = ["no_geo_ids", "not_illumina", "not_transcriptomic", "no_fastq",
                "library_type", "low_reads"]


def ena_search(query: str, fields: list[str], path: str, log) -> None:
    data = urllib.parse.urlencode({"result": "read_run", "query": query, "fields": ",".join(fields),
                                   "format": "tsv", "limit": "0"}).encode()
    req = urllib.request.Request(ENA_SEARCH, data=data, headers={"User-Agent": "build_manifest"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(path, "wb") as fh:
        fh.write(resp.read())
    log(f"ENA: {query[:70]}{'...' if len(query) > 70 else ''} -> {path}")


def read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def write_tsv(path: str, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(rows)


def geo_ids(run: dict) -> tuple[str, str]:
    """(GSE, GSM) taken from the ENA aliases; empty strings when the run is not a GEO submission."""
    gse = re.match(r"GSE\d+", run.get("study_alias", "") or "")
    gsm = None
    for key in ("experiment_alias", "sample_alias", "run_alias"):
        gsm = re.match(r"GSM\d+", run.get(key, "") or "")
        if gsm:
            break
    return (gse.group(0) if gse else "", gsm.group(0) if gsm else "")


def library_type_match(run: dict, keys=("experiment_title", "sample_title")) -> str:
    text = " | ".join(run.get(k, "") or "" for k in keys)
    return ",".join(k for k, rx in _LIBRARY_TYPE_RE.items() if rx.search(text))


def is_ecoli(tax_id: str, name: str, ecoli_tax_ids: set) -> bool:
    return tax_id in ecoli_tax_ids or (name or "").startswith("Escherichia coli")


def classify(run: dict, min_reads: int) -> list[str]:
    flags = []
    if not run["GSE"] or not run["GSM"]:
        flags.append("no_geo_ids")
    if run["instrument_platform"] != "ILLUMINA":
        flags.append("not_illumina")
    if run["library_source"] != "TRANSCRIPTOMIC":
        flags.append("not_transcriptomic")
    if not run["fastq_ftp"]:
        flags.append("no_fastq")
    if run["library_type_match"]:
        flags.append("library_type")
    if int(run["read_count"] or 0) < min_reads:
        flags.append("low_reads")
    return flags


def build(out_dir: str, min_reads: int, refresh: bool, atlas_samples: str | None, old_qc: str | None,
          log=lambda m: print(m, file=sys.stderr)) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    runs_path, study_runs_path = os.path.join(out_dir, "ena_runs.tsv"), os.path.join(out_dir, "ena_study_runs.tsv")
    if refresh or not os.path.exists(runs_path):
        ena_search(ECOLI_QUERY, RUN_FIELDS, runs_path, log)
        with open(os.path.join(out_dir, "ena_download_date.txt"), "w") as fh:
            fh.write(datetime.date.today().isoformat() + "\n")
    runs = read_tsv(runs_path)
    for r in runs:
        r["GSE"], r["GSM"] = geo_ids(r)
        r["library_type_match"] = library_type_match(r)
        study = library_type_match(r, ("study_title",))
        r["review"] = f"study_title:{study}" if study and not r["library_type_match"] else ""
    ecoli_tax_ids = {r["tax_id"] for r in runs}

    # all runs of the GEO studies, to find studies with other organisms
    geo_studies = sorted({r["study_accession"] for r in runs if r["GSE"]})
    if refresh or not os.path.exists(study_runs_path):
        parts = []
        for i in range(0, len(geo_studies), 40):
            part = study_runs_path + ".part"
            ena_search(" OR ".join(f'study_accession="{s}"' for s in geo_studies[i:i + 40]), STUDY_RUN_FIELDS, part, log)
            parts.extend(read_tsv(part))
            os.remove(part)
        write_tsv(study_runs_path, parts, STUDY_RUN_FIELDS)
    other = defaultdict(Counter)
    for r in read_tsv(study_runs_path):
        if not is_ecoli(r["tax_id"], r["scientific_name"], ecoli_tax_ids):
            other[r["study_accession"]][r["scientific_name"]] += 1
    mixed = set(other)

    atlas = set()
    if atlas_samples:
        with open(atlas_samples, encoding="utf-8") as fh:
            atlas = set(fh.readline().rstrip("\n").split("\t")[1:])
    qc = {}
    if old_qc:
        for row in read_tsv(old_qc):
            qc[row["sample"]] = row
    for r in runs:
        flags = classify(r, min_reads)
        if r["study_accession"] in mixed:
            r["review"] = ",".join(x for x in ("mixed_organism_study", r["review"]) if x)
        r["flags"] = ",".join(flags)
        r["status"] = flags[0] if flags else "candidate"
        key = f"{r['GSM']}_{r['run_accession']}"
        r["in_atlas_v1"] = "yes" if key in atlas else "no"
        frac = qc.get(key, {}).get("strandedness_fraction", "")
        r["old_qc_reverse_fraction"] = frac if frac not in ("", "NA") else ""
        r["in_old_qc"] = "yes" if key in qc else "no"

    run_cols = ["run_accession", "GSM", "GSE", "status", "flags", "library_type_match", "review", "in_atlas_v1", "in_old_qc",
                "old_qc_reverse_fraction"] + [f for f in RUN_FIELDS if f != "run_accession"]
    write_tsv(os.path.join(out_dir, "runs.tsv"), runs, run_cols)

    # samples: runs merged per GSM
    by_gsm = defaultdict(list)
    for r in runs:
        if r["GSM"] and r["GSE"]:
            by_gsm[r["GSM"]].append(r)
    samples = []
    for gsm, rs in sorted(by_gsm.items()):
        ok = [r for r in rs if r["status"] == "candidate"]
        layouts = sorted({r["library_layout"] for r in rs})
        fr = [float(r["old_qc_reverse_fraction"]) for r in rs if r["old_qc_reverse_fraction"]]
        old_strand = ""
        if fr:
            mean = sum(fr) / len(fr)
            old_strand = "reverse" if mean > 0.8 else "forward" if mean < 0.2 else "unstranded"
        samples.append({
            "GSM": gsm, "GSE": rs[0]["GSE"],
            "status": ("candidate" if len(layouts) == 1 else "mixed_layout") if len(ok) == len(rs) else
                      min((r["status"] for r in rs if r["status"] != "candidate"), key=STATUS_ORDER.index),
            "n_runs": len(rs), "runs": ",".join(r["run_accession"] for r in rs), "layout": "/".join(layouts),
            "reads": sum(int(r["read_count"] or 0) for r in rs), "scientific_name": rs[0]["scientific_name"],
            "tax_id": rs[0]["tax_id"], "library_selection": rs[0]["library_selection"],
            "instrument_model": rs[0]["instrument_model"], "first_public": min(r["first_public"] for r in rs),
            "in_atlas_v1": "yes" if any(r["in_atlas_v1"] == "yes" for r in rs) else "no",
            "old_qc_strand": old_strand, "review": rs[0]["review"], "sample_title": rs[0]["sample_title"], "study_title": rs[0]["study_title"],
        })
    sample_cols = ["GSM", "GSE", "status", "n_runs", "runs", "layout", "reads", "scientific_name", "tax_id",
                   "library_selection", "instrument_model", "first_public", "in_atlas_v1", "old_qc_strand",
                   "review", "sample_title", "study_title"]
    write_tsv(os.path.join(out_dir, "samples.tsv"), samples, sample_cols)

    by_gse = defaultdict(list)
    for s in samples:
        by_gse[s["GSE"]].append(s)
    studies = []
    for gse, ss in sorted(by_gse.items()):
        st = Counter(s["status"] for s in ss)
        acc = by_gsm[ss[0]["GSM"]][0]["study_accession"]
        studies.append({"GSE": gse, "study_accession": acc, "n_samples": len(ss), "n_candidate": st["candidate"],
                        "n_in_atlas_v1": sum(s["in_atlas_v1"] == "yes" for s in ss),
                        "status_counts": ";".join(f"{k}={v}" for k, v in st.most_common()),
                        "other_organisms": ";".join(f"{k}={v}" for k, v in other.get(acc, Counter()).most_common(3)),
                        "first_public": min(s["first_public"] for s in ss), "study_title": ss[0]["study_title"]})
    write_tsv(os.path.join(out_dir, "studies.tsv"), studies, list(studies[0]) if studies else ["GSE"])

    lines = []
    add = lines.append
    add(f"ENA query: {ECOLI_QUERY}")
    add(f"runs: {len(runs)}; GEO runs: {sum(bool(r['GSE'] and r['GSM']) for r in runs)}; "
        f"GEO samples: {len(samples)}; GEO studies: {len(studies)}")
    add("\nrun status (first rule that applies):")
    for k, v in Counter(r["status"] for r in runs).most_common():
        add(f"  {k:22s} {v}")
    add("\nGEO sample status:")
    for k, v in Counter(s["status"] for s in samples).most_common():
        add(f"  {k:22s} {v}")
    cand = [s for s in samples if s["status"] == "candidate"]
    add(f"\ncandidate samples: {len(cand)} in {len({s['GSE'] for s in cand})} studies; "
        f"layout {dict(Counter(s['layout'] for s in cand))}; with more than one run {sum(s['n_runs'] > 1 for s in cand)}")
    for label, key in (("study also holds another organism", "mixed_organism_study"),
                       ("library type named in the study title only", "study_title:")):
        hit = [s for s in cand if key in s["review"]]
        add(f"candidates to review, {label}: {len(hit)} samples in {len({s['GSE'] for s in hit})} studies")
    add("library type matches among GEO runs (a run can match several): " + str(dict(Counter(
        k for r in runs if r["GSE"] and r["GSM"] for k in r["library_type_match"].split(",") if k))))
    if atlas:
        in_atlas = [s for s in samples if s["in_atlas_v1"] == "yes"]
        found = {f"{r['GSM']}_{r['run_accession']}" for r in runs}
        add(f"\natlas v1: {len(atlas)} run-level samples; {len(atlas & found)} found in the ENA list "
            f"({len(atlas - found)} not found), in {len(in_atlas)} GEO samples")
        add("  status now of the GEO samples that are in atlas v1: " + str(dict(Counter(s["status"] for s in in_atlas))))
        add("  old-QC strand of atlas v1 samples that are still candidates: " + str(dict(Counter(
            s["old_qc_strand"] or "no record" for s in in_atlas if s["status"] == "candidate"))))
        add(f"  candidates not in atlas v1: {sum(s['in_atlas_v1'] == 'no' for s in cand)} samples in "
            f"{len({s['GSE'] for s in cand if s['in_atlas_v1'] == 'no'})} studies "
            f"({len({s['GSE'] for s in cand} - {s['GSE'] for s in in_atlas})} studies with no sample in atlas v1)")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return {"runs": runs, "samples": samples, "studies": studies}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-reads", type=int, default=500_000, help="runs with fewer reads are excluded")
    ap.add_argument("--refresh", action="store_true", help="download from ENA again even if the files exist")
    ap.add_argument("--atlas-samples", help="count table whose header lists the atlas v1 samples (GSM_SRR)")
    ap.add_argument("--old-qc", help="quality_control.tsv of the atlas v1 screen")
    args = ap.parse_args(argv)
    build(args.out_dir, args.min_reads, args.refresh, args.atlas_samples, args.old_qc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
