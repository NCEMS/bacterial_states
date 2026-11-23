#!/usr/bin/env python3
"""Study and sample text of the atlas studies from their GEO SOFT files, for `ecoli_atlas view`.

Usage: study_text.py SOFT_DIR METADATA_TSV OUT_PREFIX [--qc sample_qc.tsv] [--studies studies.tsv]

METADATA_TSV is metadata_v2.tsv (its GSM and GSE columns choose the samples). Writes
  OUT_PREFIX_studies.tsv      GSE, title, summary, overall_design, type, pubmed_ids, bioproject, sra,
                              contributors, submission_date, first_public, n_samples, study_flag, curator_notes
  OUT_PREFIX_sample_text.tsv  GSM, GSE, title, source_name, characteristics, description, runs, layout, instrument
Curator notes come from labels/<GSE>.json and study flags from study_flags.tsv next to this script
(LABEL_DIR overrides the labels folder). --qc adds the SRA runs, layout and instrument of each sample;
--studies (work/atlas_manifest/studies.tsv) adds the date the study became public.
Tabs and newlines inside fields are replaced by spaces.
"""
import argparse
import csv
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geo_digest import parse  # noqa: E402

STUDY_COLS = ["GSE", "title", "summary", "overall_design", "type", "pubmed_ids", "bioproject", "sra", "contributors",
              "submission_date", "first_public", "n_samples", "study_flag", "curator_notes"]
SAMPLE_COLS = ["GSM", "GSE", "title", "source_name", "characteristics", "description", "runs", "layout", "instrument"]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_tsv(path, key):
    with open(path, encoding="utf-8", newline="") as fh:
        return {r[key]: r for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)}


def relation(series, prefix):
    """Accession of a `!Series_relation = <prefix>: <url>` line (last path or term= part)."""
    for v in series.get("relation", []):
        if v.startswith(prefix + ":"):
            url = v.split(":", 1)[1].strip()
            return re.split(r"[/=]", url)[-1]
    return ""


def study_row(gse, series, n_samples, flags, labels_dir, public):
    notes = ""
    lab = labels_dir / f"{gse}.json"
    if lab.exists():
        notes = json.loads(lab.read_text(encoding="utf-8")).get("notes", "")
    return {"GSE": gse, "title": clean(" ".join(series.get("title", []))),
            "summary": clean(" ".join(series.get("summary", []))),
            "overall_design": clean(" ".join(series.get("overall_design", []))),
            "type": clean("; ".join(series.get("type", []))),
            "pubmed_ids": ",".join(clean(p) for p in series.get("pubmed_id", [])),
            "bioproject": relation(series, "BioProject"), "sra": relation(series, "SRA"),
            "contributors": clean(", ".join(series.get("contributor", []))),
            "submission_date": clean(" ".join(series.get("submission_date", []))),
            "first_public": public.get(gse, ""), "n_samples": n_samples,
            "study_flag": flags.get(gse, ""), "curator_notes": clean(notes)}


def sample_row(gsm, gse, s, qc):
    chars = "; ".join(clean(v) for v in s.get("characteristics_ch1", []))
    q = qc.get(gsm, {})
    return {"GSM": gsm, "GSE": gse, "title": clean(" ".join(s.get("title", []))),
            "source_name": clean(" ".join(s.get("source_name_ch1", []))), "characteristics": chars,
            "description": clean(" ".join(s.get("description", []))),
            "runs": q.get("runs", ""), "layout": q.get("full_layout") or q.get("layout", ""),
            "instrument": q.get("full_instrument_model") or q.get("instrument_model", "")}


def write(path, cols, rows):
    """Plain TSV without quoting (fields hold no tabs or newlines after `clean`)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(r.get(c, "") for c in cols) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("soft_dir")
    p.add_argument("metadata")
    p.add_argument("out_prefix")
    p.add_argument("--qc")
    p.add_argument("--studies")
    a = p.parse_args(argv)
    soft_dir = pathlib.Path(a.soft_dir)
    labels_dir = pathlib.Path(os.environ.get("LABEL_DIR", HERE / "labels"))
    meta = read_tsv(a.metadata, "GSM")
    by_study = {}
    for gsm, r in meta.items():
        by_study.setdefault(r["GSE"], []).append(gsm)
    qc = read_tsv(a.qc, "GSM") if a.qc else {}
    public = {g: r.get("first_public", "") for g, r in read_tsv(a.studies, "GSE").items()} if a.studies else {}
    flag_file = HERE / "study_flags.tsv"
    flags = {}
    if flag_file.exists():
        for g, r in read_tsv(flag_file, "GSE").items():
            flags[g] = r["flag"]
    studies, samples, missing = [], [], []
    for gse in sorted(by_study):
        path = soft_dir / f"{gse}.soft"
        if not path.exists():
            missing.append(gse)
            continue
        series, soft_samples = parse(path)
        studies.append(study_row(gse, series, len(by_study[gse]), flags, labels_dir, public))
        for gsm in by_study[gse]:
            samples.append(sample_row(gsm, gse, soft_samples.get(gsm, {}), qc))
    for r in studies + samples:
        for k, v in r.items():
            r[k] = clean(str(v))
    out = pathlib.Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(f"{out}_studies.tsv", STUDY_COLS, studies)
    write(f"{out}_sample_text.tsv", SAMPLE_COLS, samples)
    print(f"{len(studies)} studies, {len(samples)} samples; no SOFT file for {len(missing)} studies"
          + (f": {' '.join(missing[:10])}" if missing else ""))


if __name__ == "__main__":
    main()
