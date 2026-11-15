#!/usr/bin/env python
"""Maintain the sample QC table: prescreen metrics first, full-run metrics for samples that are processed.

One row per GEO sample (GSM). Columns:
  identity   GSM, GSE, runs, layout, scientific_name, tax_id, instrument_model, manifest_status,
             review, in_atlas_v1
  pre_*      metrics of the read-subsample prescreen (prescreen.py), with pre_run, pre_spots,
             pre_source, pre_date, pre_pipeline, pre_error
  pre_decision, pre_reasons   pass / fail / incomplete from the thresholds (qc_metrics.decide)
  full_*     the same metrics from a complete inGEST run, with full_sample, full_counted,
             full_results_dir, full_date, full_pipeline
  in_model   name of the atlas model the sample is part of

    qc_table.py merge-prescreen --manifest work/atlas_manifest --samples-dir work/prescreen/samples --table atlas_data/sample_qc.tsv
    qc_table.py add-full --results <run dir>/results --config <run dir>/config.yaml --table atlas_data/sample_qc.tsv
    qc_table.py decide --table atlas_data/sample_qc.tsv [--min-pct-ecoli 30 ...]
    qc_table.py mark-model --samples samples.txt --model atlas_v2 --table atlas_data/sample_qc.tsv
    qc_table.py summary --table atlas_data/sample_qc.tsv

Existing values are kept: each command only rewrites the columns it owns, for the samples it is given.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qc_metrics  # noqa: E402

IDENTITY = ["GSM", "GSE", "runs", "layout", "scientific_name", "tax_id", "instrument_model", "manifest_status",
            "review", "in_atlas_v1"]
PRE_INFO = ["pre_run", "pre_spots", "pre_source", "pre_date", "pre_pipeline", "pre_error"]
FULL_INFO = ["full_sample", "full_counted", "full_results_dir", "full_date", "full_pipeline"]
COLUMNS = (IDENTITY + PRE_INFO + [f"pre_{m}" for m in qc_metrics.METRICS] + ["pre_decision", "pre_reasons"]
           + FULL_INFO + [f"full_{m}" for m in qc_metrics.METRICS] + ["in_model"])


def read_table(path: str) -> dict:
    """GSM -> row (all columns present, missing ones empty); an absent file gives an empty table."""
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                rows[row["GSM"]] = {c: row.get(c, "") or "" for c in COLUMNS}
    return rows


def write_table(rows: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        for gsm in sorted(rows, key=lambda g: (rows[g]["GSE"], g)):
            w.writerow(rows[gsm])
    os.replace(path + ".part", path)


def _text(value) -> str:
    return "" if value is None else str(value)


def _row(rows: dict, gsm: str) -> dict:
    return rows.setdefault(gsm, {**{c: "" for c in COLUMNS}, "GSM": gsm})


def set_decision(row: dict, thresholds: dict | None = None) -> None:
    if not row["pre_date"]:
        row["pre_decision"], row["pre_reasons"] = "", ""
    elif row["pre_error"]:
        row["pre_decision"], row["pre_reasons"] = "error", ""
    else:
        decision, reasons = qc_metrics.decide({m: row[f"pre_{m}"] for m in qc_metrics.METRICS}, thresholds)
        row["pre_decision"], row["pre_reasons"] = decision, ",".join(reasons)


def merge_prescreen(table: str, manifest_dir: str | None, samples_dir: str, thresholds: dict | None = None) -> dict:
    rows = read_table(table)
    if manifest_dir:
        with open(os.path.join(manifest_dir, "samples.tsv"), encoding="utf-8", newline="") as fh:
            for s in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                row = _row(rows, s["GSM"])
                for c in IDENTITY:
                    row[c] = s.get("status" if c == "manifest_status" else c, "") or ""
    n = 0
    for path in sorted(glob.glob(os.path.join(samples_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        row = _row(rows, rec["GSM"])
        row["GSE"] = row["GSE"] or rec.get("GSE", "")
        row.update(pre_run=_text(rec.get("run")), pre_spots=_text(rec.get("spots")), pre_source=_text(rec.get("source")),
                   pre_date=_text(rec.get("date")), pre_pipeline=_text(rec.get("pipeline")),
                   pre_error=_text(rec.get("error")))
        metrics = rec.get("metrics", {})
        for m in qc_metrics.METRICS:
            row[f"pre_{m}"] = _text(metrics.get(m))
        set_decision(row, thresholds)
        n += 1
    write_table(rows, table)
    print(f"merged {n} prescreen records; table has {len(rows)} samples", file=sys.stderr)
    return rows


def sample_accessions(config_path: str) -> dict:
    """inGEST sample name -> GSM, from the run's config.yaml (`accession:` of each sample)."""
    import yaml
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    out = {}
    for name, info in (cfg.get("samples") or {}).items():
        acc = info.get("accession") if isinstance(info, dict) else None
        accs = acc if isinstance(acc, list) else [acc] if acc else []
        gsm = [a for a in accs if re.fullmatch(r"GSM\d+", str(a))]
        if len(gsm) == 1:
            out[name] = gsm[0]
    return out


def full_run_files(results_dir: str, sample: str) -> dict:
    d = os.path.join(results_dir, sample)
    return {"fastp": os.path.join(d, "fastp", f"{sample}_fastp.json"),
            "vg_stats": os.path.join(d, "vg", f"{sample}_giraffe.stats.txt"),
            "infer_experiment": os.path.join(d, "rseqc", f"{sample}_infer_experiment.txt"),
            "feature_overlap": os.path.join(d, "feature_overlap", f"{sample}_feature_overlap_mqc.tsv"),
            "kreport": os.path.join(d, "centrifuge", f"{sample}_report.txt")}


def add_full(table: str, results_dir: str, config: str | None, pipeline: str = "") -> dict:
    rows = read_table(table)
    names = sample_accessions(config) if config else {}
    counted = {}
    strand_table = os.path.join(results_dir, "strandedness.tsv")
    if os.path.exists(strand_table):
        with open(strand_table, encoding="utf-8", newline="") as fh:
            counted = {r["sample"]: r["counted"] for r in csv.DictReader(fh, delimiter="\t")}
    n = 0
    skipped = []
    for d in sorted(glob.glob(os.path.join(results_dir, "*", "fastp"))):
        sample = os.path.basename(os.path.dirname(d))
        gsm = names.get(sample) or (sample if re.fullmatch(r"GSM\d+", sample) else None)
        if not gsm:
            skipped.append(sample)
            continue
        files = full_run_files(results_dir, sample)
        metrics = qc_metrics.collect(files)
        row = _row(rows, gsm)
        mtime = max((os.path.getmtime(p) for p in files.values() if os.path.exists(p)), default=None)
        row.update(full_sample=sample, full_counted=counted.get(sample, ""),
                   full_results_dir=os.path.abspath(results_dir),
                   full_date=datetime.date.fromtimestamp(mtime).isoformat() if mtime else "", full_pipeline=pipeline)
        for m in qc_metrics.METRICS:
            row[f"full_{m}"] = _text(metrics.get(m))
        n += 1
    write_table(rows, table)
    print(f"added full-run metrics for {n} samples from {results_dir}", file=sys.stderr)
    if skipped:
        print(f"no GSM known for {len(skipped)} samples (give --config): {', '.join(skipped[:5])}", file=sys.stderr)
    return rows


def summary(table: str) -> str:
    rows = list(read_table(table).values())
    screened = [r for r in rows if r["pre_date"]]
    lines = [f"samples: {len(rows)}; prescreened: {len(screened)}; with full-run metrics: "
             f"{sum(bool(r['full_date']) for r in rows)}; in a model: {sum(bool(r['in_model']) for r in rows)}",
             "prescreen decision: " + str(dict(Counter(r["pre_decision"] for r in screened))),
             "reasons (a sample can have several): " + str(dict(Counter(
                 x for r in screened for x in r["pre_reasons"].split(",") if x))),
             "strand call: " + str(dict(Counter(r["pre_strand_call"] or "none" for r in screened))),
             "passing samples: " + str(sum(r["pre_decision"] == "pass" for r in screened)) + " in "
             + str(len({r["GSE"] for r in screened if r["pre_decision"] == "pass"})) + " studies"]
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def _threshold_args(p) -> None:
    for key, value in qc_metrics.DEFAULT_THRESHOLDS.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value)


def _thresholds(args) -> dict:
    return {k: getattr(args, k) for k in qc_metrics.DEFAULT_THRESHOLDS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge-prescreen", help="add the prescreen JSON records (and the manifest's samples)")
    m.add_argument("--samples-dir", required=True)
    m.add_argument("--manifest", default=None, help="manifest directory; adds a row for every sample in samples.tsv")
    _threshold_args(m)
    f = sub.add_parser("add-full", help="add metrics of a finished inGEST run")
    f.add_argument("--results", required=True, help="inGEST results directory")
    f.add_argument("--config", default=None, help="the run's config.yaml (maps sample names to GSM accessions)")
    d = sub.add_parser("decide", help="recompute pre_decision with (new) thresholds")
    _threshold_args(d)
    k = sub.add_parser("mark-model", help="record which samples are in a trained atlas model")
    k.add_argument("--samples", required=True, help="file with one GSM (or GSM_SRR) per line")
    k.add_argument("--model", required=True)
    sub.add_parser("summary")
    for p in sub.choices.values():
        p.add_argument("--table", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "merge-prescreen":
        merge_prescreen(args.table, args.manifest, args.samples_dir, _thresholds(args))
    elif args.cmd == "add-full":
        add_full(args.table, args.results, args.config, git_commit())
    elif args.cmd == "decide":
        rows = read_table(args.table)
        for row in rows.values():
            set_decision(row, _thresholds(args))
        write_table(rows, args.table)
    elif args.cmd == "mark-model":
        rows = read_table(args.table)
        with open(args.samples, encoding="utf-8") as fh:
            wanted = {line.split()[0].split("_")[0] for line in fh if line.strip()}
        hit = [g for g in wanted if g in rows]
        for g in hit:
            rows[g]["in_model"] = args.model
        write_table(rows, args.table)
        print(f"marked {len(hit)} of {len(wanted)} samples as in {args.model}", file=sys.stderr)
    if args.cmd in ("merge-prescreen", "decide", "summary"):
        print(summary(args.table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
