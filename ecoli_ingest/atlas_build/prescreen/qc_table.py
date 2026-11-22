#!/usr/bin/env python
"""Maintain the sample QC table: prescreen metrics first, full-run metrics for samples that are processed.

One row per GEO sample (GSM). Columns:
  identity   GSM, GSE, runs, n_runs, layout, reads, bases, scientific_name, tax_id, library_strategy,
             library_source, library_selection, instrument_platform, instrument_model, first_public,
             manifest_status, review, in_atlas_v1, twin_of (the GSM a technical twin was made from; a
             twin's row is keyed <GSM>__<variant>), study_candidates (candidate samples in the study) and large_study
             (yes when that is at least LARGE_STUDY_MIN: statistics over the atlas should be reported
             with and without such a study)
  pre_*      metrics of the read-subsample prescreen (prescreen.py), with pre_run, pre_spots,
             pre_source, pre_date, pre_pipeline, pre_error
  pre_decision, pre_reasons   pass / fail / incomplete from the thresholds (qc_metrics.decide)
  full_*     the same metrics from a complete inGEST run, with full_sample, full_counted,
             full_not_counted_reason (the QC gate or species verdict that stopped the sample),
             full_results_dir, full_date, full_pipeline; and the values only a full run has, from
             the sample's QC record (qc_metrics.RECORD_COLUMNS: read-length and fragment-size
             quantiles, RNA family shares, reads on CDS features, off-reference share, substitution
             rate, gene-body slope, DNA-background values, species verdict)
  in_model   name of the atlas model the sample is part of

    qc_table.py merge-prescreen --manifest work/atlas_manifest --samples-dir work/prescreen/samples --table atlas_data/sample_qc.tsv
    qc_table.py add-full --results <run dir>/results --config <run dir>/config.yaml --table atlas_data/sample_qc.tsv
    qc_table.py add-full --batches work/atlas_full/results --table atlas_data/sample_qc.tsv     # every <batch>/results
    qc_table.py decide --table atlas_data/sample_qc.tsv [--min-pct-aligned 50 ...]
    qc_table.py mark-model --samples samples.txt --model atlas_v2 --table atlas_data/sample_qc.tsv
    qc_table.py mark-model --metadata atlas_data/metadata_v2.tsv --min-cds-reads 200000 --max-dna-fraction 0.10 --model atlas_v2 --table atlas_data/sample_qc.tsv
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
import gc_bias  # noqa: E402  (workflow script; qc_metrics puts the scripts on the path)

IDENTITY = ["GSM", "GSE", "runs", "n_runs", "layout", "reads", "bases", "scientific_name", "tax_id",
            "library_strategy", "library_source", "library_selection", "instrument_platform", "instrument_model",
            "first_public", "manifest_status", "review", "in_atlas_v1", "study_candidates", "large_study", "twin_of"]
RUN_FIELDS = ("library_strategy", "library_source", "instrument_platform")      # taken from the manifest's runs.tsv
LARGE_STUDY_MIN = 500      # candidate samples; such a study can dominate statistics, so report with and without it
PRE_INFO = ["pre_run", "pre_spots", "pre_source", "pre_date", "pre_pipeline", "pre_error"]
FULL_INFO = ["full_sample", "full_counted", "full_not_counted_reason", "full_results_dir", "full_date", "full_pipeline"]
COLUMNS = (IDENTITY + PRE_INFO + [f"pre_{m}" for m in qc_metrics.METRICS] + ["pre_decision", "pre_reasons"]
           + FULL_INFO + [f"full_{m}" for m in qc_metrics.METRICS] + [f"full_{m}" for m in qc_metrics.RECORD_COLUMNS]
           + ["in_model"])
TWIN_RE = re.compile(r"^(GSM\d+)__(\w+)$")      # technical twins are named <GSM>__<variant>


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
                    if c in s or c == "manifest_status":
                        row[c] = s.get("status" if c == "manifest_status" else c, "") or ""
        runs_path = os.path.join(manifest_dir, "runs.tsv")
        if os.path.exists(runs_path):        # fields that the per-sample table does not carry
            per_sample: dict = {}
            with open(runs_path, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if r.get("GSM") in rows:
                        d = per_sample.setdefault(r["GSM"], {k: set() for k in RUN_FIELDS} | {"bases": 0})
                        for k in RUN_FIELDS:
                            if r.get(k):
                                d[k].add(r[k])
                        d["bases"] += int(r.get("base_count") or 0)
            for gsm, d in per_sample.items():
                for k in RUN_FIELDS:
                    rows[gsm][k] = "/".join(sorted(d[k]))
                rows[gsm]["bases"] = str(d["bases"])
        per_study = Counter(r["GSE"] for r in rows.values() if r["manifest_status"] == "candidate")
        for row in rows.values():
            row["study_candidates"] = str(per_study.get(row["GSE"], 0))
            row["large_study"] = "yes" if per_study.get(row["GSE"], 0) >= LARGE_STUDY_MIN else "no"
    n = 0
    for path in sorted(glob.glob(os.path.join(samples_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        row = _row(rows, rec["GSM"])
        row["GSE"] = row["GSE"] or rec.get("GSE", "")
        row.update(pre_run=_text(rec.get("run")), pre_spots=_text(rec.get("spots")), pre_source=_text(rec.get("source")),
                   pre_date=_text(rec.get("date")), pre_pipeline=_text(rec.get("pipeline")),
                   pre_error=_text(rec.get("error")))
        metrics = dict(rec.get("metrics", {}))
        metrics.setdefault("centrifuge_reads", rec.get("centrifuge_reads"))
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


def not_counted_reasons(results_dir: str) -> dict:
    """sample -> why it was not counted: the QC gate that stopped it before alignment
    (`qc_failures.json`, written next to the results directory by the atlas recount) or the reason
    in strandedness.tsv."""
    reasons = {}
    failures = os.path.join(os.path.dirname(os.path.abspath(results_dir)), "qc_failures.json")
    if os.path.exists(failures):
        with open(failures, encoding="utf-8") as fh:
            for sample, info in (json.load(fh) or {}).items():
                reasons[sample] = str(info.get("verdict") or info.get("reason") or "qc_failed")
    return reasons


def add_full_rows(rows: dict, results_dir: str, config: str | None, pipeline: str = "", gc: dict | None = None) -> tuple[int, list]:
    """Fill the full_* columns of `rows` from one results directory. A sample that an earlier
    directory counted is not replaced by a result in which it was not counted (a sample re-run in
    a later batch keeps its counted result whatever the order). Returns (samples added, those kept)."""
    names = sample_accessions(config) if config else {}
    counted, reasons = {}, not_counted_reasons(results_dir)
    strand_table = os.path.join(results_dir, "strandedness.tsv")
    if os.path.exists(strand_table):
        with open(strand_table, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                counted[r["sample"]] = r["counted"]
                if r.get("not_counted_reason"):
                    reasons[r["sample"]] = r["not_counted_reason"]
    n = 0
    skipped, kept = [], []
    for d in sorted(glob.glob(os.path.join(results_dir, "*", "fastp"))):
        sample = os.path.basename(os.path.dirname(d))
        twin = TWIN_RE.match(sample)
        gsm = names.get(sample) or (sample if re.fullmatch(r"GSM\d+", sample) else None) or (sample if twin else None)
        if not gsm:
            skipped.append(sample)
            continue
        files = full_run_files(results_dir, sample)
        metrics = qc_metrics.collect(files)
        # species statistics of a full run are in the sample's QC record (workflow rule species_qc)
        record = os.path.join(results_dir, sample, "qc", f"{sample}.qc.json")
        record_data = None
        if os.path.exists(record):
            with open(record, encoding="utf-8") as fh:
                record_data = json.load(fh)
            # a record made before the GC bias: the fit from the sample's coverage track (`gc` = {bed, genome})
            cov = os.path.join(results_dir, sample, "coverage", f"{sample}.cov100.npz")
            if gc and not record_data.get("gc_bias") and os.path.exists(cov):
                record_data["gc_bias"] = gc_bias.fit_files(cov, gc["bed"], None, gc["genome"])
            species = record_data.get("species") or {}
            metrics.update({k: v for k, v in species.items() if k in qc_metrics.METRICS})
        row = _row(rows, gsm)
        here = os.path.abspath(results_dir)
        if row["full_counted"] == "yes" and counted.get(sample) != "yes" and row["full_results_dir"] != here:
            kept.append(gsm)
            continue
        if twin:
            row["twin_of"] = twin.group(1)
            row["GSE"] = row["GSE"] or rows.get(twin.group(1), {}).get("GSE", "")
        mtime = max((os.path.getmtime(p) for p in files.values() if os.path.exists(p)), default=None)
        row.update(full_sample=sample, full_counted=counted.get(sample, "") or ("no" if sample in reasons else ""),
                   full_not_counted_reason="" if counted.get(sample) == "yes" else reasons.get(sample, ""),
                   full_results_dir=here,
                   full_date=datetime.date.fromtimestamp(mtime).isoformat() if mtime else "", full_pipeline=pipeline)
        for m in qc_metrics.METRICS:
            row[f"full_{m}"] = _text(metrics.get(m))
        extra = qc_metrics.record_columns(record_data) if record_data else {}
        for m in qc_metrics.RECORD_COLUMNS:
            row[f"full_{m}"] = _text(extra.get(m))
        n += 1
    if skipped:
        print(f"no GSM known for {len(skipped)} samples (give --config): {', '.join(skipped[:5])}", file=sys.stderr)
    return n, kept


def gc_inputs(bed: str | None, fasta: str | None) -> dict | None:
    """{bed, genome} for the GC-bias fit of records made before it; None without a FASTA."""
    if not fasta:
        return None
    if not bed:
        raise SystemExit("--genome needs --bed (the counting annotation BED)")
    return {"bed": bed, "genome": gc_bias.read_fasta(fasta)}


def add_full(table: str, results_dir: str, config: str | None, pipeline: str = "", gc: dict | None = None) -> dict:
    rows = read_table(table)
    n, kept = add_full_rows(rows, results_dir, config, pipeline, gc)
    write_table(rows, table)
    print(f"added full-run metrics for {n} samples from {results_dir}"
          + (f"; {len(kept)} kept their counted result from another run" if kept else ""), file=sys.stderr)
    return rows


def add_full_batches(table: str, batches_dir: str, pipeline: str = "", gc: dict | None = None) -> dict:
    """add-full for every <batches_dir>/<batch>/results (with <batch>/config.yaml), in one pass."""
    rows = read_table(table)
    total, kept, n_batches = 0, [], 0
    for results_dir in sorted(glob.glob(os.path.join(batches_dir, "*", "results"))):
        config = os.path.join(os.path.dirname(results_dir), "config.yaml")
        n, k = add_full_rows(rows, results_dir, config if os.path.exists(config) else None, pipeline, gc)
        total, kept, n_batches = total + n, kept + k, n_batches + 1
    write_table(rows, table)
    print(f"added full-run metrics for {total} samples from {n_batches} batches in {batches_dir}"
          + (f"; kept the counted result of {len(kept)}: {', '.join(sorted(kept))}" if kept else ""), file=sys.stderr)
    return rows


TRUE = {"true", "yes", "1"}


def model_gate(row: dict, usable: bool, min_cds_reads: int, max_dna_fraction: float | None = None) -> str:
    """Why a sample is not part of the model ('' = it is): the rule of the atlas recount. With
    `max_dna_fraction`, a sample whose genomic-DNA estimate (`full_dna_fraction_est`) is above it is
    left out as `high_dna`; a sample without an estimate passes."""
    if not usable:
        return "not_usable"
    if not row["full_date"]:
        return "no_full_run"
    if row["full_counted"] != "yes":
        return "not_counted:" + (row["full_not_counted_reason"] or "unknown")
    try:
        cds = float(row["full_reads_CDS"])
    except ValueError:
        return "no_qc_record"
    if cds < min_cds_reads:
        return "low_mrna"
    if max_dna_fraction is not None:
        try:
            if float(row["full_dna_fraction_est"]) > max_dna_fraction:
                return "high_dna"
        except ValueError:
            pass
    return ""


def mark_model_by_rule(table: str, metadata: str, model: str, min_cds_reads: int, max_dna_fraction: float | None = None) -> dict:
    """in_model = `model` for the samples that are usable in the metadata table, were counted by the
    full run, have at least `min_cds_reads` reads on CDS features and a genomic-DNA estimate of at
    most `max_dna_fraction`; cleared for the others that carried this model's name."""
    rows = read_table(table)
    with open(metadata, encoding="utf-8", newline="") as fh:
        usable = {r["GSM"] for r in csv.DictReader(fh, delimiter="\t") if r.get("usable", "").strip().lower() in TRUE}
    why = Counter()
    for gsm, row in rows.items():
        if row["twin_of"]:
            continue
        reason = model_gate(row, gsm in usable, min_cds_reads, max_dna_fraction)
        if not reason:
            row["in_model"] = model
        elif row["in_model"] == model:
            row["in_model"] = ""
        if gsm in usable:
            why[reason or "in_model"] += 1
    missing = sorted(usable - set(rows))
    write_table(rows, table)
    print(f"{model}: {why['in_model']} of {len(usable)} usable samples are in the model "
          f"(at least {min_cds_reads} reads on CDS features"
          + ("" if max_dna_fraction is None else f", DNA estimate at most {max_dna_fraction:g}") + ")", file=sys.stderr)
    for reason, n in sorted(why.items(), key=lambda kv: -kv[1]):
        if reason != "in_model":
            print(f"  left out, {reason}: {n}", file=sys.stderr)
    if missing:
        print(f"  usable in the metadata but not in the table: {len(missing)} ({', '.join(missing[:5])})", file=sys.stderr)
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
    large = Counter(r["GSE"] for r in screened if r["large_study"] == "yes" and r["pre_decision"] == "pass")
    if large:
        rest = [r for r in screened if r["large_study"] != "yes"]
        lines.append("large studies (report statistics with and without them): "
                     + ", ".join(f"{g} {n} passing" for g, n in large.most_common()))
        lines.append("without them: " + str(sum(r["pre_decision"] == "pass" for r in rest)) + " passing of "
                     + str(len(rest)) + " screened, in "
                     + str(len({r["GSE"] for r in rest if r["pre_decision"] == "pass"})) + " studies")
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
    f.add_argument("--results", default=None, help="inGEST results directory")
    f.add_argument("--batches", default=None,
                   help="directory of batches: every <batch>/results is added, with <batch>/config.yaml")
    f.add_argument("--config", default=None, help="the run's config.yaml (maps sample names to GSM accessions)")
    f.add_argument("--genome", default=None, help="reference FASTA: fit the GC bias from the coverage track for records "
                   "made without it (needs --bed)")
    f.add_argument("--bed", default=None, help="counting annotation BED, with --genome")
    d = sub.add_parser("decide", help="recompute pre_decision with (new) thresholds")
    _threshold_args(d)
    k = sub.add_parser("mark-model", help="record which samples are in a trained atlas model")
    k.add_argument("--samples", default=None, help="file with one GSM (or GSM_SRR) per line")
    k.add_argument("--metadata", default=None,
                   help="metadata table with a `usable` column: mark by rule (usable, counted by the full run, "
                        "--min-cds-reads) instead of a sample list")
    k.add_argument("--min-cds-reads", type=int, default=200000,
                   help="with --metadata: fewest reads on CDS features (full_reads_CDS) of a sample in the model")
    k.add_argument("--max-dna-fraction", type=float, default=0.10,
                   help="with --metadata: largest genomic-DNA estimate (full_dna_fraction_est) of a sample in the model (default 0.10)")
    k.add_argument("--model", required=True)
    sub.add_parser("summary")
    for p in sub.choices.values():
        p.add_argument("--table", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "merge-prescreen":
        merge_prescreen(args.table, args.manifest, args.samples_dir, _thresholds(args))
    elif args.cmd == "add-full":
        if bool(args.results) == bool(args.batches):
            ap.error("add-full needs --results or --batches")
        if args.batches:
            add_full_batches(args.table, args.batches, git_commit(), gc_inputs(args.bed, args.genome))
        else:
            add_full(args.table, args.results, args.config, git_commit(), gc_inputs(args.bed, args.genome))
    elif args.cmd == "decide":
        rows = read_table(args.table)
        for row in rows.values():
            set_decision(row, _thresholds(args))
        write_table(rows, args.table)
    elif args.cmd == "mark-model" and args.metadata:
        mark_model_by_rule(args.table, args.metadata, args.model, args.min_cds_reads, args.max_dna_fraction)
    elif args.cmd == "mark-model":
        if not args.samples:
            ap.error("mark-model needs --samples or --metadata")
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
