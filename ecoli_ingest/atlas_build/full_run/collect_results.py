#!/usr/bin/env python3
"""Join the count tables of the atlas recount batches into one table for the atlas model.

Reads every `<results>/<batch>/results/` written by slurm_batch.sh and writes, under --out-prefix:

    <prefix>_counts.tsv             Feature_ID, then one column per counted sample (GSM)
    <prefix>_antisense_counts.tsv   the same from the antisense counting pass
    <prefix>_samples.tsv            one row per sample of every batch: GSM, batch, layout, strand_call,
                                    counted, not_counted_reason, assignments (column sum of its counts),
                                    reads_assigned (featureCounts `Assigned`), superseded_by

A sample that is in two batches (one re-run after a failed QC gate) is taken from the batch that
counted it; if two batches counted it, from the later one in the order given (`batches.tsv` order,
then --also). The other row stays in the sample table with `superseded_by`.

All batches must have the same features in the same order (same counting annotation); the script
stops if they do not. With `-O`, a read on two features is counted for both, so a sample's column
sum is at least its `Assigned` number.

    collect_results.py --results work/atlas_full/results --batches work/atlas_full/batches.tsv \\
        [--also PE_R01] --out-prefix atlas_data/atlas_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

ANNOTATION_COLUMNS = ["Geneid", "Chr", "Start", "End", "Strand", "Length"]
SAMPLE_COLUMNS = ["GSM", "batch", "layout", "strand_call", "counted", "not_counted_reason", "assignments",
                  "reads_assigned", "superseded_by"]


def sample_of(column: str) -> str:
    """results/<sample>/vg/<sample>_sort.bam -> <sample>."""
    return re.sub(r"_sort\.bam$", "", os.path.basename(column))


def read_featurecounts(path: str) -> tuple[list, list, np.ndarray]:
    """(annotation rows, sample names, counts[features, samples]) of a featureCounts table."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh if not line.startswith("#")]
    header, body = rows[0], rows[1:]
    if header[:6] != ANNOTATION_COLUMNS:
        raise SystemExit(f"{path}: unexpected header {header[:6]}")
    counts = np.array([r[6:] for r in body], dtype=np.int64).reshape(len(body), len(header) - 6)
    return [r[:6] for r in body], [sample_of(c) for c in header[6:]], counts


def read_assigned(path: str) -> dict:
    """sample -> `Assigned` of a featureCounts summary file."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    names = [sample_of(c) for c in rows[0][1:]]
    for r in rows[1:]:
        if r[0] == "Assigned":
            return dict(zip(names, (int(x) for x in r[1:])))
    return {}


def read_batch(batch_dir: str) -> dict:
    """Everything of one batch: counts of both passes and a row per sample."""
    res = os.path.join(batch_dir, "results")
    annotation, names, sense = read_featurecounts(os.path.join(res, "all_samples_raw_counts.txt"))
    annotation2, names2, anti = read_featurecounts(os.path.join(res, "all_samples_antisense_counts.txt"))
    if annotation2 != annotation or names2 != names:
        raise SystemExit(f"{batch_dir}: the two count tables differ in features or samples")
    assigned = read_assigned(os.path.join(res, "all_samples_raw_counts.txt.summary"))
    reasons = {}
    failures = os.path.join(batch_dir, "qc_failures.json")
    if os.path.exists(failures):
        with open(failures, encoding="utf-8") as fh:
            reasons = {s: str(v.get("verdict") or v.get("reason") or "qc_failed") for s, v in (json.load(fh) or {}).items()}
    samples = {}
    with open(os.path.join(res, "strandedness.tsv"), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            samples[r["sample"]] = {"layout": r["layout"], "strand_call": r["call"], "counted": r["counted"],
                                    "not_counted_reason": r.get("not_counted_reason", "")}
    for s, reason in reasons.items():        # stopped before alignment: not in strandedness.tsv
        samples.setdefault(s, {"layout": "", "strand_call": "", "counted": "no", "not_counted_reason": reason})
    counted = {s for s, v in samples.items() if v["counted"] == "yes"}
    if counted != set(names):
        raise SystemExit(f"{batch_dir}: counted samples and count columns differ: {sorted(counted ^ set(names))[:5]}")
    for i, s in enumerate(names):
        samples[s].update(assignments=int(sense[:, i].sum()), reads_assigned=assigned.get(s, ""))
        if assigned.get(s) is not None and samples[s]["assignments"] < assigned[s]:
            raise SystemExit(f"{batch_dir}: {s} has fewer assignments than assigned reads")
    return {"annotation": annotation, "names": names, "sense": sense, "anti": anti, "samples": samples}


def write_counts(path: str, features: list, names: list, counts: np.ndarray) -> None:
    with open(path + ".part", "w", encoding="utf-8", newline="") as fh:
        fh.write("Feature_ID\t" + "\t".join(names) + "\n")
        for feature, row in zip(features, counts):
            fh.write(feature + "\t" + "\t".join(map(str, row.tolist())) + "\n")
    os.replace(path + ".part", path)


def collect(results_dir: str, batches: list, out_prefix: str) -> dict:
    annotation = None
    chosen: dict[str, tuple] = {}            # GSM -> (batch, column index)
    data, sample_rows = {}, []
    for batch in batches:
        b = read_batch(os.path.join(results_dir, batch))
        if annotation is None:
            annotation = b["annotation"]
        elif b["annotation"] != annotation:
            raise SystemExit(f"{batch}: features differ from those of {batches[0]} (another counting annotation?)")
        data[batch] = b
        for i, s in enumerate(b["names"]):
            chosen[s] = (batch, i)           # a later batch that counted the sample wins
        for s, v in sorted(b["samples"].items()):
            sample_rows.append({"GSM": s, "batch": batch, **v})
    for row in sample_rows:
        used = chosen.get(row["GSM"])
        row["superseded_by"] = used[0] if used and used[0] != row["batch"] else ""
    names = sorted(chosen)
    for key, suffix in (("sense", "_counts.tsv"), ("anti", "_antisense_counts.tsv")):
        counts = np.empty((len(annotation), len(names)), dtype=np.int64)
        for j, s in enumerate(names):
            batch, i = chosen[s]
            counts[:, j] = data[batch][key][:, i]
        write_counts(out_prefix + suffix, [a[0] for a in annotation], names, counts)
    with open(out_prefix + "_samples.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SAMPLE_COLUMNS, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(sorted(sample_rows, key=lambda r: (r["GSM"], r["batch"])))
    superseded = sorted({r["GSM"] for r in sample_rows if r["superseded_by"]})
    all_samples = {r["GSM"] for r in sample_rows}
    return {"batches": len(batches), "features": len(annotation), "samples": len(all_samples), "counted": len(names),
            "not_counted": len(all_samples) - len(names), "superseded": superseded}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory with one <batch>/ per finished batch")
    ap.add_argument("--batches", default=None, help="batches.tsv of make_batches.py (default: every directory in --results)")
    ap.add_argument("--also", nargs="*", default=[], help="batches that are not in batches.tsv (re-runs), read last")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv)
    if args.batches:
        with open(args.batches, encoding="utf-8", newline="") as fh:
            batches = [r["batch"] for r in csv.DictReader(fh, delimiter="\t")]
    else:
        batches = sorted(d for d in os.listdir(args.results) if d not in args.also
                         and os.path.isdir(os.path.join(args.results, d, "results")))
    batches += [b for b in args.also if b not in batches]
    missing = [b for b in batches if not os.path.exists(os.path.join(args.results, b, "results", "all_samples_raw_counts.txt"))]
    if missing:
        raise SystemExit(f"no finished results for {len(missing)} batch(es): {', '.join(missing[:8])}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    info = collect(args.results, batches, args.out_prefix)
    print(f"{info['batches']} batches, {info['features']} features, {info['samples']} samples: "
          f"{info['counted']} counted, {info['not_counted']} not counted", file=sys.stderr)
    if info["superseded"]:
        print(f"taken from a later batch: {len(info['superseded'])} ({', '.join(info['superseded'])})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
