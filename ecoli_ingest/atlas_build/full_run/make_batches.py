#!/usr/bin/env python3
"""Split the atlas samples into inGEST run directories ("batches") for the full recount.

One batch holds samples of one library layout (the workflow treats a run directory with both
layouts as single-end), filled in study order up to --max-gb of sequence and --max-samples
samples. A sample larger than --max-gb is a batch of its own in the class `big`.

Writes under --out:
    batches.tsv              batch, GSE (list), layout, n_samples, n_studies, gb, class, cpus, hours
    batch_samples.tsv        batch, GSM, runs, bases
    std.list / big.list      batch names per class (one line per SLURM array task)
    batches/<batch>/config.yaml
    batches/<batch>/manifest/<GSM>.json   accession records made from the candidate manifest, so
                                          that no run directory has to ask NCBI/ENA for them

Samples: `pre_decision == pass` in the QC table; with --metadata only those marked `usable` there.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "ecoli_ingest", "workflow", "scripts"))
import sra_reads  # noqa: E402

TRUE = {"true", "yes", "1"}


def read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def accession_record(gsm: str, run_rows: list[dict], single: bool) -> dict:
    """The record `sra_reads.resolve_accession` caches, from manifest rows of the sample's runs."""
    entries = []
    for row in run_rows:
        e = sra_reads.ena_files_from_row(row, row["run_accession"])
        if single and e["layout"] == "PAIRED":
            # paired in the metadata, but the archive holds one mate (seen in the prescreen)
            e = dict(e, layout="SINGLE", files={"1": {"url": None, "md5": None, "bytes": None}})
        entries.append(e)
    layouts = {e["layout"] for e in entries}
    if len(layouts) != 1:
        raise ValueError(f"{gsm}: runs mix layouts {sorted(layouts)}")
    return {"accession": gsm, "kind": sra_reads.classify(gsm), "layout": entries[0]["layout"], "runs": entries,
            "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "atlas manifest"}


def pack(samples: list[dict], max_gb: float, max_samples: int) -> list[list[dict]]:
    """Fill batches in study order, so that a study stays together where it fits; a sample larger
    than max_gb gets a batch of its own."""
    bins: list[list[dict]] = []
    load = 0.0
    for s in sorted(samples, key=lambda s: (s["GSE"], s["GSM"])):
        if s["gb"] > max_gb:
            bins.insert(0, [s])
            continue
        if not bins or not load or load + s["gb"] > max_gb or len(bins[-1]) >= max_samples:
            bins.append([])
            load = 0.0
        bins[-1].append(s)
        load += s["gb"]
    return bins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc", default="atlas_data/sample_qc.tsv")
    ap.add_argument("--manifest", default="work/atlas_manifest", help="directory with runs.tsv")
    ap.add_argument("--metadata", default=None, help="metadata table with a `usable` column; keep only usable samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gb", type=float, default=200.0)
    ap.add_argument("--max-samples", type=int, default=150)
    ap.add_argument("--cpus", type=int, default=48)
    ap.add_argument("--fastp-min-kept", type=float, default=50.0,
                    help="fastp gate of the recount; kept low so that every sample is counted and the "
                         "trimming rule is applied afterwards from full_pct_kept in sample_qc.tsv")
    args = ap.parse_args(argv)

    qc = [r for r in read_tsv(args.qc) if r["pre_decision"] == "pass"]
    if args.metadata:
        usable = {r["GSM"] for r in read_tsv(args.metadata) if r.get("usable", "").strip().lower() in TRUE}
        qc = [r for r in qc if r["GSM"] in usable]
    runs_tsv = os.path.abspath(os.path.join(args.manifest, "runs.tsv"))
    runs_by_gsm: dict[str, list[dict]] = defaultdict(list)
    for r in read_tsv(runs_tsv):
        if r["status"] == "candidate" and r["GSM"]:
            runs_by_gsm[r["GSM"]].append(r)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in qc:
        rows = sorted(runs_by_gsm.get(r["GSM"], []), key=lambda x: x["run_accession"])
        if not rows:
            print(f"skipped {r['GSM']}: no candidate run in the manifest", file=sys.stderr)
            continue
        single = r["pre_layout"] == "SE" or r["layout"] == "SINGLE"
        rec = accession_record(r["GSM"], rows, single)
        gb = sum(int(x["base_count"] or 0) for x in rows) / 1e9
        groups[rec["layout"]].append({"GSM": r["GSM"], "GSE": r["GSE"], "gb": gb, "record": rec,
                                                  "runs": [x["run_accession"] for x in rows]})

    os.makedirs(os.path.join(args.out, "batches"), exist_ok=True)
    batch_rows, sample_rows = [], []
    for layout, samples in sorted(groups.items()):
        bins = pack(samples, args.max_gb, args.max_samples)
        for i, b in enumerate(bins, 1):
            name = f"{'PE' if layout == 'PAIRED' else 'SE'}_{i:03d}"
            gb = sum(s["gb"] for s in b)
            gses = sorted({s["GSE"] for s in b})
            gse = ",".join(gses)
            big = gb > args.max_gb
            # the pilot batches took 1.4-1.6 core-hours per Gb; 2.5 are allowed. One sample uses
            # at most 12 threads, so a very large sample sets its own lower bound
            hours = max(4, round(2 + max(gb * 2.5 / args.cpus, 0.45 * max(s["gb"] for s in b))))
            d = os.path.join(args.out, "batches", name)
            os.makedirs(os.path.join(d, "manifest"), exist_ok=True)
            with open(os.path.join(d, "config.yaml"), "w", encoding="utf-8") as fh:
                fh.write(f"# Atlas recount, batch {name}: {len(b)} {layout.lower()}-end sample(s) of {len(gses)} "
                         f"study(ies), {gb:.1f} Gb.\n"
                         f"# Written by atlas_build/full_run/make_batches.py; do not edit.\n"
                         f'experiment: "{name}"\nresults_dir: "results"\nsamples:\n')
                for s in sorted(b, key=lambda s: s["GSM"]):
                    fh.write(f'  {s["GSM"]}:\n    accession: "{s["GSM"]}"\n')
                fh.write(f"centrifuge: true\nvg_index_local: true\nfeaturecounts_tmp_local: true\nfastp_min_kept_percentage: {args.fastp_min_kept}\n"
                         f'keep_alignments: false\nrun_metadata: "{runs_tsv}"\n')
            for s in b:
                with open(os.path.join(d, "manifest", s["GSM"] + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(s["record"], fh, indent=1)
                sample_rows.append({"batch": name, "GSM": s["GSM"], "runs": ",".join(s["runs"]), "bases": round(s["gb"] * 1e9)})
            batch_rows.append({"batch": name, "GSE": gse, "layout": layout, "n_samples": len(b), "n_studies": len(gses), "gb": round(gb, 2),
                               "class": "big" if big else "std", "cpus": args.cpus, "hours": hours})

    for fname, rows in (("batches.tsv", batch_rows), ("batch_samples.tsv", sample_rows)):
        with open(os.path.join(args.out, fname), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    for cls in ("std", "big"):
        names = [r["batch"] for r in batch_rows if r["class"] == cls]
        with open(os.path.join(args.out, cls + ".list"), "w", encoding="utf-8") as fh:
            fh.write("".join(n + "\n" for n in names))
    # One sbatch line per batch (each has its own time limit); nothing is submitted here.
    out_abs = os.path.abspath(args.out)
    script = os.path.join(HERE, "slurm_batch.sh")
    with open(os.path.join(args.out, "submit.sh"), "w", encoding="utf-8") as fh:
        fh.write("#!/bin/bash\n# Submit the batches that are not done yet:  RUN_ROOT=<scratch dir> bash submit.sh [batch ...]\n"
                 "# A batch that failed or ran out of time continues where it stopped when submitted again.\n"
                 "set -euo pipefail\n: \"${RUN_ROOT:?set RUN_ROOT to a scratch directory for the run directories}\"\n"
                 f"OUT={out_abs}\nmkdir -p \"$OUT/logs\" \"$RUN_ROOT\"\n"
                 "want() { [[ ! -e \"$OUT/done/$1\" ]] && { [[ ${#ONLY[@]} -eq 0 ]] || [[ \" ${ONLY[*]} \" == *\" $1 \"* ]]; }; }\n"
                 "ONLY=(\"$@\")\n")
        for r in batch_rows:
            fh.write(f"want {r['batch']} && sbatch --job-name={r['batch']} --cpus-per-task={r['cpus']} "
                     f"--mem={4 * r['cpus']}G --time={int(r['hours'])}:00:00 --export=ALL,RUN_ROOT=\"$RUN_ROOT\" "
                     f"--output=\"$OUT/logs/{r['batch']}_%j.log\" {script} \"$OUT\" {r['batch']}\n")
    n_std = sum(r["class"] == "std" for r in batch_rows)
    print(f"{len(sample_rows)} samples, {sum(r['gb'] for r in batch_rows) / 1000:.2f} Tb, "
          f"{len(batch_rows)} batches ({n_std} std, {len(batch_rows) - n_std} big) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
