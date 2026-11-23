#!/usr/bin/env python
"""Call the strand specificity of a library from RSeQC `infer_experiment.py` output.

`infer_experiment.py` reports the fraction (0-1) of reads explained by each orientation:

    single-end   "++,--"            read on the gene's strand        -> forward  (featureCounts -s 1)
                 "+-,-+"            read on the opposite strand      -> reverse  (featureCounts -s 2)
    paired-end   "1++,1--,2+-,2-+"  mate 1 on the gene's strand      -> forward
                 "1+-,1-+,2++,2--"  mate 1 on the opposite strand    -> reverse

A library is called forward or reverse when that fraction reaches `--min-fraction` (default and
lowest allowed value 0.8, `MIN_FRACTION`), otherwise unstranded. inGEST counts strand-specific
libraries only: an unstranded library is never counted. A direction stated in the config
(`--mode forward|reverse`) is checked against the measured fractions, not trusted: a library that
is unstranded, or stranded the other way (`strand_mismatch`), is not counted.

    call_strandedness.py call --infer S_infer_experiment.txt --sample S --out S_strandedness.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys

CALLS = ("forward", "reverse", "unstranded")
MIN_FRACTION = 0.8          # the least share of reads on one strand for a strand-specific library; it cannot be lowered


def check_min_fraction(min_fraction: float) -> float:
    if not min_fraction >= MIN_FRACTION:
        raise ValueError(f"the strandedness minimum fraction is at least {MIN_FRACTION} (got {min_fraction}): "
                         "unstranded libraries are never counted")
    return float(min_fraction)
FEATURECOUNTS_S = {"forward": "1", "reverse": "2"}
COLUMNS = ["sample", "layout", "fraction_forward", "fraction_reverse", "fraction_undetermined", "call",
           "call_source", "counted", "not_counted_reason"]

_FRACTION = r'[^\n]*?"{}"\s*:\s*([0-9.eE+-]+)'
_PATTERNS = {
    "PE": (r"1\+\+,1--,2\+-,2-\+", r"1\+-,1-\+,2\+\+,2--"),
    "SE": (r"\+\+,--", r"\+-,-\+"),
}


def parse_infer_experiment(text: str) -> dict:
    """Layout and orientation fractions from the text written by infer_experiment.py."""
    for layout in ("PE", "SE"):   # the paired-end labels contain the single-end ones, so try them first
        fwd = re.search(_FRACTION.format(_PATTERNS[layout][0]), text)
        rev = re.search(_FRACTION.format(_PATTERNS[layout][1]), text)
        if fwd and rev:
            und = re.search(r"failed to determine\s*:\s*([0-9.eE+-]+)", text)
            return {"layout": layout, "fraction_forward": float(fwd.group(1)),
                    "fraction_reverse": float(rev.group(1)),
                    "fraction_undetermined": float(und.group(1)) if und else float("nan")}
    raise ValueError("no orientation fractions found (is this infer_experiment.py output?)")


def call_strand(forward: float, reverse: float, min_fraction: float = MIN_FRACTION) -> str:
    check_min_fraction(min_fraction)
    if forward >= min_fraction and forward > reverse:
        return "forward"
    if reverse >= min_fraction and reverse > forward:
        return "reverse"
    return "unstranded"


def call_sample(sample: str, text: str, mode: str = "auto", min_fraction: float = MIN_FRACTION) -> dict:
    """One row of the strandedness table. The call is always measured; `mode` forward/reverse states
    the expected direction, and a library that does not show it is not counted."""
    if mode not in ("auto", "forward", "reverse"):
        raise ValueError(f"strandedness must be auto, forward or reverse, not {mode!r}")
    row = {"sample": sample, **parse_infer_experiment(text)}
    row["call"] = call_strand(row["fraction_forward"], row["fraction_reverse"], min_fraction)
    row["call_source"] = "infer_experiment" if mode == "auto" else f"infer_experiment, config {mode}"
    if row["call"] == "unstranded":
        row["counted"], row["not_counted_reason"] = "no", "unstranded"
    elif mode != "auto" and row["call"] != mode:
        row["counted"], row["not_counted_reason"] = "no", "strand_mismatch"
    else:
        row["counted"], row["not_counted_reason"] = "yes", ""
    return row


def write_table(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k]) for k in COLUMNS})


def read_table(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def counted_samples(path: str) -> list[str]:
    return [r["sample"] for r in read_table(path) if r["counted"] == "yes"]


def featurecounts_strand_flags(path: str, samples: list[str]) -> str:
    """Comma-separated -s values, one per sample in the given order."""
    call = {r["sample"]: r["call"] for r in read_table(path)}
    return ",".join(FEATURECOUNTS_S[call[s]] for s in samples)


MQC_HEADER = """# id: "strandedness"
# section_name: "Library strandedness"
# description: "Fraction of reads on the annotated strand (forward) or the opposite strand (reverse), from RSeQC infer_experiment. Only strand-specific libraries are counted."
# format: "tsv"
# plot_type: "table"
"""


def combine(tables: list[str], out_table: str, out_mqc: str | None = None,
            out_failures: str | None = None, excluded: dict | None = None) -> list[dict]:
    """Merge per-sample tables into the run table, the MultiQC table and the list of refused samples.
    `excluded` maps samples refused for another reason (the species gate) to that reason; they are
    not counted either."""
    rows = [r for t in tables for r in read_table(t)]
    rows.sort(key=lambda r: r["sample"])
    for r in rows:
        r.setdefault("not_counted_reason", "" if r["counted"] == "yes" else "unstranded")
        if excluded and r["sample"] in excluded:
            r["counted"] = "no"
            r["not_counted_reason"] = ",".join(x for x in (r["not_counted_reason"], excluded[r["sample"]]) if x)
    with open(out_table, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if out_mqc:
        with open(out_mqc, "w", encoding="utf-8", newline="") as fh:
            fh.write(MQC_HEADER)
            w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    if out_failures:
        failures = {r["sample"]: {"reason": r["not_counted_reason"], "fraction_forward": float(r["fraction_forward"]),
                                  "fraction_reverse": float(r["fraction_reverse"])}
                    for r in rows if r["counted"] != "yes"}
        with open(out_failures, "w", encoding="utf-8") as fh:
            json.dump(failures, fh, indent=2)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("call", help="call one sample")
    c.add_argument("--infer", required=True, help="infer_experiment.py output")
    c.add_argument("--sample", required=True)
    c.add_argument("--mode", default="auto", choices=("auto", "forward", "reverse"))
    c.add_argument("--min-fraction", type=float, default=MIN_FRACTION, help=f"at least {MIN_FRACTION}")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.infer, encoding="utf-8") as fh:
        text = fh.read()
    try:
        row = call_sample(args.sample, text, args.mode, args.min_fraction)
    except ValueError as err:
        raise SystemExit(f"{args.infer}: {err}")
    write_table([row], args.out)
    print(f"{args.sample}: {row['call']} (forward {row['fraction_forward']:.3f}, "
          f"reverse {row['fraction_reverse']:.3f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
