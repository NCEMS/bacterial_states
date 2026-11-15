"""Per-sample QC metrics, read from the files inGEST (or the prescreen) writes.

The prescreen (`prescreen.py`, a read subsample) and the full-run collection
(`qc_table.py add-full`, a finished inGEST results directory) both use these functions, so the
`pre_*` and `full_*` columns of the sample QC table are computed the same way.
"""
from __future__ import annotations

import json
import os
import re
import sys

_SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                                         "ecoli_ingest", "workflow", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import call_strandedness  # noqa: E402

SCRIPTS_DIR = _SCRIPTS
FEATURE_GROUPS = ("CDS", "rRNA", "tRNA", "ncRNA", "Rfam", "pseudo")

# Metric columns of the QC table, in order (each appears as pre_<name> and full_<name>).
METRICS = (
    ["reads_in", "reads_kept", "pct_kept", "mean_length_r1", "mean_length_r2", "pct_adapter_trimmed",
     "insert_size_peak", "duplication_rate", "q30_rate",
     "pct_aligned", "pct_perfect", "pct_properly_paired", "mapq_mean",
     "layout", "fraction_forward", "fraction_reverse", "fraction_undetermined", "strand_call"]
    + [f"pct_{g}" for g in FEATURE_GROUPS]
    + ["pct_ecoli", "other_taxa"]
)


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def fastp_metrics(path: str) -> dict:
    """Read counts are reads (both mates counted) as fastp reports them."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    before, after = d["summary"]["before_filtering"], d["summary"]["after_filtering"]
    n_in = before.get("total_reads", 0)
    trimmed = d.get("adapter_cutting", {}).get("adapter_trimmed_reads")
    return {
        "reads_in": n_in,
        "reads_kept": after.get("total_reads", 0),
        "pct_kept": _round(100 * after.get("total_reads", 0) / n_in, 2) if n_in else None,
        "mean_length_r1": after.get("read1_mean_length"),
        "mean_length_r2": after.get("read2_mean_length") if "read2_after_filtering" in d else None,
        "pct_adapter_trimmed": _round(100 * trimmed / n_in, 2) if trimmed is not None and n_in else None,
        "insert_size_peak": d.get("insert_size", {}).get("peak"),
        "duplication_rate": _round(d.get("duplication", {}).get("rate")),
        "q30_rate": _round(after.get("q30_rate")),
    }


def vg_stats_metrics(path: str) -> dict:
    """From `vg stats -a` on the GAM."""
    vals = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"(Total alignments|Total aligned|Total perfect|Total paired|Total properly paired):\s*(\d+)", line)
            if m:
                vals[m.group(1)] = int(m.group(2))
            m = re.match(r"Mapping quality: mean ([0-9.eE+-]+)", line)
            if m:
                vals["mapq"] = float(m.group(1))
    total = vals.get("Total alignments", 0)
    pct = lambda key, denom: _round(100 * vals[key] / denom, 2) if key in vals and denom else None  # noqa: E731
    return {
        "pct_aligned": pct("Total aligned", total),
        "pct_perfect": pct("Total perfect", total),
        "pct_properly_paired": pct("Total properly paired", vals.get("Total paired", 0)),
        "mapq_mean": _round(vals.get("mapq"), 2),
    }


def strand_metrics(infer_experiment_path: str, min_fraction: float = 0.8) -> dict:
    with open(infer_experiment_path, encoding="utf-8") as fh:
        row = call_strandedness.call_sample("s", fh.read(), "auto", min_fraction)
    return {"layout": row["layout"], "fraction_forward": _round(row["fraction_forward"]),
            "fraction_reverse": _round(row["fraction_reverse"]),
            "fraction_undetermined": _round(row["fraction_undetermined"]), "strand_call": row["call"]}


def feature_overlap_metrics(path: str) -> dict:
    """From the `*_feature_overlap_mqc.tsv` written by gene_type.sh on the counting annotation BED."""
    out = {f"pct_{g}": 0.0 for g in FEATURE_GROUPS}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            name, _, value = line.rstrip("\n").partition("\t")
            if f"pct_{name}" in out:
                out[f"pct_{name}"] = _round(value, 2)
    return out


def kreport_metrics(path: str, organism: str = "Escherichia coli", n_other: int = 3) -> dict:
    """% of classified reads assigned to `organism` (the first report line naming it, which is what
    the workflow's QC gate reads), and the largest other species."""
    pct = 0.0
    found = False
    species = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            name = cols[5].strip()
            if organism in line and not found:
                try:
                    pct = float(cols[0])
                except ValueError:
                    pct = 0.0
                found = True
            if cols[3] == "S" and organism not in name:
                try:
                    species.append((float(cols[0]), name))
                except ValueError:
                    pass
    species.sort(reverse=True)
    return {"pct_ecoli": _round(pct, 2),
            "other_taxa": "; ".join(f"{name} {p:.1f}" for p, name in species[:n_other] if p > 0)}


def collect(files: dict, min_fraction: float = 0.8) -> dict:
    """Metrics from whichever of these files exist: fastp, vg_stats, infer_experiment,
    feature_overlap, kreport. Missing files leave their metrics unset."""
    readers = (("fastp", fastp_metrics), ("vg_stats", vg_stats_metrics),
               ("infer_experiment", lambda p: strand_metrics(p, min_fraction)),
               ("feature_overlap", feature_overlap_metrics), ("kreport", kreport_metrics))
    out = {}
    for key, reader in readers:
        path = files.get(key)
        if path and os.path.exists(path):
            out.update(reader(path))
    return out


# ---------------------------------------------------------------------------- decision

DEFAULT_THRESHOLDS = {
    "min_strand_fraction": 0.8,     # dominant orientation fraction; below it the library is unstranded
    "min_pct_ecoli": 30.0,          # the workflow's centrifuge_min_percentage
    "min_pct_kept": 75.0,           # the workflow's fastp_min_kept_percentage
    "min_mean_length": 30.0,        # mean read length after trimming (mate 1)
    "min_pct_aligned": 50.0,
}


def decide(metrics: dict, thresholds: dict | None = None) -> tuple[str, list[str]]:
    """('pass' | 'fail' | 'incomplete', reasons). A missing metric makes the decision incomplete."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    get = lambda k: None if metrics.get(k) in (None, "") else float(metrics[k])  # noqa: E731
    reasons, missing = [], []
    fwd, rev = get("fraction_forward"), get("fraction_reverse")
    if fwd is None or rev is None:
        missing.append("strandedness")
    elif max(fwd, rev) < t["min_strand_fraction"]:
        reasons.append("unstranded")
    for key, limit, label in (("pct_ecoli", "min_pct_ecoli", "contamination"), ("pct_kept", "min_pct_kept", "trimming"),
                              ("mean_length_r1", "min_mean_length", "short_reads"),
                              ("pct_aligned", "min_pct_aligned", "low_alignment")):
        value = get(key)
        if value is None:
            missing.append(label)
        elif value < t[limit]:
            reasons.append(label)
    if reasons:
        return "fail", reasons
    if missing:
        return "incomplete", [f"no_{m}" for m in missing]
    return "pass", []
