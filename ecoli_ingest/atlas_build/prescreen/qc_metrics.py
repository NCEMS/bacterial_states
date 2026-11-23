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
    + ["pct_ecoli", "other_taxa", "pct_unclassified", "pct_enterobacteriaceae", "pct_synthetic", "top_non_enteric",
       "species_reads", "pct_escherichia_shigella", "top_other_genus", "pct_top_other_genus",
       "species_reads_nostable", "pct_escherichia_shigella_nostable", "top_other_genus_nostable",
       "pct_top_other_genus_nostable", "centrifuge_reads"]
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


# Species statistics live with the workflow scripts so that full runs compute them the same way.
from species_metrics import (ECOLI_GENERA, ENTEROBACTERIACEAE, PHAGE_GENERA, SPECIES_THRESHOLDS,  # noqa: E402,F401
                             SYNTHETIC, genus_of_taxa, kreport_all_metrics, species_metrics, species_verdict)

def collect(files: dict, min_fraction: float = 0.8) -> dict:
    """Metrics from whichever of these files exist: fastp, vg_stats, infer_experiment,
    feature_overlap, kreport, kreport_all. Missing files leave their metrics unset."""
    readers = (("fastp", fastp_metrics), ("vg_stats", vg_stats_metrics),
               ("infer_experiment", lambda p: strand_metrics(p, min_fraction)),
               ("feature_overlap", feature_overlap_metrics), ("kreport", kreport_metrics),
               ("kreport_all", kreport_all_metrics))
    out = {}
    for key, reader in readers:
        path = files.get(key)
        if path and os.path.exists(path):
            out.update(reader(path))
    cls, rep = files.get("classifications"), files.get("kreport_all")
    if cls and rep and os.path.exists(cls) and os.path.exists(rep):
        out.update(species_metrics(cls, rep))
        stable = files.get("stable_ids")
        if stable and os.path.exists(stable):
            with open(stable, encoding="utf-8") as fh:
                out.update(species_metrics(cls, rep, set(fh.read().split()), "_nostable"))
    return out


# ---------------------------------------------------------------------------- full-run record

# Columns of the QC table that only a full inGEST run has (full_<name>), derived from the sample's
# QC record (<sample>.qc.json, workflow/scripts/sample_qc.py).
RECORD_COLUMNS = [
    "length_q05_r1", "length_q50_r1", "length_q95_r1", "length_q50_r2", "cycles_r1", "gc_content",
    "fragment_q25", "fragment_q50", "fragment_q75",
    "assigned", "counting_unit", "pct_assigned_not_stable", "reads_CDS",
    "pct_16S", "pct_23S", "pct_5S", "pct_tRNA_family", "pct_ssrA", "pct_rnpB", "pct_ssrS", "pct_ffs", "pct_short_features",
    "off_reference_fraction", "substitution_rate", "gene_body_slope", "gc_slope", "gc_curv",
    "intergenic_share", "antisense_share", "antisense_per_base", "dna_fraction_est", "dna_fraction_limit",
    "minstrand_cov", "dna_fraction_fit", "floor_a", "leakage_b",
    "floor_cov_plus", "floor_cov_minus", "intergenic_cov_plus", "intergenic_cov_minus",
    "n_runs", "library_selection", "instrument_model", "pipeline_commit", "species_verdict",
]


def histogram_quantile(hist: list, q: float):
    """Quantile of a [[value, count], ...] histogram."""
    total = sum(n for _, n in hist)
    acc = 0
    for value, n in sorted(hist):
        acc += n
        if acc >= q * total:
            return value
    return None


def record_columns(record: dict) -> dict:
    """The RECORD_COLUMNS values of one QC record. Shares are % of all read-feature assignments
    (a read on two features counts twice), except pct_assigned_not_stable, which is % of assigned reads."""
    reads, comp = record.get("reads", {}), record.get("composition", {})
    lq = reads.get("length_quantiles", {})
    out = {
        "length_q05_r1": lq.get("r1", {}).get("q05"), "length_q50_r1": lq.get("r1", {}).get("q50"),
        "length_q95_r1": lq.get("r1", {}).get("q95"), "length_q50_r2": lq.get("r2", {}).get("q50"),
        "cycles_r1": (reads.get("cycles") or [None])[0], "gc_content": reads.get("after", {}).get("gc_content"),
    }
    tlen = [x for x in record.get("fragment", {}).get("tlen_hist", [])]
    for name, q in (("fragment_q25", 0.25), ("fragment_q50", 0.5), ("fragment_q75", 0.75)):
        out[name] = histogram_quantile(tlen, q) if tlen else None
    ladder = record.get("ladder", {})
    assigned = ladder.get("assigned")
    out["assigned"], out["counting_unit"] = assigned, ladder.get("counting_unit")
    out["pct_assigned_not_stable"] = (_round(100 * ladder["assigned_not_stable_rna"] / assigned, 2)
                                      if assigned and ladder.get("assigned_not_stable_rna") is not None else None)
    out["reads_CDS"] = comp.get("by_type", {}).get("CDS")      # assignments to CDS features: the mRNA depth of the sample
    total = sum(comp.get("by_type", {}).values())
    family = comp.get("by_family", {})
    for key, name in (("16S", "pct_16S"), ("23S", "pct_23S"), ("5S", "pct_5S"), ("tRNA", "pct_tRNA_family"),
                      ("ssrA", "pct_ssrA"), ("rnpB", "pct_rnpB"), ("ssrS", "pct_ssrS"), ("ffs", "pct_ffs")):
        out[name] = _round(100 * family[key] / total, 3) if total and key in family else None
    short = comp.get("short_features", {}).get("reads")
    out["pct_short_features"] = _round(100 * short / total, 3) if total and short is not None else None
    align, dna = record.get("alignment", {}), record.get("dna", {})
    out.update(off_reference_fraction=align.get("off_reference_fraction"), substitution_rate=align.get("substitution_rate"),
               gene_body_slope=record.get("gene_body_slope"))
    gc = record.get("gc_bias") or {}                      # GC bias inside genes (workflow gc_bias.py)
    out.update(gc_slope=gc.get("gc_slope"), gc_curv=gc.get("gc_curv"))
    for key in ("intergenic_share", "antisense_share", "antisense_per_base", "dna_fraction_est",
                "dna_fraction_limit", "minstrand_cov", "dna_fraction_fit", "floor_a", "leakage_b",
                "floor_cov_plus", "floor_cov_minus", "intergenic_cov_plus", "intergenic_cov_minus"):
        out[key] = dna.get(key)
    runs = record.get("runs") or []
    out["n_runs"] = len(runs) or None
    out["library_selection"] = "/".join(sorted({r["library_selection"] for r in runs if r.get("library_selection")}))
    out["instrument_model"] = "/".join(sorted({r["instrument_model"] for r in runs if r.get("instrument_model")}))
    out["pipeline_commit"] = record.get("pipeline", {}).get("commit")
    out["species_verdict"] = (record.get("species") or {}).get("verdict")
    return out


# ---------------------------------------------------------------------------- decision

DEFAULT_THRESHOLDS = {
    "min_strand_fraction": 0.8,     # dominant orientation fraction; below it the library is unstranded
    # species test: the workflow's gate (species_metrics.SPECIES_THRESHOLDS)
    **{k: v for k, v in SPECIES_THRESHOLDS.items() if not k.startswith("early_")},
    "min_pct_kept": 75.0,           # the workflow's fastp_min_kept_percentage
    "min_mean_length": 30.0,        # mean read length after trimming (mate 1)
    "min_pct_aligned": 50.0,
}


def decide(metrics: dict, thresholds: dict | None = None) -> tuple[str, list[str]]:
    """('pass' | 'fail' | 'incomplete', reasons). A missing metric makes the decision incomplete."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not t["min_strand_fraction"] >= DEFAULT_THRESHOLDS["min_strand_fraction"]:
        raise ValueError(f"min_strand_fraction is at least {DEFAULT_THRESHOLDS['min_strand_fraction']}: "
                         "unstranded libraries are never accepted")
    get = lambda k: None if metrics.get(k) in (None, "") else float(metrics[k])  # noqa: E731
    reasons, missing = [], []
    fwd, rev = get("fraction_forward"), get("fraction_reverse")
    if fwd is None or rev is None:
        missing.append("strandedness")
    elif max(fwd, rev) < t["min_strand_fraction"]:
        reasons.append("unstranded")
    verdict = species_verdict(metrics, t)        # the same rule as the workflow's species gate
    if verdict == "no_species":
        missing.append("species")
    elif verdict != "ok":
        reasons.append(verdict)
    for key, limit, label in (("pct_kept", "min_pct_kept", "trimming"),
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
