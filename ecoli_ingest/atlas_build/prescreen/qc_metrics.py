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
from species_metrics import (ECOLI_GENERA, ENTEROBACTERIACEAE, SYNTHETIC, genus_of_taxa,  # noqa: E402,F401
                             kreport_all_metrics, species_metrics)

# Bacteriophage genera (Centrifuge names). A sample whose largest other genus is one of these is a
# phage infection or induction experiment and is failed. Not listed on purpose: Phix174microvirus
# (the Illumina PhiX spike-in) and viruses of plants and animals (e.g. Begomovirus, which matches
# parts of some engineered plasmids).
PHAGE_GENERA = frozenset({
    "Lambdavirus", "T7virus", "T4virus", "T5virus", "T1virus", "Rb69virus", "P1virus", "P2virus", "Muvirus",
    "Mulikevirus", "N4virus", "N15virus", "Hk97virus", "Levivirus", "Allolevivirus", "Inovirus", "Epsilon15virus",
    "P22virus", "Phikzvirus", "Felixo1virus", "Sp6virus", "T3virus", "Phieco32virus", "Rtpvirus", "Tunavirus",
})


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
    "assigned", "counting_unit", "pct_assigned_not_stable",
    "pct_16S", "pct_23S", "pct_5S", "pct_tRNA_family", "pct_ssrA", "pct_rnpB", "pct_ssrS", "pct_ffs", "pct_short_features",
    "off_reference_fraction", "substitution_rate", "gene_body_slope",
    "intergenic_share", "antisense_share", "antisense_per_base", "floor_a", "leakage_b", "dna_fraction_est",
    "floor_cov_plus", "floor_cov_minus", "intergenic_cov_plus", "intergenic_cov_minus",
    "n_runs", "library_selection", "instrument_model", "pipeline_commit",
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
    for key in ("intergenic_share", "antisense_share", "antisense_per_base", "floor_a", "leakage_b", "dna_fraction_est",
                "floor_cov_plus", "floor_cov_minus", "intergenic_cov_plus", "intergenic_cov_minus"):
        out[key] = dna.get(key)
    runs = record.get("runs") or []
    out["n_runs"] = len(runs) or None
    out["library_selection"] = "/".join(sorted({r["library_selection"] for r in runs if r.get("library_selection")}))
    out["instrument_model"] = "/".join(sorted({r["instrument_model"] for r in runs if r.get("instrument_model")}))
    out["pipeline_commit"] = record.get("pipeline", {}).get("commit")
    return out


# ---------------------------------------------------------------------------- decision

DEFAULT_THRESHOLDS = {
    "min_strand_fraction": 0.8,     # dominant orientation fraction; below it the library is unstranded
    # Species test (species_metrics; the version without rRNA/tRNA reads when it is available):
    "min_pct_escherichia_shigella": 50.0,   # share of genus-resolved reads in Escherichia / Shigella ...
    "min_pct_foreign_of_all": 2.0,          # ... below it the sample is not E. coli, provided the largest other
                                            #     genus is at least this % of ALL reads given to Centrifuge
    "max_pct_foreign_of_all": 10.0,         # largest other genus above this % of all reads: mixed sample
    "min_species_reads": 100,               # fewer informative reads than this: no species decision
    "min_pct_phage_of_all": 0.5,            # largest other genus is a phage at this % of all reads: phage sample
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
    # Species. A low Escherichia/Shigella share alone is not enough: libraries of short reads leave
    # few classifiable reads, spread over many genera, and are judged by the alignment rule instead.
    # The largest other genus must also be a real part of the library (% of all classified input).
    sfx = "_nostable" if get("species_reads_nostable") is not None else ""
    n_species, share, top = get("species_reads" + sfx), get("pct_escherichia_shigella" + sfx), get("pct_top_other_genus" + sfx)
    n_all = get("centrifuge_reads")
    if n_species is None or share is None or n_species < t["min_species_reads"]:
        missing.append("species")
    else:
        foreign = (top or 0) * n_species / n_all if n_all else (top or 0)
        if metrics.get("top_other_genus" + sfx) in PHAGE_GENERA and foreign >= t["min_pct_phage_of_all"]:
            reasons.append("phage")
        elif share < t["min_pct_escherichia_shigella"] and foreign >= t["min_pct_foreign_of_all"]:
            reasons.append("not_ecoli")
        elif foreign >= t["max_pct_foreign_of_all"]:
            reasons.append("mixed_organism")
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
