"""Read an Ecoli_inGEST results directory: counts, antisense counts and the QC table that `embed`
needs, the latter from the per-sample QC records (``<sample>/qc/<sample>.qc.json``).

The QC values are computed as the atlas build computes them for ``sample_qc.tsv``
(``ecoli_ingest/atlas_build/prescreen/qc_metrics.py``: ``fastp_metrics``, ``vg_metrics``,
``record_columns``). This package does not import ``ecoli_ingest``, so the few formulas are repeated
here; the parity of the two is checked on the atlas samples (``work/qc_parity/``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import read_counts

RAW_COUNTS = "all_samples_raw_counts.txt"
SYMBOL_COUNTS = "all_samples_counts.tsv"
ANTISENSE_COUNTS = "all_samples_antisense_counts.txt"


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def _quantile(hist: list, q: float):
    """Quantile of a [[value, count], ...] histogram (qc_metrics.histogram_quantile)."""
    total = sum(n for _, n in hist)
    acc = 0
    for value, n in sorted(hist):
        acc += n
        if acc >= q * total:
            return value
    return None


def record_row(record: dict) -> dict:
    """The QC-table row of one sample, in the plain layout (column ``<name>``) that
    ``data.qc_covariates`` and ``data.dna_density`` read."""
    reads, comp = record.get("reads", {}), record.get("composition", {})
    before, after = reads.get("before", {}), reads.get("after", {})
    lq = reads.get("length_quantiles", {})
    n_in, n_kept = before.get("total_reads"), after.get("total_reads")
    trimmed = (reads.get("adapter") or {}).get("trimmed_reads")
    vg = (record.get("alignment") or {}).get("vg_stats", {})
    strand, dna, species = record.get("strand", {}), record.get("dna", {}), record.get("species") or {}
    tlen = (record.get("fragment") or {}).get("tlen_hist") or []
    total = sum(comp.get("by_type", {}).values())
    short = comp.get("short_features", {}).get("reads")
    runs = record.get("runs") or []
    row = {
        "layout": strand.get("layout", ""),
        "instrument_model": "/".join(sorted({r["instrument_model"] for r in runs if r.get("instrument_model")})),
        # fastp (qc_metrics.fastp_metrics)
        "reads_in": n_in, "reads_kept": n_kept,
        "pct_kept": _round(100 * (n_kept or 0) / n_in, 2) if n_in else None,
        "pct_adapter_trimmed": _round(100 * trimmed / n_in, 2) if trimmed is not None and n_in else None,
        "duplication_rate": _round(reads.get("duplication_rate")),
        # vg stats (qc_metrics.vg_metrics)
        "pct_aligned": _round(100 * vg["aligned"] / vg["total"], 2) if vg.get("total") and "aligned" in vg else None,
        # infer_experiment
        "fraction_forward": _round(strand.get("fraction_forward")), "fraction_reverse": _round(strand.get("fraction_reverse")),
        # the record (qc_metrics.record_columns)
        "length_q05_r1": lq.get("r1", {}).get("q05"), "length_q50_r1": lq.get("r1", {}).get("q50"),
        "length_q95_r1": lq.get("r1", {}).get("q95"), "length_q50_r2": lq.get("r2", {}).get("q50"),
        "cycles_r1": (reads.get("cycles") or [None])[0], "gc_content": after.get("gc_content"),
        "fragment_q25": _quantile(tlen, 0.25) if tlen else None, "fragment_q50": _quantile(tlen, 0.5) if tlen else None,
        "fragment_q75": _quantile(tlen, 0.75) if tlen else None,
        "reads_CDS": comp.get("by_type", {}).get("CDS"),
        "pct_short_features": _round(100 * short / total, 3) if total and short is not None else None,
        "substitution_rate": (record.get("alignment") or {}).get("substitution_rate"),
        "gene_body_slope": record.get("gene_body_slope"),
        "dna_fraction_est": dna.get("dna_fraction_est"), "leakage_b": dna.get("leakage_b"),
        "intergenic_share": dna.get("intergenic_share"),
        # Centrifuge (absent when the workflow ran with centrifuge: false)
        "pct_top_other_genus_nostable": species.get("pct_top_other_genus_nostable"),
        "species_verdict": species.get("verdict"),
        "pipeline_commit": (record.get("pipeline") or {}).get("commit"),
    }
    return row


def find_results_dir(path: str | Path) -> Path:
    """The inGEST results directory: `path` itself, or the one subdirectory of a run directory
    that holds the count table (e.g. ``experiment3_results/`` or ``results/``)."""
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"{p} is not a directory")
    if (p / RAW_COUNTS).exists() or (p / SYMBOL_COUNTS).exists():
        return p
    found = sorted(d for d in p.iterdir() if d.is_dir() and ((d / RAW_COUNTS).exists() or (d / SYMBOL_COUNTS).exists()))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"no {RAW_COUNTS} in {p} or its subdirectories: give the inGEST run or results directory "
                         "of a finished run")
    raise ValueError(f"{p} holds several results directories ({', '.join(d.name for d in found)}): give one of them")


def read_qc_records(results_dir: str | Path, samples: list) -> pd.DataFrame:
    """QC table (indexed by sample) of `samples` from their QC records; a sample without a record is an error."""
    d = Path(results_dir)
    rows, lacking = {}, []
    for s in samples:
        f = d / s / "qc" / f"{s}.qc.json"
        if not f.exists():
            lacking.append(s)
            continue
        with open(f, encoding="utf-8") as fh:
            rows[s] = record_row(json.load(fh))
    if lacking:
        raise ValueError(f"{len(lacking)} counted samples have no QC record (<sample>/qc/<sample>.qc.json) in {d}: "
                         f"{lacking[:5]}; the run was made with an inGEST version without QC records, re-run it")
    qc = pd.DataFrame.from_dict(rows, orient="index")
    qc.index.name = "sample"
    # text table, as data.read_table gives it
    return qc.astype(object).where(qc.notna(), "").astype(str)


def load_ingest_results(path: str | Path) -> dict:
    """Everything `embed` needs from an inGEST run: counts, antisense counts (None when the run has
    none), the QC table, and the results directory used."""
    d = find_results_dir(path)
    counts_file = d / RAW_COUNTS if (d / RAW_COUNTS).exists() else d / SYMBOL_COUNTS
    counts = read_counts(counts_file)
    antisense = read_counts(d / ANTISENSE_COUNTS) if (d / ANTISENSE_COUNTS).exists() else None
    qc = read_qc_records(d, list(counts.columns))
    return {"results_dir": d, "counts": counts, "antisense": antisense, "qc": qc}
