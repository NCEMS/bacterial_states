"""`ecoli_atlas prepare`: turn Ecoli_inGEST results into the input tables of `embed`.

Output
  counts.tsv            features x samples, first column ``gene`` (the inGEST feature id), sample columns named by sample id
  antisense_counts.tsv  the same for the antisense counts (when the run has them)
  qc.tsv                one row per sample: the QC values `embed` needs, from the samples' QC records
  samples.tsv           one row per sample: sample_id, source_column, plus any columns of a tab-separated metadata table

`embed --ingest-results` reads the same from a run directly; `prepare` is for looking at the tables,
or for joining several runs before one `embed --counts ... --qc ... --antisense ...`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import read_table
from .io import read_counts, sample_id_from_column, write_tsv
from .qc_record import ANTISENSE_COUNTS, RAW_COUNTS, SYMBOL_COUNTS, find_results_dir, read_qc_records


def find_counts_file(results_dir: Path) -> Path:
    for name in (RAW_COUNTS, SYMBOL_COUNTS):
        p = results_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no {RAW_COUNTS} or {SYMBOL_COUNTS} in {results_dir}")


def prepare(results_dir: str | Path | None, counts_file: str | Path | None, metadata: str | Path | None,
            out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """With `results_dir` (an inGEST run or results directory) also writes antisense_counts.tsv and qc.tsv."""
    qc = antisense = d = None
    if counts_file is None:
        if results_dir is None:
            raise ValueError("give --ingest-results or --counts")
        d = find_results_dir(results_dir)
        counts_file = find_counts_file(d)
        if (d / ANTISENSE_COUNTS).exists():
            antisense = read_counts(d / ANTISENSE_COUNTS)
    raw_cols = pd.read_csv(counts_file, sep="\t", comment="#", nrows=0).columns
    counts = read_counts(counts_file)
    if d is not None:
        qc = read_qc_records(d, list(counts.columns))
    source = {sample_id_from_column(c): c for c in raw_cols if sample_id_from_column(c) in counts.columns}
    samples = pd.DataFrame({"sample_id": list(counts.columns),
                            "source_column": [source.get(s, s) for s in counts.columns]}).set_index("sample_id")
    if metadata is not None:
        meta = read_table(metadata)
        missing = [s for s in samples.index if s not in meta.index]
        if missing:
            raise ValueError(f"metadata lacks rows for samples: {missing[:5]}")
        samples = samples.join(meta.loc[samples.index], how="left")
    out = Path(out_dir)
    write_tsv(counts, out / "counts.tsv")
    if antisense is not None:
        write_tsv(antisense, out / "antisense_counts.tsv")
    if qc is not None:
        write_tsv(qc, out / "qc.tsv")
    write_tsv(samples, out / "samples.tsv")
    return counts, samples
