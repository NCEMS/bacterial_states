"""`ecoli_atlas prepare`: turn Ecoli_inGEST results into the atlas input format.

Output
  counts.tsv   features x samples, first column ``gene`` (the inGEST feature id), sample columns named by sample id
  samples.tsv  one row per sample: sample_id, source_column, plus any columns of a tab-separated metadata table
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import read_table
from .io import read_counts, sample_id_from_column, write_tsv

RAW_COUNTS = "all_samples_raw_counts.txt"
SYMBOL_COUNTS = "all_samples_counts.tsv"


def find_counts_file(results_dir: Path) -> Path:
    for name in (RAW_COUNTS, SYMBOL_COUNTS):
        p = results_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no {RAW_COUNTS} or {SYMBOL_COUNTS} in {results_dir}")


def prepare(results_dir: str | Path | None, counts_file: str | Path | None, metadata: str | Path | None,
            out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if counts_file is None:
        if results_dir is None:
            raise ValueError("give --ingest-results or --counts")
        counts_file = find_counts_file(Path(results_dir))
    raw_cols = pd.read_csv(counts_file, sep="\t", comment="#", nrows=0).columns
    counts = read_counts(counts_file)
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
    write_tsv(samples, out / "samples.tsv")
    return counts, samples
