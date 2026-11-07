"""Reading count tables (inGEST outputs or plain TSVs), metadata, and the NCBI feature table."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# featureCounts / inGEST annotation columns that are not samples
ANNOTATION_COLUMNS = ("Chr", "Start", "End", "Strand", "Length", "Gene_IDs_Field", "Gene_Summary")
GENE_ID_COLUMNS = ("gene", "Gene_Symbol", "Geneid", "gene_id", "GeneID", "symbol")
_BAM_RE = re.compile(r"(?:^|/)([^/]+?)(?:_sort)?\.bam$")


def sample_id_from_column(col: str) -> str:
    """`experiment3_results/KO_1/vg/KO_1_sort.bam` -> `KO_1`; anything else is returned unchanged."""
    m = _BAM_RE.search(str(col))
    return m.group(1) if m else str(col)


def read_counts(path: str | Path, strip_sample_names: bool = True) -> pd.DataFrame:
    """Read a genes x samples count table.

    Accepts featureCounts output (``# Program`` comment line; ``Geneid Chr Start End Strand Length``
    then one column per BAM), the inGEST ``*_gene_symbols.tsv`` / ``*_counts_extended.tsv`` tables,
    or a plain TSV whose first column is the gene identifier. Returns a DataFrame indexed by gene
    (index name ``gene``) with one integer/float column per sample. Missing values become 0.
    """
    df = pd.read_csv(path, sep="\t", comment="#", dtype={0: str})
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected a gene column plus at least one sample column")
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    df.index = df.index.astype(str).str.strip()
    df.index.name = "gene"
    df = df.drop(columns=[c for c in ANNOTATION_COLUMNS if c in df.columns])
    if strip_sample_names:
        df.columns = [sample_id_from_column(c) for c in df.columns]
    if df.columns.duplicated().any():
        dups = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(f"{path}: duplicate sample names after stripping BAM paths: {dups[:5]}")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Sample metadata with the sample identifier in the first column."""
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.astype(str).str.strip()
    meta.index.name = "sample_id"
    return meta


def read_feature_table(path: str | Path) -> pd.DataFrame:
    """NCBI assembly feature table; the leading ``# `` is stripped from the header."""
    feat = pd.read_csv(path, sep="\t", low_memory=False)
    feat.columns = feat.columns.str.replace("# ", "", regex=False).str.strip()
    for col in ("class", "symbol", "locus_tag"):
        if col not in feat.columns:
            raise ValueError(f"{path}: feature table lacks column {col!r}")
    return feat


def write_tsv(df: pd.DataFrame, path: str | Path, index: bool = True, float_format: str | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index, float_format=float_format)
