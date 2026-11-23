"""Gene-set enrichment of a gene ranking (preranked GSEA with gseapy) against GO terms.

The gene sets are the package table ``reference/go_gene_sets_ecocyc.tsv.gz`` (made by
``scripts/go_gene_sets.py`` from the EcoCyc GO annotations; its header gives the GO release), or any
table of the same columns (``term``, ``namespace``, ``name``, ``genes`` as comma-separated symbols).
Genes are matched by lower-case symbol.

The test is two-sided: a positive NES means the term's genes sit at the top of the ranking, a
negative NES that they sit at the bottom. gseapy is an optional dependency (``pip install gseapy``
or the ``ecoli_atlas`` conda environment).
"""
from __future__ import annotations

import gzip
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

GENE_SETS_FILE = "go_gene_sets_ecocyc.tsv.gz"
NAMESPACES = ("biological_process", "molecular_function", "cellular_component")
RESULT_COLUMNS = ["term", "name", "namespace", "size", "es", "nes", "pval", "fdr", "lead_genes"]


def gene_sets_path(path=None) -> Path:
    return Path(path) if path else Path(str(resources.files("ecoli_atlas") / "reference" / GENE_SETS_FILE))


def read_gene_sets(path=None, namespaces=NAMESPACES) -> tuple[dict, pd.DataFrame, list]:
    """(term -> list of lower-case symbols, term table indexed by term, the file's `#` header lines)."""
    p = gene_sets_path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        header = [line.rstrip("\n")[2:] for line in fh if line.startswith("#")]
    table = pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = {"term", "genes"} - set(table.columns)
    if missing:
        raise ValueError(f"{p}: gene-set table lacks the column(s) {sorted(missing)}")
    for col in ("namespace", "name"):
        if col not in table.columns:
            table[col] = ""
    if namespaces and (table["namespace"] != "").any():
        table = table[table["namespace"].isin(namespaces)]
    sets = {t: sorted({g.strip().lower() for g in s.split(",") if g.strip()}) for t, s in zip(table["term"], table["genes"])}
    return {t: g for t, g in sets.items() if g}, table.set_index("term")[["namespace", "name"]], header


def ranking_by_symbol(scores, symbols) -> pd.Series:
    """One score per lower-case symbol, largest first. Features without a symbol are dropped; of a
    symbol that several features share, the one with the largest absolute score is kept."""
    s = pd.Series(np.asarray(scores, dtype=float), index=pd.Index([str(x).strip().lower() for x in symbols]))
    s = s[(s.index != "") & np.isfinite(s.to_numpy())]
    s = s.iloc[np.argsort(-np.abs(s.to_numpy()), kind="stable")]
    return s[~s.index.duplicated()].sort_values(ascending=False)


def coverage(ranking: pd.Series, gene_sets: dict) -> float:
    """Share of the ranked genes that are in at least one gene set."""
    members = set().union(*gene_sets.values()) if gene_sets else set()
    return float(pd.Index(ranking.index).isin(members).mean()) if len(ranking) else 0.0


def prerank(ranking: pd.Series, gene_sets: dict, terms: pd.DataFrame | None = None, permutations: int = 10000,
            min_size: int = 5, max_size: int = 500, seed: int = 0, threads: int = 4) -> pd.DataFrame:
    """Preranked GSEA (gseapy, weight 0: every gene counts the same, as fgsea's gseaParam 0), two-sided.
    Returns one row per tested term (RESULT_COLUMNS), sorted by NES, largest first; ``pval`` is the
    permutation p-value (at least 1 / permutations), ``fdr`` gseapy's FDR q-value."""
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError("gene-set enrichment needs gseapy: `pip install gseapy` or `mamba install -c bioconda gseapy` "
                          "(it is in ecoli_atlas/environment.yaml)") from e
    rnk = ranking[~ranking.index.duplicated()].sort_values(ascending=False)
    present = set(rnk.index)
    usable = {t: g for t, g in gene_sets.items() if min_size <= len(present.intersection(g)) <= max_size}
    if not usable:
        raise ValueError(f"no gene set has {min_size}-{max_size} of the {len(rnk)} ranked genes")
    import logging
    logging.getLogger("gseapy").setLevel(logging.ERROR)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                      # ties in the ranking: gseapy warns, the order is kept
        res = gp.prerank(rnk=rnk, gene_sets=usable, weight=0, min_size=min_size, max_size=max_size,
                         permutation_num=permutations, seed=seed, threads=threads, outdir=None, no_plot=True,
                         verbose=False).res2d
    out = pd.DataFrame({
        "term": res["Term"].astype(str),
        "size": res["Tag %"].astype(str).str.split("/").str[1].astype(int),
        "es": res["ES"].astype(float), "nes": res["NES"].astype(float),
        "pval": res["NOM p-val"].astype(float), "fdr": res["FDR q-val"].astype(float),
        "lead_genes": res["Lead_genes"].astype(str).str.replace(";", ",", regex=False),
    })
    info = terms if terms is not None else pd.DataFrame(columns=["namespace", "name"])
    out["name"] = out["term"].map(info["name"]).fillna("") if "name" in info else ""
    out["namespace"] = out["term"].map(info["namespace"]).fillna("") if "namespace" in info else ""
    out["pval"] = out["pval"].clip(lower=1.0 / permutations)
    return out[RESULT_COLUMNS].sort_values("nes", ascending=False).reset_index(drop=True)


def run(ranking: pd.Series, gene_sets_file=None, permutations: int = 10000, min_size: int = 5, max_size: int = 500,
        seed: int = 0, threads: int = 4, namespaces=NAMESPACES) -> tuple[pd.DataFrame, dict]:
    """``prerank`` with the gene sets read from ``gene_sets_file`` (default: the package table).
    Returns (result table, summary for a JSON report)."""
    sets, terms, header = read_gene_sets(gene_sets_file, namespaces)
    res = prerank(ranking, sets, terms, permutations, min_size, max_size, seed, threads)
    summary = {"gene_sets": str(gene_sets_path(gene_sets_file)), "gene_sets_header": header, "ranked_genes": int(len(ranking)),
               "ranked_genes_in_a_set": coverage(ranking, sets), "terms_tested": int(len(res)), "permutations": permutations,
               "min_size": min_size, "max_size": max_size, "fdr_below_0.05": {"up": int(((res.fdr < 0.05) & (res.nes > 0)).sum()),
                                                                              "down": int(((res.fdr < 0.05) & (res.nes < 0)).sum())},
               "top_up": res.loc[res.nes > 0].head(10)["name"].tolist(),
               "top_down": res.loc[res.nes < 0].sort_values("nes").head(10)["name"].tolist()}
    return res, summary
