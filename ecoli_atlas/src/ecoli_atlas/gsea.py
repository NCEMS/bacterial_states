"""Gene-set enrichment of a gene ranking (preranked GSEA with gseapy) against GO terms.

The gene sets are the package table ``reference/go_gene_sets_ecocyc.tsv.gz`` (made by
``scripts/go_gene_sets.py`` from the EcoCyc GO annotations; its header gives the GO release), or any
table of the same columns (``term``, ``namespace``, ``name``, ``genes`` as comma-separated symbols).
Genes are matched by lower-case symbol.

Two tests. ``sides="two"`` (default; gseapy): a positive NES means the term's genes sit at the top of the
ranking, a negative NES that they sit at the bottom; a set with no enrichment still gets an NES near +1 or -1.
``sides="one"`` (``prerank_one_sided``, numpy only): the question "are the term's genes at the top of the ranking?",
as fgsea ``scoreType = "pos"`` with ``gseaParam = 0``; every NES is >= 0 and a set with no
enrichment sits near 0. gseapy is an optional dependency (``pip install gseapy`` or the ``ecoli_atlas`` conda
environment), needed for the two-sided test only.
"""
from __future__ import annotations

import gzip
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

GENE_SETS_FILE = "go_gene_sets_ecocyc.tsv.gz"
NAMESPACES = ("biological_process", "molecular_function", "cellular_component")
RESULT_COLUMNS = ["term", "database", "name", "namespace", "category", "size", "es", "nes", "pval", "fdr", "lead_genes"]
DATABASES = {"imodulon": "iModulon", "regulondb": "RegulonDB", "kegg": "KEGG", "ecocyc": "EcoCyc"}


def database_of(namespace) -> str:
    """The database a gene set comes from, by its namespace: GO for the GO namespaces, else the part before the first
    ``_`` of a module table's namespace (``imodulon_stress_responses`` -> iModulon, ``kegg_pathway`` -> KEGG)."""
    ns = str(namespace or "")
    if ns in NAMESPACES:
        return "GO"
    head = ns.split("_")[0]
    if head == "kegg" and ns in ("kegg_pathway", "kegg_module"):         # the two kinds of KEGG sets are kept apart
        return f"KEGG {ns.split('_')[1]}"
    return DATABASES.get(head, head)


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
    for col in ("namespace", "name", "category"):
        if col not in table.columns:
            table[col] = ""
    if namespaces and table["namespace"].isin(namespaces).any():      # a GO table; other tables (modules) are kept whole
        table = table[table["namespace"].isin(namespaces)]
    sets = {t: sorted({g.strip().lower() for g in s.split(",") if g.strip()}) for t, s in zip(table["term"], table["genes"])}
    return {t: g for t, g in sets.items() if g}, table.set_index("term")[["namespace", "name", "category"]], header


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
    out["category"] = out["term"].map(info["category"]).fillna("") if "category" in info else ""
    out["database"] = out["namespace"].map(database_of)
    out["pval"] = out["pval"].clip(lower=1.0 / permutations)
    return out[RESULT_COLUMNS].sort_values("nes", ascending=False).reset_index(drop=True)


def _positive_es(pos: np.ndarray, k: int, n: int) -> np.ndarray:
    """One-sided enrichment score with every gene weighted the same (gseaParam 0): ``pos`` holds the sorted 0-based
    ranks of a set's ``k`` genes among ``n`` (one row per set, last axis the members). The running sum steps +1/k at a
    member and -1/(n-k) elsewhere; its maximum is reached just after a member: max_i [i/k - (pos_i + 1 - i)/(n-k)],
    and never below 0 (the sum starts at 0)."""
    i = np.arange(1, k + 1)
    return np.maximum((i / k - (pos + 1 - i) / (n - k)).max(axis=-1), 0.0)


def _bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def prerank_one_sided(ranking: pd.Series, gene_sets: dict, terms: pd.DataFrame | None = None, permutations: int = 10000,
                      min_size: int = 5, max_size: int = 500, seed: int = 0, chunk: int = 1000) -> pd.DataFrame:
    """One-sided preranked GSEA: are a set's genes at the top of the ranking? The fgsea ``scoreType = "pos"`` test
    with ``gseaParam = 0``: ES = the largest positive deviation of the unweighted running sum (``_positive_es``);
    its null is the ES of ``permutations`` random sets of the same size drawn from the ranked genes; ``pval`` = (number of
    null ES >= ES + 1) / (permutations + 1); NES = ES / mean null ES; ``fdr`` = Benjamini-Hochberg over the tested
    sets (fgsea's ``padj``). Ties in the ranking keep their order. ``lead_genes``: the set's genes ranked at or above the
    peak of the running sum. Same columns as ``prerank``; every ES and NES >= 0."""
    rnk = ranking[~ranking.index.duplicated()].sort_values(ascending=False)
    n = len(rnk)
    rank_of = {g: i for i, g in enumerate(rnk.index)}
    rows = []
    for t, genes in gene_sets.items():
        pos = np.sort(np.fromiter((rank_of[g] for g in set(genes) if g in rank_of), dtype=np.int64))
        if min_size <= len(pos) <= max_size and len(pos) < n:
            rows.append((t, pos))
    if not rows:
        raise ValueError(f"no gene set has {min_size}-{max_size} of the {n} ranked genes")
    sizes = sorted({len(p) for _, p in rows})
    rng = np.random.default_rng(seed)
    null_sum = {k: 0.0 for k in sizes}
    null = {k: [] for k in sizes}
    done = 0
    while done < permutations:                    # random orders of the genes; the first k of each order is a random set
        m = min(chunk, permutations - done)
        order = np.argsort(rng.random((m, n)), axis=1).astype(np.int32)
        for k in sizes:
            es = _positive_es(np.sort(order[:, :k], axis=1), k, n)
            null[k].append(es)
        done += m
    null = {k: np.concatenate(v) for k, v in null.items()}
    out = []
    for t, pos in rows:
        k = len(pos)
        i = np.arange(1, k + 1)
        dev = i / k - (pos + 1 - i) / (n - k)
        es = max(float(dev.max()), 0.0)
        nk = null[k]
        mean = float(nk.mean()) or 1.0
        peak = int(np.argmax(dev)) + 1 if es > 0 else 0
        out.append({"term": t, "size": k, "es": es, "nes": es / mean, "pval": (int((nk >= es).sum()) + 1) / (len(nk) + 1),
                    "lead_genes": ",".join(rnk.index[pos[:peak]])})
    res = pd.DataFrame(out)
    res["fdr"] = _bh(res["pval"].to_numpy())
    info = terms if terms is not None else pd.DataFrame(columns=["namespace", "name"])
    res["name"] = res["term"].map(info["name"]).fillna("") if "name" in info else ""
    res["namespace"] = res["term"].map(info["namespace"]).fillna("") if "namespace" in info else ""
    res["category"] = res["term"].map(info["category"]).fillna("") if "category" in info else ""
    res["database"] = res["namespace"].map(database_of)
    return res[RESULT_COLUMNS].sort_values("nes", ascending=False).reset_index(drop=True)


def run(ranking: pd.Series, gene_sets_file=None, permutations: int = 10000, min_size: int = 5, max_size: int = 500,
        seed: int = 0, threads: int = 4, namespaces=NAMESPACES, sides: str = "two") -> tuple[pd.DataFrame, dict]:
    """``prerank`` (``sides="two"``) or ``prerank_one_sided`` (``sides="one"``) with the gene sets read from
    ``gene_sets_file`` (default: the package table). Returns (result table, summary for a JSON report)."""
    if sides not in ("one", "two"):
        raise ValueError(f"sides must be 'one' or 'two', not {sides!r}")
    sets, terms, header = read_gene_sets(gene_sets_file, namespaces)
    res = (prerank(ranking, sets, terms, permutations, min_size, max_size, seed, threads) if sides == "two"
           else prerank_one_sided(ranking, sets, terms, permutations, min_size, max_size, seed))
    summary = {"test": "one-sided (top of the ranking)" if sides == "one" else "two-sided", "gene_sets": str(gene_sets_path(gene_sets_file)),
               "gene_sets_header": header, "ranked_genes": int(len(ranking)),
               "ranked_genes_in_a_set": coverage(ranking, sets), "terms_tested": int(len(res)), "permutations": permutations,
               "min_size": min_size, "max_size": max_size, "fdr_below_0.05": {"up": int(((res.fdr < 0.05) & (res.nes > 0)).sum()),
                                                                              "down": int(((res.fdr < 0.05) & (res.nes < 0)).sum())},
               "top_up": res.loc[res.nes > 0].head(10)["name"].tolist(),
               "top_down": res.loc[res.nes < 0].sort_values("nes").head(10)["name"].tolist()}
    return res, summary
