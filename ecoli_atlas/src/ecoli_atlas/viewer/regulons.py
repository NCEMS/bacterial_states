"""Regulon expression of the atlas samples for `ecoli_atlas view`.

The sets are the RegulonDB rows of a module table (`ecoli_atlas download modules`): for each regulator
the genes it activates (``regulondb:<regulator>+``) and those it represses (``-``), every set, whatever
its size. The value of a set in a sample is the mean over the set's genes of their standardised log
expression:

  expression   log2(count / size factor + 1), size factors by the bundle's reference profile of mRNA
               counts (the counts as they are, not GC-cleaned), genes called missing left out
  standardised per gene: minus its mean, divided by its SD, both over the atlas samples that have the gene
  set value    mean over the set's genes the sample has, minus the sample's median over all its genes;
               no value when the sample has none of the set's genes

So a value says how far the set's genes lie above or below their atlas means, in SDs of single genes,
compared with the sample's other genes. The median is taken off because a shallow library has more genes
without reads and so sits a little low in every set (about 0.2 SD below a million mRNA reads). It
is the expression of the targets, not the activity of the regulator. The page stores one byte per sample
and set (0 = no value, 1 to 255 = -limit to +limit, the set's own limit).
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import FeatureSet
from ..modules import read_module_table
from ..preprocess import align_counts, log_expression, size_factors

PREFIX = "regulondb:"
MIN_LIMIT = 0.5          # SDs: the smallest half-range of a set's colour scale
LIMIT_QUANTILE = 0.995   # of |value| over the atlas samples


def feature_symbols(ids: list) -> tuple[list, list]:
    """(symbol, locus tag) of the model features from their ids (``CDS_b0002_thrA`` -> thrA, b0002;
    ``ncRNA_<Rfam id>`` -> the id's tail, no locus tag)."""
    sym, locus = [], []
    for f in ids:
        parts = str(f).split("_", 2)
        if len(parts) == 3 and parts[1][:1] == "b" and parts[1][1:].isdigit():
            locus.append(parts[1]), sym.append(parts[2])
        else:
            locus.append(""), sym.append(str(f).split("_", 1)[-1])
    return sym, locus


def regulon_sets(modules: str | Path | pd.DataFrame, ids: list) -> list[dict]:
    """The RegulonDB sets of a module table with their genes matched to the model features (lower-case
    symbol or locus tag): ``{term, name, category, genes (as listed), members (feature indices), absent}``.
    Sets without any model feature are kept (the page says so); no size limits."""
    table = read_module_table(modules) if not isinstance(modules, pd.DataFrame) else modules
    table = table[table["term"].astype(str).str.startswith(PREFIX)]
    if table.empty:
        raise ValueError(f"the module table has no RegulonDB sets (terms starting with {PREFIX!r}); "
                         "make it with `ecoli_atlas download modules`")
    sym, locus = feature_symbols(ids)
    index = {}
    for i, (s, l) in enumerate(zip(sym, locus)):
        for key in (s.lower(), l.lower()):
            if key and key not in index:
                index[key] = i
    out = []
    for _, row in table.iterrows():
        genes = [g.strip() for g in str(row["genes"]).split(",") if g.strip()]
        members = sorted({index[g.lower()] for g in genes if g.lower() in index})
        out.append({"term": str(row["term"]), "name": str(row.get("name", "") or row["term"]),
                    "category": str(row.get("category", "") or ""), "genes": genes, "members": members,
                    "absent": [g for g in genes if g.lower() not in index]})
    return out


def missing_matrix(path: str | Path | None, ids: list, features: list) -> np.ndarray:
    """samples x features bool from a ``missing_genes.tsv`` (sample, feature); all False without the file."""
    M = np.zeros((len(ids), len(features)), dtype=bool)
    if path is None or not Path(path).exists():
        return M
    calls = pd.read_csv(path, sep="\t", dtype=str)
    si = {s: i for i, s in enumerate(ids)}
    fi = {f: j for j, f in enumerate(features)}
    r, c = calls["sample"].map(si), calls["feature"].map(fi)
    ok = r.notna() & c.notna()
    M[r[ok].astype(int).to_numpy(), c[ok].astype(int).to_numpy()] = True
    return M


def expression(counts: pd.DataFrame, fs: FeatureSet, ref_log: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """samples x features log2 normalised counts, NaN where the gene is called missing."""
    Y, _, _ = align_counts(counts, fs)
    E = log_expression(Y, size_factors(Y, ref_log, missing))
    E[missing] = np.nan
    return E


def gene_stats(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per gene mean and SD over the samples that have it; SD NaN for a gene that does not vary."""
    with np.errstate(invalid="ignore"), _quiet():
        mu, sd = np.nanmean(E, axis=0), np.nanstd(E, axis=0)
    return mu, np.where(sd > 1e-9, sd, np.nan)


def set_values(E: np.ndarray, mu: np.ndarray, sd: np.ndarray, sets: list[dict]) -> np.ndarray:
    """samples x sets: mean standardised expression of each set's genes minus the sample's median over all
    genes (NaN: none of the set's genes in the sample)."""
    V = np.full((E.shape[0], len(sets)), np.nan)
    with np.errstate(invalid="ignore"), _quiet():
        Z = (E - mu) / sd
        centre = np.nanmedian(Z, axis=1)
        for j, s in enumerate(sets):
            m = [g for g in s["members"] if np.isfinite(sd[g])]
            if m:
                V[:, j] = np.nanmean(Z[:, m], axis=1) - centre
    return V


class _quiet:
    """numpy warns about all-NaN slices; NaN is the wanted result there."""
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        return self._c.__exit__(*a)


def limits(V: np.ndarray) -> np.ndarray:
    """Half-range of each set's colour scale: a high quantile of |value| over the samples."""
    out = np.full(V.shape[1], MIN_LIMIT)
    for j in range(V.shape[1]):
        v = np.abs(V[:, j][np.isfinite(V[:, j])])
        if len(v):
            out[j] = max(MIN_LIMIT, float(np.quantile(v, LIMIT_QUANTILE)))
    return np.round(out, 3)


def encode(v: np.ndarray, limit: float) -> str:
    """One byte per sample, base64: 0 = no value, 1..255 = -limit..+limit (clipped)."""
    code = np.zeros(len(v), dtype=np.uint8)
    ok = np.isfinite(v)
    code[ok] = (1 + np.rint((np.clip(v[ok] / limit, -1, 1) + 1) * 127)).astype(np.uint8)
    return base64.b64encode(code.tobytes()).decode("ascii")


def decode(text: str, limit: float) -> np.ndarray:
    code = np.frombuffer(base64.b64decode(text), dtype=np.uint8).astype(float)
    return np.where(code == 0, np.nan, ((code - 1) / 127 - 1) * limit)


def build_regulons(modules, counts: pd.DataFrame, fs: FeatureSet, ref_log: np.ndarray, ids: list, missing_path=None,
                   overlay: list[tuple] | None = None, log=print) -> dict:
    """The page's ``regulons`` block. ``counts``: features x samples table holding the atlas samples ``ids``.
    ``overlay``: ``[(sample ids in the page's order, their ids in the count table, count table, missing_genes.tsv or None)]``
    of the embedded samples, or None; their genes are standardised with the atlas statistics."""
    lacking = [s for s in ids if s not in counts.columns]
    if lacking:
        raise ValueError(f"{len(lacking)} atlas samples are not in the count table (e.g. {lacking[:3]}); the regulon values need "
                         "the counts of every atlas sample")
    sets = regulon_sets(modules, fs.ids)
    E = expression(counts[ids], fs, ref_log, missing_matrix(missing_path, ids, fs.ids))
    mu, sd = gene_stats(E)
    V = set_values(E, mu, sd, sets)
    lim = limits(V)
    sym, _ = feature_symbols(fs.ids)
    out = {"source": "RegulonDB", "sets": [], "n_empty": 0}
    for j, s in enumerate(sets):
        used = [g for g in s["members"] if np.isfinite(sd[g])]
        out["n_empty"] += not used
        out["sets"].append({"term": s["term"], "name": s["name"], "category": s["category"], "genes": [sym[g] for g in used],
                            "absent": s["absent"] + [sym[g] for g in s["members"] if g not in used],
                            "limit": float(lim[j]), "values": encode(V[:, j], lim[j])})
    log(f"regulons: {len(sets)} RegulonDB sets ({out['n_empty']} without a model gene), "
        f"{int(np.median([len(s['genes']) for s in out['sets']]))} genes in the median set")
    if overlay:
        parts = []
        for page_ids, table_ids, table, miss in overlay:
            if table is None:
                parts.append(np.full((len(page_ids), len(sets)), np.nan))
                continue
            absent = [s for s in table_ids if s not in table.columns]
            if absent:
                raise ValueError(f"the count table of your samples lacks {absent[:5]}")
            Eo = expression(table[table_ids], fs, ref_log, missing_matrix(miss, table_ids, fs.ids))
            parts.append(set_values(Eo, mu, sd, sets))
        Vo = np.vstack(parts)
        out["overlay"] = [encode(Vo[:, j], lim[j]) for j in range(len(sets))]
    return out
