"""Curated gene modules (regulons, iModulons, pathways) scored with the attributions of `ecoli_atlas drivers`.

Module tables have the columns of the gene-set tables of ``gsea`` (``term``, ``namespace``, ``name``, ``genes`` as
comma-separated gene symbols or locus tags), so one table serves both `drivers --modules` and `--gsea --gene-sets`.
``build_module_table`` makes one from the sources that `ecoli_atlas download modules` fetches (``download.MODULE_SOURCES``):

  regulondb:<regulator>+ / -   RegulonDB regulon (gene-level TRN of SBRG/precise1k) split by the direction of the
                               regulation: the genes a regulator activates (``+``, "(activated)") and those it represses
                               (``-``, "(repressed)"); links of unknown or dual effect are left out (sigma factors: activated).
                               ``category`` marks a regulator with known effects in one direction only: "activator
                               only", "repressor only" or "sigma factor" (empty when it has sets in both directions)
  imodulon:<name>              PRECISE-1K iModulon: the ``imodulon_size`` genes with the largest |weight|
  kegg:<pathway or module>     KEGG pathway or module of E. coli K-12 (global overview maps left out)

Any other table of the same columns works too (an EcoCyc export, a lab's own gene lists).

Scoring (``score_modules``). A module's ``net`` is the sum of its members' attributions, averaged over the seeds and
the high-class studies: its part of the score gap between the two classes (the attributions of a sample add up to its
score minus the baseline's). It is compared with random modules of the same size that respect co-transcription: a
null draw is a random order of the expression units, whose genes are taken in that order up to the module's size
(one draw gives a null value for every size). ``z_net`` and the two-sided empirical ``p_net`` follow; ``fdr_net`` is
Benjamini-Hochberg within a source (the part of the namespace before its first ``_``: regulondb, imodulon, kegg). The null is a little anti-conservative (genes correlate beyond units) and
heavy-tailed (a few genes carry large attributions): read ``z_net`` with ``boot_in_top`` (the share of study draws in
which the module stays among the top modules of its source) rather than ``fdr_net`` alone.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MODULES_FILE = "modules.tsv"
RESULT_COLUMNS = ["term", "source", "namespace", "name", "size", "net", "share_gap", "gross", "coherence", "z_net", "p_net",
                  "fdr_net", "seeds_same_sign", "boot_same_sign", "boot_in_top", "mean_log2fc", "frac_same_dir",
                  "lead_genes"]


# ------------------------------------------------------------------------------------------------------- building

def _feature_maps(features: pd.DataFrame) -> tuple[dict, dict]:
    """locus tag -> symbol and lower-case symbol -> symbol, from a counting feature table (Symbol, Locus_Tag)."""
    sym = features["Symbol"].astype(str)
    lt = features["Locus_Tag"].astype(str) if "Locus_Tag" in features.columns else pd.Series("", index=features.index)
    by_locus = {l: s for l, s in zip(lt, sym) if l and s}
    return by_locus, {s.lower(): s for s in sym if s}


def kegg_categories(kegg_dir) -> dict:
    """KEGG classes from the BRITE hierarchies ``brite_pathways.keg`` (br08901) and ``brite_modules.keg`` (ko00002):
    map number ("00010") or module ("M00001") -> (class A, class B), e.g. ("Metabolism", "Carbohydrate metabolism").
    Empty when the files are absent (a module table built before they were downloaded)."""
    out = {}
    for fname, leaf in (("brite_pathways.keg", "C"), ("brite_modules.keg", "D")):
        path = Path(kegg_dir) / fname
        if not path.exists():
            continue
        a = b = ""
        for line in open(path, encoding="utf-8"):
            tag, rest = line[:1], line[1:].strip()
            if tag == "A":
                a, b = re.sub(r"<[^>]+>", "", rest), ""
            elif tag == "B":
                b = rest
            elif tag == leaf and rest:
                out[rest.split()[0]] = (a, b)
    return out


def build_module_table(source_dir, features: pd.DataFrame, sources=("precise1k", "kegg")) -> tuple[pd.DataFrame, dict]:
    """The module table (term, namespace, name, n_genes, genes as symbols) from the files that
    ``download.download_module_sources`` wrote under ``source_dir``; members are mapped to the symbols of the counting
    feature table by locus tag (b-number), else by symbol. Returns (table, per source: modules, listed members, matched)."""
    src = Path(source_dir)
    by_locus, by_sym = _feature_maps(features)
    rows, stats = [], defaultdict(lambda: [0, 0, 0])

    def add(source, namespace, term, name, members, category=""):
        listed = list(dict.fromkeys(m for m in members if m))
        genes = list(dict.fromkeys(g for g in (by_locus.get(m) or by_sym.get(m.lower()) for m in listed) if g))
        st = stats[source]
        st[0] += 1; st[1] += len(listed); st[2] += len(genes)
        if genes:
            rows.append({"term": term, "namespace": namespace, "name": name, "category": category, "n_genes": len(genes),
                         "genes": ",".join(genes)})

    if "precise1k" in sources:
        trn = pd.read_csv(src / "precise1k" / "TRN.csv")
        trn = trn[trn["source"].isin(["RegulonDB_TF", "RegulonDB_sigma"])]
        for reg, sub in trn.groupby("regulator"):
            # one set per direction of the regulation: the genes the regulator activates and those it represses; links
            # of unknown or dual effect ("?") are left out, and no set holds a whole regulon of mixed directions
            kind = "sigma_regulon" if (sub["source"] == "RegulonDB_sigma").all() else "tf_regulon"
            ids = [b if isinstance(b, str) and b else g for b, g in zip(sub["gene_id"], sub["gene_name"])]
            eff = sub["effect"].fillna("?").to_numpy()
            both = "+" in eff and "-" in eff
            for e, tag, only in (("+", "activated", "activator only"), ("-", "repressed", "repressor only")):
                part = [i for i, x in zip(ids, eff) if x == e]
                if part:                                 # category: empty for a regulator with sets in both directions
                    cat = "sigma factor" if kind == "sigma_regulon" else ("" if both else only)
                    add("regulondb", f"regulondb_{kind}_{tag}", f"regulondb:{reg}{e}", f"{reg} ({tag})", part, cat)
        M = pd.read_csv(src / "precise1k" / "M.csv", index_col=0)
        tab = pd.read_csv(src / "precise1k" / "imodulon_table.csv", index_col=0)
        for im in M.columns:
            if im not in tab.index or not int(tab.loc[im, "imodulon_size"] or 0):
                continue
            top = M[im].abs().sort_values(ascending=False).index[: int(tab.loc[im, "imodulon_size"])]
            cat = tab.loc[im, "system_category"]
            add("imodulon", f"imodulon_{str(cat).lower().replace(' ', '_') if isinstance(cat, str) else 'other'}",
                f"imodulon:{im}", im, [str(b) for b in top])
    if "kegg" in sources:
        kd = src / "kegg"
        category = kegg_categories(kd)
        names = {}
        for fname in ("list_pathway_eco.txt", "list_module.txt"):
            for line in open(kd / fname, encoding="utf-8"):
                k, _, v = line.rstrip("\n").partition("\t")
                names[k.split(":")[-1]] = v.replace(" - Escherichia coli K-12 MG1655", "")
        for fname, kind in (("link_pathway_eco.txt", "pathway"), ("link_module_eco.txt", "module")):
            sets = defaultdict(list)
            for line in open(kd / fname, encoding="utf-8"):
                g, _, p = line.rstrip("\n").partition("\t")
                sets[p.split(":")[-1]].append(g.split(":")[-1])
            for p, gs in sets.items():
                if kind == "pathway" and re.match(r"eco01[12]\d\d", p):
                    continue                                   # global and overview maps
                key = p[3:] if kind == "pathway" else p.split("_")[-1]          # eco00010 -> 00010, eco_M00001 -> M00001
                add("kegg", f"kegg_{kind}", f"kegg:{p}", names.get(p, names.get(p.replace("eco_", ""), p)), gs,
                    category.get(key, ("", ""))[1])
    table = pd.DataFrame(rows, columns=["term", "namespace", "name", "category", "n_genes", "genes"])
    return table, {k: {"modules": v[0], "members_listed": v[1], "members_matched": v[2]} for k, v in stats.items()}


def write_module_table(table: pd.DataFrame, path, header_lines=()) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        table.to_csv(fh, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def read_module_table(path) -> pd.DataFrame:
    """A module table (or any gene-set table: term, [namespace], [name], genes)."""
    t = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = {"term", "genes"} - set(t.columns)
    if missing:
        raise ValueError(f"{path}: module table lacks the column(s) {sorted(missing)}")
    for col in ("namespace", "name"):
        if col not in t.columns:
            t[col] = "modules" if col == "namespace" else t["term"]
    return t


def membership(table: pd.DataFrame, symbols, locus_tags, min_size: int = 3, max_size: int = 500) -> tuple[pd.DataFrame, np.ndarray]:
    """(modules kept, features x modules 0/1 matrix): members matched to the model features by lower-case symbol or
    locus tag; modules with ``min_size``..``max_size`` model features are kept."""
    idx = {}
    for i, (s, l) in enumerate(zip(symbols, locus_tags)):
        for key in (str(s).strip().lower(), str(l).strip().lower()):
            if key and key not in idx:
                idx[key] = i
    cols, keep = [], []
    for k, genes in enumerate(table["genes"]):
        m = sorted({idx[g.strip().lower()] for g in str(genes).split(",") if g.strip().lower() in idx})
        if min_size <= len(m) <= max_size:
            cols.append(m); keep.append(k)
    G = np.zeros((len(symbols), len(cols)), dtype=np.float32)
    for j, m in enumerate(cols):
        G[m, j] = 1.0
    return table.iloc[keep].reset_index(drop=True), G


# -------------------------------------------------------------------------------------------------------- scoring

def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); n = len(p)
    if not n:
        return p
    o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


def score_modules(attr: np.ndarray, unit: np.ndarray, modules: pd.DataFrame, G: np.ndarray, symbols, log2fc=None,
                  n_null: int = 10000, top_k: int = 25, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Scores of the modules (rows of ``modules``, columns of ``G``) for one contrast.

    ``attr``: seeds x high-class studies x features (the per-study mean attributions of `drivers`); ``unit``: the
    expression unit of every feature (``drivers.expression_units``); ``log2fc``: the data's fold change per feature.
    Returns RESULT_COLUMNS, sorted by |z_net| within each source."""
    A = np.asarray(attr, dtype=np.float64)
    a = A.mean((0, 1))
    per_study = A.mean(0)
    K = G.shape[1]
    size = G.sum(0).astype(int)
    net, gross = a @ G, np.abs(a) @ G
    # null: random orders of the expression units, cumulative sums taken at the module sizes
    uniq, inv = np.unique(unit, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(uniq))]
    sizes = np.unique(size)
    col = np.searchsorted(sizes, size)
    rng = np.random.default_rng(seed)
    nn = np.empty((n_null, len(sizes)))
    for d in range(n_null):
        order = np.concatenate([members[u] for u in rng.permutation(len(uniq))])
        nn[d] = np.cumsum(a[order])[sizes - 1]
    mu, sd = nn.mean(0)[col], nn.std(0)[col]
    sd = np.where(sd > 0, sd, 1.0)
    z = (net - mu) / sd
    p = (1 + (np.abs(nn[:, col] - mu) >= np.abs(net - mu)[None, :]).sum(0)) / (n_null + 1)
    out = modules[["term", "namespace", "name"]].copy()
    out.insert(1, "source", out["namespace"].astype(str).str.split("_").str[0])
    out["size"] = size
    out["net"], out["share_gap"], out["gross"] = net, net / (a.sum() if a.sum() else 1.0), gross
    out["coherence"] = np.abs(net) / np.maximum(gross, 1e-12)
    out["z_net"], out["p_net"] = z, p
    out["fdr_net"] = np.nan
    for _, ix in out.groupby("source").groups.items():
        out.loc[ix, "fdr_net"] = _bh(out.loc[ix, "p_net"].to_numpy())
    out["seeds_same_sign"] = (np.sign(A.mean(1) @ G) == np.sign(net)[None, :]).sum(0) if A.shape[0] > 1 else np.nan
    # study bootstrap with the null fixed (the draws of drivers.study_bootstrap)
    J = per_study.shape[0]
    brng = np.random.default_rng(0)
    same, intop = np.zeros(K), np.zeros(K)
    src_arr = out["source"].to_numpy()
    groups = {sv: np.where(src_arr == sv)[0] for sv in np.unique(src_arr)}
    for _ in range(n_boot):
        w = np.bincount(brng.integers(0, J, J), minlength=J) / J
        nb = (w @ per_study) @ G
        zb = (nb - mu) / sd
        same += np.sign(nb) == np.sign(net)
        for ix in groups.values():
            t = min(top_k, len(ix))
            intop[ix[np.argpartition(-np.abs(zb[ix]), t - 1)[:t]]] += 1
    out["boot_same_sign"], out["boot_in_top"] = same / n_boot, intop / n_boot
    if log2fc is not None:
        f = np.nan_to_num(np.asarray(log2fc, dtype=float))
        out["mean_log2fc"] = (f @ G) / np.maximum(size, 1)
        out["frac_same_dir"] = ((np.sign(f)[:, None] == np.sign(out["mean_log2fc"].to_numpy())[None, :]) * G).sum(0) / np.maximum(size, 1)
    else:
        out["mean_log2fc"] = out["frac_same_dir"] = np.nan
    sym = np.asarray(symbols, dtype=object)
    lead = []
    for k in range(K):
        m = np.where(G[:, k] > 0)[0]
        m = m[np.argsort(-np.abs(a[m]))][:6]
        lead.append(",".join(f"{sym[i]}({a[i]:+.3f})" for i in m))
    out["lead_genes"] = lead
    out = out.iloc[np.lexsort((-np.abs(out["z_net"].to_numpy()), out["source"].to_numpy()))].reset_index(drop=True)
    return out[RESULT_COLUMNS]


def summary(table: pd.DataFrame, n: int = 5) -> dict:
    """Per source: modules scored, |z| >= 3, and the top ``n`` by |z_net| that hold in 3 of 4 study draws."""
    out = {}
    for ns, sub in table.groupby("source"):
        robust = sub[sub["boot_in_top"] >= 0.75]
        out[ns] = {"modules": int(len(sub)), "abs_z_ge_3": int((sub["z_net"].abs() >= 3).sum()),
                   "top": [f"{r.name} ({r.z_net:+.1f})" for r in robust.head(n).itertuples()]}
    return out
