"""`ecoli_atlas compare`: place new samples relative to the atlas (the samples a bundle was trained on).

Inputs are the output directory of `ecoli_atlas embed` for the new samples (``embedding.tsv``,
``predictions.tsv``) and the model bundle, whose ``embedding.tsv`` / ``predictions.tsv`` are the atlas.
The atlas sample metadata (``metadata_v2.tsv``) is optional and adds descriptive columns to the
neighbours.

Outputs
  neighbors.tsv  one row per (query, rank): atlas sample, distance, and its metadata / calls
  summary.tsv    one row per query: nearest atlas sample, dominant study, the vote of the neighbours
                 on every call and its agreement with the model's own call, per-block neighbour
                 distance, novelty score
  umap_*.tsv     optional 2-D UMAP fitted on the atlas with the queries projected onto it
  query_distances.tsv  distances between the new samples themselves (same metric / blocks)

With sample metadata (``--metadata``) its columns are added to summary.tsv, and ``--group-by``
writes group_summary.tsv (one row per group, e.g. per time point or condition) plus the figures of
``compare_plots``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .config import HEADS

EMBEDDING_FILE = "embedding.tsv"
PREDICTIONS_FILE = "predictions.tsv"
MANIFEST_FILE = "manifest.json"
# columns of the atlas sample metadata shown beside every neighbour
NEIGHBOR_META_COLS = ("gse",) + tuple(HEADS) + ("strain_name", "strain_group", "medium_type", "base_medium", "growth_phase", "carbon_source",
                      "oxygen", "treatment", "genotype_class")
DEFAULT_FACTORS = tuple(HEADS)


def read_sample_table(path: str | Path) -> pd.DataFrame:
    """A table of samples (first column = sample id), comma- or tab-separated, as text."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
    df = pd.read_csv(path, sep="\t" if "\t" in first else ",", dtype=str, keep_default_na=False)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.strip()
    return df.replace("", np.nan)


def calls_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """The calls of a ``predictions.tsv`` (of a bundle or of `embed`) in one layout: per head
    ``<head>_pred`` (the class called at 0.5) and ``<head>_prob_<positive class>``."""
    out = pd.DataFrame(index=predictions.index.astype(str))
    for h, (neg, pos) in HEADS.items():
        if f"p_{pos}" not in predictions.columns:
            continue
        p = pd.to_numeric(predictions[f"p_{pos}"], errors="coerce").to_numpy()
        out[f"{h}_pred"] = np.where(np.isnan(p), None, np.where(p >= 0.5, pos, neg))
        out[f"{h}_prob_{pos}"] = p
    return out


def layout_from_manifest(manifest: dict) -> pd.DataFrame:
    rows = [{"block": b["block"], "kind": "residual" if b["block"] == "residual" else "factor", "start": int(b["start"]),
             "end": int(b["end"]), "width": int(b["end"]) - int(b["start"])} for b in manifest["embedding_blocks"]]
    return pd.DataFrame(rows)


def read_embedding(path: str | Path) -> pd.DataFrame:
    """``embedding.tsv`` of a bundle or of `embed` (a directory holding one is accepted)."""
    p = Path(path)
    emb = pd.read_csv(p / EMBEDDING_FILE if p.is_dir() else p, sep="\t", index_col=0)
    emb.index = emb.index.astype(str)
    return emb


@dataclass
class Atlas:
    """The embedded training corpus of a bundle.

    embedding    samples x embedding dimensions
    predictions  the model's calls (``calls_frame``)
    metadata     ``gse``, the curated label of every call under the head's name (``phase``,
                 ``medium``, ``lab_strain``; missing where a sample has none) and, when the atlas
                 sample metadata was given, its columns
    layout       block, start, end of the embedding blocks
    """
    embedding: pd.DataFrame
    predictions: pd.DataFrame | None
    metadata: pd.DataFrame | None
    layout: pd.DataFrame
    manifest: dict | None
    source: Path | None = None

    @classmethod
    def load(cls, path: str | Path, metadata: str | Path | None = None) -> "Atlas":
        d = Path(path)
        emb = read_embedding(d)
        manifest = json.loads((d / MANIFEST_FILE).read_text())
        raw = pd.read_csv(d / PREDICTIONS_FILE, sep="\t", index_col=0, dtype={"gse": str})
        raw.index = raw.index.astype(str)
        raw = raw.reindex(emb.index)
        meta = pd.DataFrame({"gse": raw["gse"]}, index=emb.index)
        for h in HEADS:
            if f"{h}_label" in raw.columns:
                meta[h] = raw[f"{h}_label"].astype(object).where(raw[f"{h}_label"].notna(), None)
        if metadata is not None:
            extra = read_sample_table(metadata).reindex(emb.index)
            meta = meta.join(extra[[c for c in extra.columns if c not in meta.columns and c != "GSE"]])
        return cls(emb, calls_frame(raw), meta, layout_from_manifest(manifest), manifest, d)

    @property
    def studies(self) -> np.ndarray | None:
        if self.metadata is not None and "gse" in self.metadata.columns:
            return self.metadata.reindex(self.embedding.index)["gse"].astype(str).where(
                self.metadata.reindex(self.embedding.index)["gse"].notna(), None).values
        return None

    def block_columns(self, blocks) -> list:
        if blocks in (None, "all", ("all",), ["all"]):
            return list(self.embedding.columns)
        wanted = set(blocks)
        unknown = wanted - set(self.layout["block"])
        if unknown:
            raise ValueError(f"unknown blocks {sorted(unknown)}; available: {list(self.layout['block'])}")
        cols = []
        for _, b in self.layout.iterrows():
            if b["block"] in wanted:
                cols += list(self.embedding.columns[int(b["start"]):int(b["end"])])
        return cols

    def label_source(self, factor: str) -> pd.Series | None:
        """Atlas labels of a call: the curated labels when the atlas has any, else the model's calls."""
        if self.metadata is not None and factor in self.metadata.columns:
            s = self.metadata.reindex(self.embedding.index)[factor]
            if s.notna().any():
                return s.astype(object).where(s.notna(), None)
        if self.predictions is not None and f"{factor}_pred" in self.predictions.columns:
            return self.predictions.reindex(self.embedding.index)[f"{factor}_pred"].astype(object)
        return None


def _check_columns(query: pd.DataFrame, atlas: Atlas) -> None:
    if list(query.columns) != list(atlas.embedding.columns):
        raise ValueError("query embedding columns do not match the atlas embedding columns; "
                         "were both produced by the same model bundle?")


def _study_array(atlas: Atlas) -> np.ndarray:
    """Study label per atlas row as an object array; rows without one get a unique placeholder."""
    studies = atlas.studies
    if studies is None:
        return np.array([f"__na{i}" for i in range(len(atlas.embedding))], dtype=object)
    return np.array([x if x is not None else f"__na{i}" for i, x in enumerate(studies)], dtype=object)


def _masked_knn(query: pd.DataFrame, atlas: Atlas, cols, k: int, metric: str, exclude_self: bool,
                exclude_studies: pd.Series | None):
    """Brute-force neighbours with per-query exclusions (self and/or same-study atlas rows)."""
    A = atlas.embedding[cols].values
    D = pairwise_distances(query[cols].values, A, metric=metric)
    ids = atlas.embedding.index.values
    if exclude_self:
        pos = pd.Series(np.arange(len(ids)), index=atlas.embedding.index)
        for qi, q in enumerate(query.index):
            if q in pos.index:
                D[qi, np.atleast_1d(pos.loc[q])] = np.inf
    if exclude_studies is not None:
        atlas_study = _study_array(atlas)
        qs = exclude_studies.reindex(query.index)
        for qi, s in enumerate(qs.values):
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                D[qi, atlas_study == str(s)] = np.inf
    kk = min(k, D.shape[1])
    idx = np.argsort(D, axis=1, kind="stable")[:, :kk]
    dist = np.take_along_axis(D, idx, axis=1)
    keep = np.isfinite(dist)
    return dist, idx, keep


def knn(query: pd.DataFrame, atlas: Atlas, k: int = 15, metric: str = "cosine", blocks="all",
        exclude_self: bool = False, exclude_studies: pd.Series | None = None) -> pd.DataFrame:
    """k nearest atlas samples per query row. By default the atlas may return the query itself
    (useful as a sanity check when the query is part of the atlas). ``exclude_self`` drops atlas rows
    with the query's own id; ``exclude_studies`` (a Series ``query id -> study``) drops atlas rows
    from the query's own study. Excluded rows are replaced by the next nearest ones."""
    _check_columns(query, atlas)
    cols = atlas.block_columns(blocks)
    ids = atlas.embedding.index.values
    if exclude_self or exclude_studies is not None:
        dist, idx, keep = _masked_knn(query, atlas, cols, k, metric, exclude_self, exclude_studies)
    else:
        nn = NearestNeighbors(n_neighbors=min(k, len(atlas.embedding)), metric=metric).fit(atlas.embedding[cols].values)
        dist, idx = nn.kneighbors(query[cols].values)
        keep = np.ones_like(dist, dtype=bool)
    rows = []
    for qi, q in enumerate(query.index):
        for r in range(idx.shape[1]):
            if keep[qi, r]:
                rows.append({"query": q, "rank": r + 1, "atlas_sample": ids[idx[qi, r]], "distance": float(dist[qi, r])})
    out = pd.DataFrame(rows, columns=["query", "rank", "atlas_sample", "distance"])
    if atlas.metadata is not None:
        meta_cols = [c for c in NEIGHBOR_META_COLS if c in atlas.metadata.columns]
        out = out.join(atlas.metadata[meta_cols], on="atlas_sample")
    if atlas.predictions is not None:
        pred_cols = [c for c in atlas.predictions.columns if c.endswith("_pred")]
        out = out.join(atlas.predictions[pred_cols].add_prefix("atlas_"), on="atlas_sample")
    return out


def label_votes(neighbors: pd.DataFrame, atlas: Atlas, factors=DEFAULT_FACTORS) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.Index(neighbors["query"].unique(), name="sample"))
    for f in factors:
        src = atlas.label_source(f)
        if src is None:
            continue
        lab = neighbors["atlas_sample"].map(src.to_dict())
        df = neighbors.assign(_lab=lab.astype(object)).dropna(subset=["_lab"])
        votes = df.groupby("query")["_lab"].agg(lambda s: s.astype(str).value_counts().idxmax())
        frac = df.groupby("query")["_lab"].agg(lambda s: s.astype(str).value_counts(normalize=True).max())
        n_lab = df.groupby("query").size()
        out[f"{f}_knn_vote"] = votes.reindex(out.index)
        out[f"{f}_knn_frac"] = frac.reindex(out.index)
        out[f"{f}_knn_n_labelled"] = n_lab.reindex(out.index).fillna(0).astype(int)
    return out


def per_block_distances(query: pd.DataFrame, atlas: Atlas, neighbors: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    """Mean distance from each query to its neighbours within each block's sub-space."""
    out = pd.DataFrame(index=pd.Index(query.index, name="sample"))
    for _, b in atlas.layout.iterrows():
        cols = list(atlas.embedding.columns[int(b["start"]):int(b["end"])])
        vals = []
        for q in query.index:
            nb = neighbors.loc[neighbors["query"] == q, "atlas_sample"].values
            d = pairwise_distances(query.loc[[q], cols].values, atlas.embedding.loc[nb, cols].values, metric=metric)
            vals.append(float(d.mean()))
        out[f"{b['block']}_knn_dist"] = vals
    return out


def novelty(query: pd.DataFrame, atlas: Atlas, k: int = 15, metric: str = "cosine", blocks="all",
            query_studies: pd.Series | None = None) -> pd.DataFrame:
    """Distance to the k-th nearest atlas neighbour, and its percentile within the same statistic
    computed for every atlas sample against other studies (self and same-study samples excluded when
    study labels are known). A percentile near 1 means the query is farther from the atlas than
    almost any atlas sample is from the rest of the atlas. When ``query_studies`` (``query id ->
    study``) is given, the query's own atlas row and same-study atlas rows are excluded as well, so
    atlas samples can be scored on the same footing as the reference distribution."""
    cols = atlas.block_columns(blocks)
    A = atlas.embedding[cols].values
    kk = min(k, len(A) - 1)
    D = pairwise_distances(A, A, metric=metric)
    np.fill_diagonal(D, np.inf)
    studies = atlas.studies
    s = _study_array(atlas)
    if studies is not None:
        D[s[:, None] == s[None, :]] = np.inf
    ref = np.partition(D, kk - 1, axis=1)[:, kk - 1]
    ref = ref[np.isfinite(ref)]
    Dq = pairwise_distances(query[cols].values, A, metric=metric)
    if query_studies is not None:
        pos = pd.Series(np.arange(len(A)), index=atlas.embedding.index)
        qs = query_studies.reindex(query.index)
        for qi, q in enumerate(query.index):
            if q in pos.index:
                Dq[qi, np.atleast_1d(pos.loc[q])] = np.inf
            st = qs.iloc[qi]
            if st is not None and not (isinstance(st, float) and np.isnan(st)):
                Dq[qi, s == str(st)] = np.inf
    dq = np.partition(Dq, kk - 1, axis=1)[:, kk - 1]
    pct = np.searchsorted(np.sort(ref), dq, side="right") / max(len(ref), 1)
    return pd.DataFrame({"novelty_knn_dist": dq, "novelty_percentile": pct}, index=pd.Index(query.index, name="sample"))


def umap_fit(E: np.ndarray, seed: int = 77, n_neighbors: int = 30, min_dist: float = 0.3, metric: str = "cosine"):
    """Fit the notebook's UMAP (n_neighbors=30, min_dist=0.3, cosine) and return the fitted reducer;
    ``n_neighbors`` is clamped for tiny inputs."""
    try:
        import umap
    except ImportError as e:  # pragma: no cover
        raise SystemExit("UMAP needs umap-learn: pip install 'ecoli-atlas[umap]'") from e
    n = int(np.asarray(E).shape[0])
    nn = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(n_neighbors=nn, min_dist=min_dist, metric=metric, random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")
        reducer.fit(np.asarray(E))
    return reducer


def umap_project(atlas: Atlas, query: pd.DataFrame, seed: int = 77, blocks="all", n_neighbors: int = 30,
                 min_dist: float = 0.3):
    cols = atlas.block_columns(blocks)
    reducer = umap_fit(atlas.embedding[cols].values, seed, n_neighbors, min_dist)
    a_xy = reducer.embedding_
    q_xy = reducer.transform(query[cols].values)
    a = pd.DataFrame(a_xy, index=atlas.embedding.index, columns=["umap1", "umap2"])
    q = pd.DataFrame(q_xy, index=query.index, columns=["umap1", "umap2"])
    return a, q


def query_distances(query: pd.DataFrame, atlas: Atlas, metric: str = "cosine", blocks="all") -> pd.DataFrame:
    """Square distance matrix between the query samples in the chosen blocks."""
    cols = atlas.block_columns(blocks)
    D = pairwise_distances(query[cols].values, metric=metric)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return pd.DataFrame(D, index=pd.Index(query.index, name="sample"), columns=list(query.index))


def _as_number(values) -> pd.Series | None:
    """The values as numbers when every non-missing one parses, else None."""
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    return num if num.notna().sum() == s.notna().sum() and s.notna().any() else None


def group_labels(metadata: pd.DataFrame, group_by) -> pd.Series:
    """One group label per sample: the ``group_by`` column, or several joined with `` | ``."""
    cols = [group_by] if isinstance(group_by, str) else list(group_by)
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"metadata lacks column(s) {missing}; available: {list(metadata.columns)}")
    parts = []
    for c in cols:
        num = _as_number(metadata[c].values)
        if num is not None:
            v = pd.Series([f"{x:g}" if pd.notna(x) else None for x in num], index=metadata.index, dtype=object)
        else:
            v = metadata[c].astype(object).where(metadata[c].notna(), None).map(lambda x: None if x is None else str(x))
        parts.append(v)
    lab = parts[0]
    for v in parts[1:]:
        lab = pd.Series([None if a is None or b is None else f"{a} | {b}" for a, b in zip(lab, v)], index=metadata.index, dtype=object)
    return lab.rename("group")


def group_order(labels: pd.Series, metadata: pd.DataFrame, order_by: str | None = None) -> list:
    """Groups in display order: by the group mean of ``order_by`` when it is numeric (else by its
    first value), without ``order_by`` by the label itself (numerically when the labels are numbers)."""
    groups = [g for g in pd.unique(labels.dropna())]
    if order_by is not None:
        if order_by not in metadata.columns:
            raise ValueError(f"metadata lacks the --order-by column {order_by!r}; available: {list(metadata.columns)}")
        num = _as_number(metadata[order_by].values)
        if num is not None:
            key = pd.Series(num.values, index=metadata.index).groupby(labels).mean()
        else:
            key = metadata[order_by].astype(str).groupby(labels).first()
        return sorted(groups, key=lambda g: (key.get(g), str(g)))
    num = _as_number(groups)
    if num is not None:
        return [g for _, g in sorted(zip(num.values, groups))]
    return sorted(groups, key=str)


def group_summary(query: pd.DataFrame, atlas: Atlas, neighbors: pd.DataFrame, summary: pd.DataFrame,
                  labels: pd.Series, order: list, query_predictions: pd.DataFrame | None = None,
                  metric: str = "cosine", blocks="all", factors=DEFAULT_FACTORS) -> pd.DataFrame:
    """One row per group of query samples, in ``order``.

    n; per factor the fraction of samples called each class (``<f>_called_<class>``), the mean class
    probability (``<f>_prob_<class>``), the majority label among the pooled neighbours
    (``<f>_knn_vote`` / ``_knn_frac``) and how often the encoder agrees with a sample's own vote
    (``<f>_agree``); ``top_studies`` of the pooled neighbours; mean nearest distance and novelty
    percentile; ``within_group_distance`` (mean distance between the group's samples, NaN for one
    sample); ``distance_to_previous`` (between the centroids of this group and the one before)."""
    cols = atlas.block_columns(blocks)
    labels = labels.reindex(query.index)
    preds = None
    if query_predictions is not None:
        preds = query_predictions.copy()
        preds.index = preds.index.astype(str)
        preds = preds.reindex(query.index)
    src = {f: atlas.label_source(f) for f in factors}
    rows, prev_centroid = [], None
    for g in order:
        members = list(labels.index[labels == g])
        row = {"group": g, "n": len(members), "samples": ";".join(members)}
        nb = neighbors[neighbors["query"].isin(members)]
        for f in factors:
            if preds is not None and f"{f}_pred" in preds.columns:
                called = preds.loc[members, f"{f}_pred"].astype(str).value_counts(normalize=True)
                for cls in sorted(preds[f"{f}_pred"].dropna().astype(str).unique()):
                    row[f"{f}_called_{cls}"] = float(called.get(cls, 0.0))
                for c in [c for c in preds.columns if c.startswith(f"{f}_prob_")]:
                    row[c] = float(preds.loc[members, c].mean())
            if src[f] is not None:
                lab = nb["atlas_sample"].map(src[f].to_dict()).dropna().astype(str)
                if len(lab):
                    vc = lab.value_counts(normalize=True)
                    row[f"{f}_knn_vote"], row[f"{f}_knn_frac"] = vc.idxmax(), float(vc.max())
            if f"{f}_agree" in summary.columns:
                row[f"{f}_agree"] = float(summary.loc[members, f"{f}_agree"].astype(float).mean())
        if "gse" in nb.columns and nb["gse"].notna().any():
            vc = nb["gse"].dropna().astype(str).value_counts(normalize=True)
            row["top_study"], row["top_study_frac"] = vc.idxmax(), float(vc.max())
            row["top_studies"] = ";".join(f"{a}:{b:.2f}" for a, b in vc.head(5).items())
        for c in ("nearest_distance", "mean_knn_distance", "novelty_knn_dist", "novelty_percentile"):
            if c in summary.columns:
                row[f"mean_{c}" if not c.startswith("mean_") else c] = float(summary.loc[members, c].mean())
        E = query.loc[members, cols].values
        if len(members) > 1:
            D = pairwise_distances(E, metric=metric)
            row["within_group_distance"] = float(D[np.triu_indices(len(members), 1)].mean())
        else:
            row["within_group_distance"] = np.nan
        centroid = E.mean(axis=0, keepdims=True)
        row["distance_to_previous"] = (np.nan if prev_centroid is None
                                       else float(pairwise_distances(centroid, prev_centroid, metric=metric)[0, 0]))
        prev_centroid = centroid
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def compare(query: pd.DataFrame, atlas: Atlas, out_dir: str | Path, k: int = 15, metric: str = "cosine",
            blocks="all", query_predictions: pd.DataFrame | None = None, factors=DEFAULT_FACTORS,
            do_umap: bool = False, seed: int = 77, log=None, metadata: pd.DataFrame | None = None,
            group_by=None, order_by: str | None = None, color_by: str | None = None, plots: bool = False,
            formats=("pdf", "png"), trajectory: bool | None = None) -> pd.DataFrame:
    log = log or (lambda *_: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    query = query.copy()
    query.index = query.index.astype(str)
    _check_columns(query, atlas)
    if metadata is not None:
        metadata = metadata.copy()
        metadata.index = metadata.index.astype(str)
        missing = [s for s in query.index if s not in metadata.index]
        if missing:
            raise ValueError(f"metadata lacks rows for {len(missing)} sample(s), e.g. {missing[:5]}")
        metadata = metadata.loc[query.index]
    elif group_by or order_by or color_by:
        raise ValueError("--group-by / --order-by / --color-by need --metadata")
    labels = group_labels(metadata, group_by) if group_by else None
    order = group_order(labels, metadata, order_by) if labels is not None else None

    nb = knn(query, atlas, k, metric, blocks)
    nb.to_csv(out / "neighbors.tsv", sep="\t", index=False)

    summary = pd.DataFrame(index=pd.Index(query.index, name="sample"))
    first = nb[nb["rank"] == 1].set_index("query")
    summary["nearest_atlas_sample"] = first["atlas_sample"].reindex(summary.index)
    summary["nearest_distance"] = first["distance"].reindex(summary.index)
    summary["mean_knn_distance"] = nb.groupby("query")["distance"].mean().reindex(summary.index)
    if "gse" in nb.columns:
        top = nb.dropna(subset=["gse"]).groupby("query")["gse"].agg(lambda s: s.value_counts().idxmax())
        frac = nb.dropna(subset=["gse"]).groupby("query")["gse"].agg(lambda s: s.value_counts(normalize=True).max())
        summary["top_study"] = top.reindex(summary.index)
        summary["top_study_frac"] = frac.reindex(summary.index)

    votes = label_votes(nb, atlas, factors)
    summary = summary.join(votes)
    if query_predictions is not None:
        qp = query_predictions.copy()
        qp.index = qp.index.astype(str)
        for f in factors:
            if f"{f}_pred" in qp.columns and f"{f}_knn_vote" in summary.columns:
                summary[f"{f}_encoder_pred"] = qp[f"{f}_pred"].reindex(summary.index).astype(str)
                summary[f"{f}_agree"] = summary[f"{f}_encoder_pred"] == summary[f"{f}_knn_vote"].astype(str)

    summary = summary.join(per_block_distances(query, atlas, nb, metric))
    summary = summary.join(novelty(query, atlas, k, metric, blocks))
    if metadata is not None:
        clash = [c for c in metadata.columns if c in summary.columns]
        meta_cols = metadata.rename(columns={c: f"meta_{c}" for c in clash})
        if labels is not None and "group" not in meta_cols.columns:
            meta_cols = meta_cols.assign(group=labels.values)
        summary = meta_cols.join(summary)
        summary.index.name = "sample"
    summary.to_csv(out / "summary.tsv", sep="\t")
    log(f"compared {len(query)} samples against {len(atlas.embedding)} atlas samples (k={k}, {metric}, blocks={blocks})")

    qd = query_distances(query, atlas, metric, blocks)
    qd.to_csv(out / "query_distances.tsv", sep="\t")
    groups = None
    if labels is not None:
        groups = group_summary(query, atlas, nb, summary, labels, order, query_predictions, metric, blocks, factors)
        groups.to_csv(out / "group_summary.tsv", sep="\t")
        log(f"{len(groups)} groups by {group_by}: wrote {out / 'group_summary.tsv'}")

    a_xy = q_xy = None
    if do_umap:
        a_xy, q_xy = umap_project(atlas, query, seed, blocks)
        a_xy.to_csv(out / "umap_atlas.tsv", sep="\t")
        q_xy.to_csv(out / "umap_query.tsv", sep="\t")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(a_xy["umap1"], a_xy["umap2"], s=4, c="0.75", label="atlas")
            ax.scatter(q_xy["umap1"], q_xy["umap2"], s=40, c="crimson", edgecolor="k", label="new samples")
            for s, r in q_xy.iterrows():
                ax.annotate(str(s), (r["umap1"], r["umap2"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out / "umap.png", dpi=200)
        except ImportError:
            log("matplotlib not installed; wrote UMAP coordinates only")

    if plots or labels is not None:
        try:
            from . import compare_plots
            color = None
            if color_by is not None:
                if color_by not in metadata.columns:
                    raise ValueError(f"metadata lacks the --color-by column {color_by!r}; available: {list(metadata.columns)}")
                color = group_labels(metadata, color_by)
                color_order = group_order(color, metadata, color_by)
            elif labels is not None:
                color, color_order = labels, order
            written = compare_plots.write_all(out, atlas=atlas, neighbors=nb, summary=summary, distances=qd,
                                              predictions=query_predictions, labels=labels, order=order,
                                              color=color, color_order=color_order if color is not None else None,
                                              color_name=color_by or (group_by if isinstance(group_by, str) else "group"),
                                              group_name=group_by if isinstance(group_by, str) or group_by is None else " | ".join(group_by),
                                              atlas_xy=a_xy, query_xy=q_xy, factors=factors, formats=formats,
                                              trajectory=trajectory)
            log("figures: " + ", ".join(written))
        except ImportError:
            log("matplotlib not installed; wrote tables only")
    return summary
