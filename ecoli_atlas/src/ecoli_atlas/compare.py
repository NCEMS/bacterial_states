"""`ecoli_atlas compare`: place new samples relative to the atlas (the embedded baseline corpus).

Inputs are the ``final_embedding.csv`` of the new samples and an atlas directory holding
``final_embedding.csv`` plus, optionally, ``sample_predictions.csv``, ``atlas_metadata.csv`` and
``embedding_block_layout.csv``.

Outputs
  neighbors.tsv  one row per (query, rank): atlas sample, distance, and its metadata / predicted labels
  summary.tsv    one row per query: nearest atlas sample, dominant study, per-factor kNN vote and
                 agreement with the encoder prediction, per-block neighbour distance, novelty score
  umap_*.csv     optional 2-D UMAP fitted on the atlas with the queries projected onto it
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

EMBEDDING_FILE = "final_embedding.csv"
PREDICTIONS_FILE = "sample_predictions.csv"
METADATA_FILE = "atlas_metadata.csv"
LAYOUT_FILE = "embedding_block_layout.csv"
MANIFEST_FILE = "atlas_manifest.json"
NEIGHBOR_META_COLS = ("gse", "strain", "strain_background", "is_lab_strain", "medium_type", "base_medium",
                      "growth_phase", "carbon_source", "oxygen", "stress")
DEFAULT_FACTORS = ("growth_phase", "medium_type", "is_lab_strain")


def layout_from_columns(columns) -> pd.DataFrame:
    """Recover the block layout from column names like ``growth_phase[0]``."""
    rows, start = [], 0
    current, width = None, 0
    for c in list(columns) + [None]:
        name = c.split("[", 1)[0] if c else None
        if name != current:
            if current is not None:
                rows.append({"block": current, "kind": "residual" if current == "residual" else "factor",
                             "start": start, "end": start + width, "width": width})
                start += width
            current, width = name, 0
        width += 1
    return pd.DataFrame(rows)


@dataclass
class Atlas:
    embedding: pd.DataFrame
    predictions: pd.DataFrame | None
    metadata: pd.DataFrame | None
    layout: pd.DataFrame
    manifest: dict | None
    source: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Atlas":
        p = Path(path)
        emb_file = p if p.is_file() else p / EMBEDDING_FILE
        d = emb_file.parent
        emb = pd.read_csv(emb_file, index_col=0)
        emb.index = emb.index.astype(str)
        preds = pd.read_csv(d / PREDICTIONS_FILE, index_col=0) if (d / PREDICTIONS_FILE).exists() else None
        meta = pd.read_csv(d / METADATA_FILE, index_col=0, low_memory=False) if (d / METADATA_FILE).exists() else None
        if preds is not None:
            preds.index = preds.index.astype(str)
        if meta is not None:
            meta.index = meta.index.astype(str)
        layout = pd.read_csv(d / LAYOUT_FILE) if (d / LAYOUT_FILE).exists() else layout_from_columns(emb.columns)
        if "width" not in layout.columns:
            layout["width"] = layout["end"] - layout["start"]
        manifest = json.loads((d / MANIFEST_FILE).read_text()) if (d / MANIFEST_FILE).exists() else None
        return cls(emb, preds, meta, layout, manifest, d)

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
        """Atlas labels for a factor: curated metadata when present, else the encoder prediction."""
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


def compare(query: pd.DataFrame, atlas: Atlas, out_dir: str | Path, k: int = 15, metric: str = "cosine",
            blocks="all", query_predictions: pd.DataFrame | None = None, factors=DEFAULT_FACTORS,
            do_umap: bool = False, seed: int = 77, log=None) -> pd.DataFrame:
    log = log or (lambda *_: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    query = query.copy()
    query.index = query.index.astype(str)
    _check_columns(query, atlas)

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
    summary.to_csv(out / "summary.tsv", sep="\t")
    log(f"compared {len(query)} samples against {len(atlas.embedding)} atlas samples (k={k}, {metric}, blocks={blocks})")

    if do_umap:
        a_xy, q_xy = umap_project(atlas, query, seed, blocks)
        a_xy.to_csv(out / "umap_atlas.csv")
        q_xy.to_csv(out / "umap_query.csv")
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
    return summary
