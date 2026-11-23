"""`ecoli_atlas neighbors`: where do whole GEO studies that are already in the atlas sit?

For every sample of the given GSE accessions the command reports the nearest atlas samples in the
embedding, the votes of those neighbours on every call against the sample's own curated label and
the model's call, and a novelty score; a per-study table summarises each GSE. Same-study neighbours
are excluded by default (``--within-study`` keeps them).

Outputs
  neighbors.tsv       one row per (query sample, rank): atlas sample, distance, its metadata, split bucket, calls
  summary.tsv         one row per query sample
  study_summary.tsv   one row per GSE
  umap_atlas.tsv, umap.png   with --umap: a UMAP of the whole atlas with the query studies coloured
  neighbors_manifest.json    parameters, found / missing studies
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .artifacts import SPLITS_FILE
from .compare import DEFAULT_HEADS, NEIGHBOR_META_COLS, Atlas, knn, label_votes, novelty, per_block_distances, umap_fit
from .splits import bucket_labels, load_splits

BUCKET_ORDER = ("test", "fold", "train_only", "withheld", "projected", "no_split")


def split_buckets(splits: pd.DataFrame | None, withheld_studies, index) -> pd.Series:
    """Split bucket per atlas sample id: ``test`` | ``fold<k>`` | ``train_only`` (in the training pool
    but in no fold) | ``withheld`` (a study kept out of every split) | ``projected`` (in no split,
    e.g. the treated samples of a TEST study) | ``no_split`` (not in splits.csv)."""
    idx = pd.Index(index).astype(str)
    out = pd.Series("no_split", index=idx, dtype=object)
    if splits is None:
        return out
    lab = bucket_labels(splits)
    lab[(splits["split"] == 0).to_numpy() & splits["gse"].astype(str).isin([str(g) for g in withheld_studies or ()]).to_numpy()] = "withheld"
    lab.index = lab.index.astype(str)
    known = idx[idx.isin(lab.index)]
    out.loc[known] = lab.loc[known].to_numpy()
    return out


def select_query(atlas: Atlas, gses) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Atlas rows belonging to the given studies: (embedding, study per row, {found: {gse: n}, missing: [...]})."""
    if atlas.metadata is None or "gse" not in atlas.metadata.columns:
        raise ValueError("the atlas has no study (gse) for its samples; cannot select studies")
    gse_of = atlas.metadata.reindex(atlas.embedding.index)["gse"].astype(object)
    gse_of = gse_of.where(gse_of.notna(), None).map(lambda x: None if x is None else str(x))
    wanted = [str(g) for g in gses]
    mask = gse_of.isin(wanted).values
    found = {g: int((gse_of == g).sum()) for g in wanted if (gse_of == g).any()}
    missing = [g for g in wanted if g not in found]
    query = atlas.embedding.loc[mask].copy()
    return query, gse_of.loc[mask], {"found": found, "missing": missing}


def study_summary(neighbors: pd.DataFrame, summary: pd.DataFrame, query_meta: pd.DataFrame, heads, n_top: int = 5) -> pd.DataFrame:
    rows = []
    for gse, s in summary.groupby("query_gse", sort=True):
        nb = neighbors[neighbors["query"].isin(s.index)]
        rec = {"gse": gse, "n_query_samples": int(len(s)),
               "split_buckets": ";".join(f"{k}:{v}" for k, v in s["query_split_bucket"].value_counts().items())}
        if "gse" in nb.columns:
            vc = nb["gse"].dropna().astype(str).value_counts()
            rec["n_neighbor_studies"] = int(len(vc))
            rec["top_neighbor_studies"] = ";".join(f"{g}:{c}" for g, c in vc.head(n_top).items())
        for f in heads:
            if f"{f}_knn_vote" in s.columns:
                votes = s[f"{f}_knn_vote"].dropna().astype(str)
                rec[f"{f}_dominant_vote"] = votes.value_counts().idxmax() if len(votes) else None
                rec[f"{f}_dominant_vote_frac"] = float(votes.value_counts(normalize=True).max()) if len(votes) else np.nan
            if f"{f}_agree_meta" in s.columns:
                a = s[f"{f}_agree_meta"].dropna()
                rec[f"{f}_agree_meta_frac"] = float(a.astype(float).mean()) if len(a) else np.nan
            if f"{f}_agree_pred" in s.columns:
                a = s[f"{f}_agree_pred"].dropna()
                rec[f"{f}_agree_pred_frac"] = float(a.astype(float).mean()) if len(a) else np.nan
        rec["median_nearest_distance"] = float(s["nearest_distance"].median())
        rec["mean_novelty_percentile"] = float(s["novelty_percentile"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("gse")


def run_neighbors(gses, models_dir, out_dir, metadata=None, k: int = 15, metric: str = "cosine", blocks="all",
                  exclude_same_study: bool = True, do_umap: bool = False, seed: int = 77,
                  allow_missing: bool = False, heads=DEFAULT_HEADS, log=None) -> dict:
    """``models_dir``: the bundle whose embedding is the atlas. ``metadata``: the atlas sample
    metadata (optional; adds descriptive columns)."""
    log = log or (lambda *_: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atlas = Atlas.load(models_dir, metadata)
    query, q_gse, report = select_query(atlas, gses)
    for g, n in report["found"].items():
        log(f"  {g}: {n} atlas samples")
    if report["missing"]:
        msg = f"not in the atlas: {report['missing']}"
        if not allow_missing or not report["found"]:
            raise ValueError(msg + ("" if report["found"] else " (nothing found)") + "; use --allow-missing to continue")
        log("warning: " + msg)

    sp = Path(models_dir) / SPLITS_FILE
    splits = load_splits(sp) if sp.exists() else None
    withheld = list(((atlas.manifest or {}).get("config") or {}).get("withheld_studies") or [])
    bucket = split_buckets(splits, withheld, atlas.embedding.index)

    nb = knn(query, atlas, k, metric, blocks, exclude_self=True, exclude_studies=q_gse if exclude_same_study else None)
    nb.insert(1, "query_gse", nb["query"].map(q_gse))
    nb.insert(2, "query_split_bucket", nb["query"].map(bucket))
    nb["atlas_split_bucket"] = nb["atlas_sample"].map(bucket)
    nb.to_csv(out / "neighbors.tsv", sep="\t", index=False)

    summary = pd.DataFrame(index=pd.Index(query.index, name="sample"))
    summary["query_gse"] = q_gse.reindex(summary.index).values
    summary["query_split_bucket"] = bucket.reindex(summary.index).values
    if atlas.metadata is not None:
        meta_cols = [c for c in NEIGHBOR_META_COLS if c in atlas.metadata.columns and c != "gse" and c not in heads]
        summary = summary.join(atlas.metadata.reindex(summary.index)[meta_cols].add_prefix("query_"))
    first = nb[nb["rank"] == 1].set_index("query")
    summary["nearest_atlas_sample"] = first["atlas_sample"].reindex(summary.index)
    summary["nearest_distance"] = first["distance"].reindex(summary.index)
    summary["nearest_gse"] = first["gse"].reindex(summary.index) if "gse" in first.columns else None
    summary["mean_knn_distance"] = nb.groupby("query")["distance"].mean().reindex(summary.index)
    if "gse" in nb.columns:
        g = nb.dropna(subset=["gse"]).groupby("query")["gse"]
        summary["top_study"] = g.agg(lambda s: s.value_counts().idxmax()).reindex(summary.index)
        summary["top_study_frac"] = g.agg(lambda s: s.value_counts(normalize=True).max()).reindex(summary.index)
        summary["top_studies"] = g.agg(lambda s: ";".join(f"{a}:{b:.2f}" for a, b in s.value_counts(normalize=True).head(5).items())).reindex(summary.index)
    votes = label_votes(nb, atlas, heads)
    summary = summary.join(votes)
    for f in heads:
        if f"{f}_knn_vote" not in summary.columns:
            continue
        if atlas.metadata is not None and f in atlas.metadata.columns:
            own = atlas.metadata.reindex(summary.index)[f].astype(object)
            summary[f"{f}_meta"] = own.where(own.notna(), None).map(lambda x: None if x is None else str(x))
            summary[f"{f}_agree_meta"] = np.where(summary[f"{f}_meta"].isna() | summary[f"{f}_knn_vote"].isna(), None,
                                                  summary[f"{f}_meta"].astype(str) == summary[f"{f}_knn_vote"].astype(str))
        if atlas.predictions is not None and f"{f}_pred" in atlas.predictions.columns:
            summary[f"{f}_encoder_pred"] = atlas.predictions.reindex(summary.index)[f"{f}_pred"].astype(str)
            summary[f"{f}_agree_pred"] = np.where(summary[f"{f}_knn_vote"].isna(), None,
                                                  summary[f"{f}_encoder_pred"] == summary[f"{f}_knn_vote"].astype(str))
    summary = summary.join(per_block_distances(query, atlas, nb, metric))
    summary = summary.join(novelty(query, atlas, k, metric, blocks, query_studies=q_gse if exclude_same_study else None))

    if do_umap:
        cols = atlas.block_columns(blocks)
        reducer = umap_fit(atlas.embedding[cols].values, seed)
        a_xy = pd.DataFrame(reducer.embedding_, index=atlas.embedding.index, columns=["umap1", "umap2"])
        a_xy["gse"] = atlas.metadata.reindex(a_xy.index)["gse"].values if atlas.metadata is not None else None
        a_xy["queried"] = a_xy.index.isin(summary.index)
        a_xy.to_csv(out / "umap_atlas.tsv", sep="\t")
        summary["umap1"] = a_xy["umap1"].reindex(summary.index)
        summary["umap2"] = a_xy["umap2"].reindex(summary.index)
        _umap_plot(a_xy, summary, out / "umap.png")
    summary.to_csv(out / "summary.tsv", sep="\t")

    ss = study_summary(nb, summary, atlas.metadata, heads)
    ss.to_csv(out / "study_summary.tsv", sep="\t")
    report.update({"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "ecoli_atlas_version": __version__,
                   "models_dir": str(models_dir), "metadata": None if metadata is None else str(metadata),
                   "n_query": int(len(summary)), "k": k, "metric": metric, "blocks": blocks if blocks == "all" else list(blocks),
                   "exclude_same_study": exclude_same_study, "umap": do_umap, "seed": seed,
                   "withheld_studies": withheld, "buckets": {str(k_): int(v) for k_, v in summary["query_split_bucket"].value_counts().items()}})
    (out / "neighbors_manifest.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"{len(summary)} query samples from {len(report['found'])} studies; buckets {report['buckets']}")
    return report


def _umap_plot(a_xy: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(a_xy["umap1"], a_xy["umap2"], s=4, c="0.8", label=f"atlas (n={len(a_xy)})", zorder=1)
    gses = sorted(summary["query_gse"].dropna().unique())
    cmap = plt.cm.tab10 if len(gses) <= 10 else plt.cm.tab20
    for i, g in enumerate(gses):
        m = summary["query_gse"] == g
        ax.scatter(summary.loc[m, "umap1"], summary.loc[m, "umap2"], s=36, color=cmap(i % cmap.N), edgecolor="k",
                   linewidth=0.4, label=f"{g} (n={int(m.sum())})", zorder=2)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.2, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
