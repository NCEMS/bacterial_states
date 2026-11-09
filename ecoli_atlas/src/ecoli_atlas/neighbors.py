"""`ecoli_atlas neighbors`: where do whole GEO studies that are already in the atlas sit?

For every sample of the given GSE accessions the command reports the nearest atlas samples in latent
space, the label votes of those neighbours against the sample's own metadata and encoder prediction,
and a novelty score; a per-study table summarises each GSE. Same-study neighbours are excluded by
default (``--within-study`` keeps them), and the sequencing runs of one GSM are averaged into one
row by default (``--no-collapse-runs`` keeps SRR-level rows).

Outputs
  neighbors.tsv       one row per (query sample, rank): atlas sample, distance, its metadata, split bucket, predictions
  summary.tsv         one row per query sample
  study_summary.tsv   one row per GSE
  umap_atlas.csv, umap.png   with --umap: a UMAP of the whole atlas with the query studies coloured
  neighbors_manifest.json    parameters, found / missing studies
"""
from __future__ import annotations

import datetime
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .artifacts import MANIFEST_FILE, load_splits
from .compare import DEFAULT_FACTORS, NEIGHBOR_META_COLS, Atlas, knn, label_votes, novelty, per_block_distances, umap_fit
from .labels import collapse_runs

BUCKET_ORDER = ("test", "fold", "train(aug)", "withheld", "projected_only", "no_metadata")


def split_buckets(splits: pd.DataFrame | None, withheld_study: str | None, index, studies: pd.Series | None = None) -> pd.Series:
    """Split bucket per atlas sample id: ``test`` | ``fold<k>`` | ``train(aug)`` | ``withheld`` |
    ``projected_only`` | ``no_metadata`` (not in splits.csv). For collapsed (GSM-level) ids the bucket
    of the first run with that accession is used."""
    idx = pd.Index(index).astype(str)
    out = pd.Series("no_metadata", index=idx, dtype=object)
    if splits is None:
        return out
    sp = splits.copy()
    sp.index = sp.index.astype(str)
    from .labels import gsm_accession
    by_gsm = sp.groupby(gsm_accession(sp.index).fillna(pd.Series(sp.index, index=sp.index)).values).first()
    for key, table in ((sp, idx[idx.isin(sp.index)]), (by_gsm, idx[~idx.isin(sp.index) & idx.isin(by_gsm.index)])):
        if len(table) == 0:
            continue
        rows = key.loc[table]
        lab = pd.Series("projected_only", index=table, dtype=object)
        lab[(rows["split"] == 3).values] = "test"
        pool = (rows["split"] == 1).values
        fold = rows["cv_fold"]
        lab[pool & fold.notna().values] = "fold" + fold[pool & fold.notna().values].astype(int).astype(str)
        lab[pool & fold.isna().values] = "train(aug)"
        if withheld_study is not None and "gse" in rows.columns:
            lab[(rows["split"] == 0).values & (rows["gse"].astype(str) == str(withheld_study)).values] = "withheld"
        out.loc[table] = lab
    return out


def collapse_atlas(atlas: Atlas, how: str = "mean") -> Atlas:
    """Average the SRR-level rows of one GSM in the embedding (and predictions / metadata)."""
    emb = collapse_runs(atlas.embedding, how)
    preds = collapse_runs(atlas.predictions, how).reindex(emb.index) if atlas.predictions is not None else None
    meta = collapse_runs(atlas.metadata, how).reindex(emb.index) if atlas.metadata is not None else None
    if preds is not None:
        for c in [c for c in preds.columns if c.endswith("_pred")]:
            # a GSM's runs may disagree: keep the majority call among its runs
            votes = atlas.predictions[c].groupby(_gsm_key(atlas.predictions.index)).agg(lambda s: s.astype(str).value_counts().idxmax())
            preds[c] = votes.reindex(preds.index).values
    return replace(atlas, embedding=emb, predictions=preds, metadata=meta)


def _gsm_key(index) -> np.ndarray:
    from .labels import gsm_accession
    acc = gsm_accession(index)
    return acc.where(acc.notna(), pd.Series(pd.Index(index).astype(str), index=acc.index)).values


def select_query(atlas: Atlas, gses) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Atlas rows belonging to the given studies: (embedding, study per row, {found: {gse: n}, missing: [...]})."""
    if atlas.metadata is None or "gse" not in atlas.metadata.columns:
        raise ValueError("the atlas has no atlas_metadata.csv with a gse column; cannot select studies")
    gse_of = atlas.metadata.reindex(atlas.embedding.index)["gse"].astype(object)
    gse_of = gse_of.where(gse_of.notna(), None).map(lambda x: None if x is None else str(x))
    wanted = [str(g) for g in gses]
    mask = gse_of.isin(wanted).values
    found = {g: int((gse_of == g).sum()) for g in wanted if (gse_of == g).any()}
    missing = [g for g in wanted if g not in found]
    query = atlas.embedding.loc[mask].copy()
    return query, gse_of.loc[mask], {"found": found, "missing": missing}


def study_summary(neighbors: pd.DataFrame, summary: pd.DataFrame, query_meta: pd.DataFrame, factors, n_top: int = 5) -> pd.DataFrame:
    rows = []
    for gse, s in summary.groupby("query_gse", sort=True):
        nb = neighbors[neighbors["query"].isin(s.index)]
        rec = {"gse": gse, "n_rows": int(s["n_runs"].sum()) if "n_runs" in s.columns else int(len(s)), "n_query_samples": int(len(s)),
               "split_buckets": ";".join(f"{k}:{v}" for k, v in s["query_split_bucket"].value_counts().items())}
        if "gse" in nb.columns:
            vc = nb["gse"].dropna().astype(str).value_counts()
            rec["n_neighbor_studies"] = int(len(vc))
            rec["top_neighbor_studies"] = ";".join(f"{g}:{c}" for g, c in vc.head(n_top).items())
        for f in factors:
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


def run_neighbors(gses, atlas_dir, out_dir, models_dir=None, k: int = 15, metric: str = "cosine", blocks="all",
                  exclude_same_study: bool = True, collapse_runs_: bool = True, do_umap: bool = False, seed: int = 77,
                  allow_missing: bool = False, factors=DEFAULT_FACTORS, log=None, **kw) -> dict:
    log = log or (lambda *_: None)
    if "collapse_runs" in kw:
        collapse_runs_ = kw.pop("collapse_runs")
    if kw:
        raise TypeError(f"unexpected arguments {sorted(kw)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atlas = Atlas.load(atlas_dir)
    n_rows = len(atlas.embedding)
    if collapse_runs_:
        atlas = collapse_atlas(atlas)
        log(f"atlas: {n_rows} rows -> {len(atlas.embedding)} samples after averaging runs per GSM")
    query, q_gse, report = select_query(atlas, gses)
    for g, n in report["found"].items():
        log(f"  {g}: {n} atlas samples")
    if report["missing"]:
        msg = f"not in the atlas: {report['missing']}"
        if not allow_missing or not report["found"]:
            raise ValueError(msg + ("" if report["found"] else " (nothing found)") + "; use --allow-missing to continue")
        log("warning: " + msg)

    splits, withheld = None, None
    if models_dir:
        splits = load_splits(models_dir)
        mf = Path(models_dir) / MANIFEST_FILE
        if mf.exists():
            withheld = json.loads(mf.read_text()).get("split_config", {}).get("withheld_study")
    bucket = split_buckets(splits, withheld, atlas.embedding.index)

    nb = knn(query, atlas, k, metric, blocks, exclude_self=True, exclude_studies=q_gse if exclude_same_study else None)
    nb.insert(1, "query_gse", nb["query"].map(q_gse))
    nb.insert(2, "query_split_bucket", nb["query"].map(bucket))
    nb["atlas_split_bucket"] = nb["atlas_sample"].map(bucket)
    nb.to_csv(out / "neighbors.tsv", sep="\t", index=False)

    summary = pd.DataFrame(index=pd.Index(query.index, name="sample"))
    summary["query_gse"] = q_gse.reindex(summary.index).values
    summary["query_split_bucket"] = bucket.reindex(summary.index).values
    if collapse_runs_:
        runs = pd.Series(_gsm_key(pd.read_csv(Path(atlas_dir) / "final_embedding.csv", usecols=[0]).iloc[:, 0].astype(str).values))
        summary["n_runs"] = runs.value_counts().reindex(summary.index).fillna(1).astype(int).values
    if atlas.metadata is not None:
        meta_cols = [c for c in NEIGHBOR_META_COLS if c in atlas.metadata.columns and c != "gse"]
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
    votes = label_votes(nb, atlas, factors)
    summary = summary.join(votes)
    for f in factors:
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
        a_xy.to_csv(out / "umap_atlas.csv")
        summary["umap1"] = a_xy["umap1"].reindex(summary.index)
        summary["umap2"] = a_xy["umap2"].reindex(summary.index)
        _umap_plot(a_xy, summary, out / "umap.png")
    summary.to_csv(out / "summary.tsv", sep="\t")

    ss = study_summary(nb, summary, atlas.metadata, factors)
    ss.to_csv(out / "study_summary.tsv", sep="\t")
    report.update({"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "ecoli_atlas_version": __version__,
                   "atlas_dir": str(atlas_dir), "models_dir": None if models_dir is None else str(models_dir),
                   "n_query": int(len(summary)), "k": k, "metric": metric, "blocks": blocks if blocks == "all" else list(blocks),
                   "exclude_same_study": exclude_same_study, "collapse_runs": collapse_runs_, "umap": do_umap, "seed": seed,
                   "withheld_study": withheld, "buckets": {str(k_): int(v) for k_, v in summary["query_split_bucket"].value_counts().items()}})
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
