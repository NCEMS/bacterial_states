"""`ecoli_atlas extract`: the samples of atlas studies as an `embed` output.

The bundle already holds the embedding and the calls of every atlas sample (training, TEST, withheld). `extract` writes
those of the chosen studies in the layout of `embed` (``embedding.tsv``, ``predictions.tsv`` and, with every seed,
``seeds/seed_<n>/`` and ``predictions_seeds.tsv``), so that the tools made for new samples (`axes --query`, `compare`,
the call figures) can be used on a study the atlas holds. Those tools then need the study left out of the atlas
(``--exclude-study``), or its samples are their own nearest neighbours.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import HEADS
from .seeds import SEED_PREDICTIONS_FILE, SEEDS_DIR, seed_dirs, seed_prediction_summary


def _bundle_rows(bundle_dir: Path, studies: list) -> tuple:
    """(embedding, predictions in the layout of `embed`) of the samples of ``studies`` in one bundle directory."""
    emb = pd.read_csv(bundle_dir / "embedding.tsv", sep="\t", index_col=0)
    emb.index = emb.index.astype(str)
    raw = pd.read_csv(bundle_dir / "predictions.tsv", sep="\t", index_col=0, dtype={"gse": str})
    raw.index = raw.index.astype(str)
    ids = [s for s in raw.index[raw["gse"].astype(str).isin(studies)] if s in emb.index]
    pred = pd.DataFrame(index=pd.Index(ids, name="sample"))
    for h, (neg, pos) in HEADS.items():
        if f"p_{pos}" in raw.columns:
            p = raw.loc[ids, f"p_{pos}"].astype(float)
            pred[f"p_{pos}"] = p.to_numpy()
            pred[h] = [pos if v >= 0.5 else neg for v in p]
    out = emb.loc[ids]
    out.index.name = "sample"
    return out, pred, raw.loc[ids, ["gse", "split"]] if "split" in raw.columns else raw.loc[ids, ["gse"]]


def extract_studies(models, studies: list, out, seeds: bool = False) -> dict:
    """Write the samples of ``studies`` of the bundle ``models`` to ``out`` as `embed` would; returns a summary."""
    models, out = Path(models), Path(out)
    studies = [str(s) for s in studies]
    emb, pred, info = _bundle_rows(models, studies)
    if emb.empty:
        raise ValueError(f"no sample of {', '.join(studies)} in the embedding of {models}")
    found = sorted(set(info["gse"].astype(str)))
    out.mkdir(parents=True, exist_ok=True)
    emb.to_csv(out / "embedding.tsv", sep="\t", float_format="%.6g")
    pred.to_csv(out / "predictions.tsv", sep="\t", float_format="%.6g")
    info.rename_axis("sample").to_csv(out / "samples.tsv", sep="\t")
    summary = {"models": str(models), "studies": found, "missing_studies": sorted(set(studies) - set(found)),
               "n_samples": int(len(emb)), "seeds": []}
    if seeds:
        dirs = seed_dirs(models)
        if not dirs:
            raise ValueError(f"--seeds: {models} has no seed_<n> bundles (a bundle trained with one seed)")
        per_seed = {}
        for d in dirs:
            e, p, _ = _bundle_rows(d, studies)
            if list(e.index) != list(emb.index):
                raise ValueError(f"--seeds: the embedding of {d.name} does not hold the same samples of {', '.join(found)}")
            sd = out / SEEDS_DIR / d.name
            sd.mkdir(parents=True, exist_ok=True)
            e.to_csv(sd / "embedding.tsv", sep="\t", float_format="%.6g")
            p.to_csv(sd / "predictions.tsv", sep="\t", float_format="%.6g")
            per_seed[d.name] = p
        seed_prediction_summary(per_seed).to_csv(out / SEED_PREDICTIONS_FILE, sep="\t", float_format="%.6g")
        summary["seeds"] = list(per_seed)
    (out / "extract.json").write_text(json.dumps(summary, indent=1))
    return summary
