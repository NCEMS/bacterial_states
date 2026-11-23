"""The seeds of a bundle (``seed_<n>/`` beside the selected model) as a spread of the calls and scores.

A bundle trained with several seeds keeps every seed's model in ``seed_<n>/``; the bundle itself is the seed chosen on
out-of-fold scores. Scoring a sample with every seed shows how much its call or axis score depends on the randomness of
training (initialisation, batch order) with the data and the split fixed. It is not the sampling noise of the sample's
own counts, and the seeds can all agree on a call that is wrong for a sample unlike the training data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEEDS_DIR = "seeds"                      # `embed --seeds`: per-seed embedding.tsv and predictions.tsv in <out>/seeds/seed_<n>/
SEED_PREDICTIONS_FILE = "predictions_seeds.tsv"


def seed_dirs(root) -> list:
    """The ``seed_<n>`` bundle directories of ``root`` in seed order (empty when it holds one model only)."""
    root = Path(root)
    found = [d for d in root.glob("seed_*") if d.is_dir() and (d / "manifest.json").exists() and d.name[5:].isdigit()]
    return sorted(found, key=lambda d: int(d.name[5:]))


def spread(values: pd.DataFrame) -> pd.DataFrame:
    """Median, min and max over the columns (seeds) of ``values`` (samples x seeds)."""
    return pd.DataFrame({"median": values.median(axis=1), "min": values.min(axis=1), "max": values.max(axis=1)})


def seed_prediction_summary(per_seed: dict) -> pd.DataFrame:
    """{seed name: predictions.tsv frame} -> one row per sample: for every call probability ``p_*`` its median, min and
    max over the seeds, the number of seeds that make the call (probability at least 0.5), and the median, min and max
    of its log-odds (the probabilities pile up at 0 and 1 over the seeds; the log-odds show how far past the boundary
    each seed puts the sample)."""
    names = list(per_seed)
    first = per_seed[names[0]]
    probs = [c for c in first.columns if c.startswith("p_") and pd.api.types.is_numeric_dtype(first[c])]
    out = pd.DataFrame(index=first.index)
    out.index.name = first.index.name or "sample"
    out["n_seeds"] = len(names)
    for p in probs:
        P = pd.concat([per_seed[n][p].rename(n) for n in names], axis=1).reindex(first.index)
        s = spread(P)
        for k in ("median", "min", "max"):
            out[f"{p}_{k}"] = s[k]
        out[f"{p}_n_called"] = (P >= 0.5).sum(axis=1)
        clipped = P.clip(1e-6, 1 - 1e-6)
        L = spread(np.log(clipped / (1 - clipped)))
        for k in ("median", "min", "max"):
            out[f"{p}_logit_{k}"] = L[k]
    return out
