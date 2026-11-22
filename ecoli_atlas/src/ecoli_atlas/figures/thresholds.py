"""Operating points: balanced accuracy against the threshold on the out-of-fold probabilities.

The out-of-fold scores are those of `train --cv` (``cv_predictions.tsv``: every sample scored by a
model that was fitted without its study). A bundle that has no such file gets them from one more
run of the study folds, kept as ``cv_predictions.tsv`` in the figure directory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HEADS
from ..metrics import classification_metrics
from .plotting import savefig
from .split_balance import TITLES

CV_FILE = "cv_predictions.tsv"
GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)
CALL_THRESHOLD = 0.5


def load_cv_predictions(*dirs) -> tuple:
    """The first ``cv_predictions.tsv`` found in ``dirs``: (frame, path), or (None, None)."""
    for d in dirs:
        p = Path(d) / CV_FILE
        if p.exists():
            return pd.read_csv(p, sep="\t", index_col=0, dtype={"sample": str}), p
    return None, None


def oof_scores(cv: pd.DataFrame, head: str) -> tuple:
    """(y, p) of the samples the out-of-fold metrics are made of: reference samples with a scored label."""
    neg, pos = HEADS[head]
    yes = lambda s: s.astype(str).str.strip().str.lower().isin(("true", "1"))       # noqa: E731
    sub = cv[yes(cv["reference"]) & yes(cv[f"{head}_scored"]) & cv[f"p_{pos}"].notna()]
    return (sub[f"{head}_label"] == pos).to_numpy().astype(int), sub[f"p_{pos}"].to_numpy(dtype=float)


def cv_metrics(cv: pd.DataFrame) -> dict:
    """As the manifest's ``cv_metrics``: every head at 0.5 on its out-of-fold scores."""
    out = {}
    for h in HEADS:
        y, p = oof_scores(cv, h)
        if len(y) and len(np.unique(y)) == 2:
            out[h] = classification_metrics(y, p, CALL_THRESHOLD)
    return out


def fig_threshold_curves(cv: pd.DataFrame, out_dir, formats) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import balanced_accuracy_score
    metrics = cv_metrics(cv)
    heads = list(metrics)
    fig, axes = plt.subplots(1, max(len(heads), 1), figsize=(5.5 * max(len(heads), 1), 4))
    axes = np.atleast_1d(axes)
    result = {}
    for ax, h in zip(axes, heads):
        y, p = oof_scores(cv, h)
        curve = np.array([balanced_accuracy_score(y, (p >= t).astype(int)) for t in GRID])
        best = int(np.argmax(curve))
        ax.plot(GRID, curve, color="#1f77b4")
        ax.axvline(CALL_THRESHOLD, color="crimson", ls="--", lw=1, label=f"call at {CALL_THRESHOLD}")
        if abs(GRID[best] - CALL_THRESHOLD) > 1e-9:
            ax.axvline(GRID[best], color="0.4", ls=":", lw=1.2, label=f"best {GRID[best]:.2f}")
        ax.set_title(f"{TITLES[h]} (n={len(y)}, AUROC={metrics[h]['auroc']:.2f})", fontsize=11)
        ax.set_xlabel(f"threshold on P({HEADS[h][1]})", fontsize=11)
        ax.set_ylabel("balanced accuracy", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="lower center")
        result[h] = metrics[h] | {"best_threshold": float(GRID[best]), "balanced_acc_at_best": float(curve[best]),
                                  "best_at_grid_boundary": bool(best in (0, len(GRID) - 1))}
    for ax in axes[len(heads):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_threshold_curves", formats)
    return result
