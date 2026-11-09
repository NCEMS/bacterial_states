"""Operating points: balanced accuracy vs threshold on out-of-fold scores (notebook cell 25).

The out-of-fold scores need one refit per fold and factor (the saved bundle holds only the final
encoders), so they are computed once, cached as ``oof_scores.csv`` in the output directory and
reused unless ``--force-retrain`` is given."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FactorHParams
from ..metrics import tune_threshold
from ..train_factor import factor_oof_scores
from .plotting import savefig

OOF_FILE = "oof_scores.csv"


def compute_oof(ctx, log=None) -> pd.DataFrame:
    """Refit every factor encoder on each fold and score the held-out fold (long table)."""
    log = log or ctx.log
    frames = []
    for f in ctx.factors:
        hp = ctx.cfg.factor_hparams.get(f)
        if hp is None:
            log(f"[{f}] no hyper-parameters in the manifest; refitting folds with defaults {FactorHParams()}")
            hp = FactorHParams()
        log(f"[{f}] refitting {ctx.cfg.n_folds} fold models ({ctx.cfg.factor_epochs} epochs each) on {ctx.device}")
        rows, p = factor_oof_scores(ctx.data, f, hp, ctx.cfg, ctx.device)
        if not len(rows):
            log(f"[{f}] no out-of-fold rows (no labelled validation fold)")
            continue
        codes = ctx.codes(f)[rows]
        frames.append(pd.DataFrame({"factor": f, "row": rows, "sample": ctx.meta.index.values[rows],
                                    "y_bin": (codes == ctx.data.factors[f].pos_idx).astype(int), "p_pos": p}))
    cols = ["factor", "row", "sample", "y_bin", "p_pos"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def save_oof(oof: pd.DataFrame, out_dir) -> Path:
    p = Path(out_dir) / OOF_FILE
    oof.to_csv(p, index=False)
    return p


def load_oof(out_dir) -> pd.DataFrame | None:
    p = Path(out_dir) / OOF_FILE
    return pd.read_csv(p) if p.exists() else None


def fig_threshold_curves(ctx, out_dir, formats, oof: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    grid = np.asarray(ctx.cfg.thresh_grid, dtype=float)
    factors = [f for f in ctx.factors if (oof["factor"] == f).any()]
    fig, axes = plt.subplots(1, max(len(factors), 1), figsize=(5 * max(len(factors), 1), 3.6))
    axes = np.atleast_1d(axes)
    result = {}
    for ax, f in zip(axes, factors):
        sub = oof[oof["factor"] == f]
        y_bin, p = sub["y_bin"].values.astype(int), sub["p_pos"].values.astype(float)
        curve = [balanced_accuracy_score(y_bin, (p >= t).astype(int)) for t in grid]
        thr_new, ba_new, at_edge = tune_threshold(y_bin, p, grid)
        thr_saved = ctx.bundle.factors[f].threshold
        auroc = float(roc_auc_score(y_bin, p)) if len(np.unique(y_bin)) > 1 else float("nan")
        ax.plot(grid, curve, color="#1f77b4")
        if thr_saved is not None:
            ax.axvline(thr_saved, color="crimson", ls="--", lw=1, label=f"saved thr={thr_saved:.3f}")
        if thr_saved is None or abs(thr_new - thr_saved) > 1e-9:
            ax.axvline(thr_new, color="0.4", ls=":", lw=1, label=f"refit thr={thr_new:.3f}")
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("threshold on P(positive)")
        ax.set_ylabel("balanced accuracy")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        result[f] = {"n": int(len(sub)), "threshold_saved": thr_saved, "threshold_refit": float(thr_new),
                     "balanced_acc_refit": float(ba_new), "auroc_refit": auroc, "refit_at_grid_boundary": bool(at_edge),
                     "balanced_acc_at_saved_threshold": (float(balanced_accuracy_score(y_bin, (p >= thr_saved).astype(int)))
                                                         if thr_saved is not None else None)}
    for ax in axes[len(factors):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_threshold_curves", formats)
    return result
