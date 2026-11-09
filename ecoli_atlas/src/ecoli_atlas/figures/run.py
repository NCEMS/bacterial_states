"""Orchestrate the figure functions over one context and record every number in a JSON sidecar."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import FIGURES
from .checks import all_pass, compare_to_manifest
from .context import build_context
from .plotting import DEFAULT_FORMATS, apply_style

SIDECAR = "figure_metrics.json"


def refuse_reference_dir(out_dir: Path, models_dir: Path) -> None:
    """Never write into the bundle or next to the Colab reference figures / atlas."""
    out = out_dir.resolve()
    protected = {models_dir.resolve(), (models_dir.parent / "figures").resolve(), (models_dir.parent / "atlas").resolve()}
    if out in protected:
        raise SystemExit(f"refusing to write figures into {out_dir}: that is the model bundle or a reference directory; "
                         "use e.g. outputs/figures_repro")


def select_figures(only=None, skip=None, retrain: bool = True) -> list:
    names = list(FIGURES)
    if only:
        unknown = [n for n in only if n not in FIGURES]
        if unknown:
            raise SystemExit(f"unknown figure(s) {unknown}; available: {names}")
        names = [n for n in names if n in set(only)]
    if skip:
        unknown = [n for n in skip if n not in FIGURES]
        if unknown:
            raise SystemExit(f"unknown figure(s) {unknown}; available: {list(FIGURES)}")
        names = [n for n in names if n not in set(skip)]
    if not retrain:
        names = [n for n in names if not FIGURES[n].needs_retrain]
    return names


def atlas_embedding_diff(ctx, atlas_dir: Path) -> dict:
    """Max absolute difference between the context's embedding and a saved atlas on shared samples."""
    emb = pd.read_csv(Path(atlas_dir) / "final_embedding.csv", index_col=0)
    emb.index = emb.index.astype(str)
    common = ctx.res.embedding.index.intersection(emb.index)
    if len(common) == 0 or list(emb.columns) != list(ctx.res.embedding.columns):
        return {"n_common": int(len(common)), "max_abs_diff": None}
    d = np.abs(ctx.res.embedding.loc[common].values - emb.loc[common].values)
    return {"n_common": int(len(common)), "n_atlas": int(len(emb)), "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean())}


def run_figures(counts, meta_raw, feat, models_dir, out_dir, only=None, skip=None, formats=DEFAULT_FORMATS,
                seed=None, device="cpu", retrain: bool = True, force_retrain: bool = False, recombat: bool = False,
                atlas_dir=None, reference_sweep=None, log=None, ctx=None) -> tuple[dict, int]:
    """Returns (sidecar dict, number of failed figures). Figures run independently: a failure is
    recorded with its traceback and the remaining figures still run."""
    log = log or (lambda *_: None)
    models_dir, out = Path(models_dir), Path(out_dir)
    refuse_reference_dir(out, models_dir)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    names = select_figures(only, skip, retrain)
    t0 = time.time()
    if ctx is None:
        ctx = build_context(counts, meta_raw, feat, models_dir, device=device, seed=seed, log=log)
    side = {"models_dir": str(models_dir), "out_dir": str(out), "seed": ctx.seed, "device": str(device),
            "n_samples": int(ctx.data.n_samples), "n_genes": int(ctx.data.n_genes),
            "n_test": int(len(ctx.data.te_rows)), "n_pool": int(len(ctx.data.tr_rows)),
            "zscore_max_abs_dev": ctx.zscore_max_dev, "gene_coverage": ctx.gene_coverage,
            "context_seconds": round(time.time() - t0, 1), "figures": {}, "metrics": {}, "failed": []}
    if atlas_dir:
        side["atlas_dir"] = str(atlas_dir)
        side["atlas_comparison"] = atlas_embedding_diff(ctx, Path(atlas_dir))
        side["metrics"]["atlas_embedding_max_abs_diff"] = side["atlas_comparison"]["max_abs_diff"]
        log(f"atlas comparison: {side['atlas_comparison']}")

    rc_emb = None
    if recombat and "study_leakage" in names:
        from .recombat import recombat_embedding
        rc_emb = recombat_embedding(ctx, log)   # ImportError propagates: --recombat was explicit

    for name in names:
        spec = FIGURES[name]
        t1 = time.time()
        try:
            result = _run_one(name, ctx, out, formats, force_retrain, rc_emb, log, side)
            side["figures"][name] = {"status": "ok", "seconds": round(time.time() - t1, 1), "outputs": list(spec.outputs),
                                     "result": result}
            log(f"[{name}] ok ({time.time() - t1:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            side["figures"][name] = {"status": "failed", "seconds": round(time.time() - t1, 1), "error": repr(exc),
                                     "traceback": traceback.format_exc()}
            side["failed"].append(name)
            log(f"[{name}] FAILED: {exc!r}")

    # figures of an earlier run into the same directory stay in the sidecar (e.g. a later --only threshold_curves)
    previous = _load_sidecar(out)
    if previous and previous.get("models_dir") == str(models_dir):
        for name, rec in previous.get("figures", {}).items():
            if name not in side["figures"] and name not in names:
                side["figures"][name] = rec
        for key, val in previous.get("metrics", {}).items():
            side["metrics"].setdefault(key, val)
    sweep = None
    if reference_sweep:
        sweep = json.loads(Path(reference_sweep).read_text()).get("seed_sweep")
        side["reference_sweep"] = str(reference_sweep)
        if sweep is None:
            log(f"warning: {reference_sweep} has no seed_sweep block; out-of-fold checks use the fixed tolerances")
    side["checks"] = compare_to_manifest(side["metrics"], ctx.bundle.manifest, sweep=sweep)
    side["all_checks_pass"] = all_pass(side["checks"])
    side["total_seconds"] = round(time.time() - t0, 1)
    (out / SIDECAR).write_text(json.dumps(side, indent=2, default=_json_default))
    return side, len(side["failed"])


def _load_sidecar(out: Path) -> dict | None:
    p = out / SIDECAR
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _run_one(name, ctx, out, formats, force_retrain, rc_emb, log, side):
    if name == "corpus_composition":
        from .corpus import fig_corpus_composition
        return fig_corpus_composition(ctx, out, formats)
    if name == "factor_test_eval":
        from .test_eval import fig_factor_test_eval, growth_phase_misclassified_table
        m = fig_factor_test_eval(ctx, out, formats)
        side["metrics"]["test_metrics"] = m
        for f, v in m.items():
            log(f"[{f}] TEST n={v['n']} | balanced acc={v['balanced_acc']:.3f} [{v['ba_ci'][0]:.3f}, {v['ba_ci'][1]:.3f}] "
                f"| AUROC={v['auroc']:.3f}")
        return {"test_metrics": m, "growth_phase_misclassified": growth_phase_misclassified_table(ctx, out)}
    if name == "split_balance":
        from .split_balance import fig_split_balance
        return fig_split_balance(ctx, out, formats)
    if name == "pca_batch":
        from .pca import fig_pca_batch
        return fig_pca_batch(ctx, out, formats)
    if name == "batch_genes":
        from .pca import fig_batch_genes
        return fig_batch_genes(ctx, out, formats)
    if name == "factor_umaps":
        from .umaps import fig_factor_umaps
        return fig_factor_umaps(ctx, out, formats)
    if name == "factor_gene_correlates":
        from .gene_correlates import fig_factor_gene_correlates
        return fig_factor_gene_correlates(ctx, out, formats)
    if name == "study_leakage":
        from .leakage import fig_study_leakage
        st = fig_study_leakage(ctx, out, formats, rc_emb)
        side["metrics"]["leakage"] = st
        return st
    if name == "threshold_curves":
        from .thresholds import compute_oof, fig_threshold_curves, load_oof, save_oof
        oof = None if force_retrain else load_oof(out)
        if oof is not None and set(oof["factor"]) >= set(ctx.factors):
            log(f"[threshold_curves] reusing {out / 'oof_scores.csv'} (use --force-retrain to refit)")
        else:
            oof = compute_oof(ctx, log)
            save_oof(oof, out)
        diag = fig_threshold_curves(ctx, out, formats, oof)
        side["metrics"]["threshold_diagnostics"] = diag
        for f, v in diag.items():
            log(f"[{f}] out-of-fold n={v['n']} | refit threshold={v['threshold_refit']:.3f} (saved {v['threshold_saved']}) "
                f"| balanced acc={v['balanced_acc_refit']:.3f} | AUROC={v['auroc_refit']:.3f}")
        return diag
    raise KeyError(name)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    return str(o)
