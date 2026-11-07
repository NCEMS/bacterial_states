"""Seed sweeps: summarise metrics across seeds, pick one bundle, record the spread.

With a TEST set of about 120 samples from a handful of studies, a single fit's balanced accuracy
moves by 0.1 to 0.2 between seeds. Judging a refit by one number is therefore not meaningful; the
honest statement is the spread over several seeds, which this module records in the promoted
bundle's ``manifest.json`` under ``seed_sweep`` and in ``seed_sweep.{json,tsv}``.

Seeds are ranked on the study-level out-of-fold (CV) scores of the training pool
(``threshold_diagnostics`` in each manifest, ~900 samples per factor), not on TEST: picking the
best of many seeds by TEST would make the promoted bundle's TEST metrics optimistically biased.
``on="test"`` is kept only to reproduce older sweeps.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import MANIFEST_FILE, PREPROCESSING_FILE, RESIDUAL_FILE, SPLITS_FILE, encoder_file

METRICS = ("balanced_acc", "auroc")


def summarize_sweep(manifests: dict) -> dict:
    """manifests: seed -> manifest dict (each with ``test_metrics`` and ``factor_thresholds``)."""
    seeds = list(manifests)
    factors = list(next(iter(manifests.values()))["factor_order"])
    per_seed = {}
    for s in seeds:
        m = manifests[s]
        cv = m.get("threshold_diagnostics") or {}
        per_seed[str(s)] = {f: {k: m["test_metrics"].get(f, {}).get(k) for k in METRICS}
                            | {f"cv_{k}": cv.get(f, {}).get(k) for k in METRICS}
                            | {"threshold": m["factor_thresholds"].get(f)}
                            for f in factors}

    def _spread(key):
        out = {}
        for f in factors:
            out[f] = {}
            for k in METRICS:
                vals = np.array([per_seed[str(s)][f][key(k)] for s in seeds if per_seed[str(s)][f][key(k)] is not None],
                                dtype=float)
                out[f][k] = ({"n": int(len(vals)), "min": float(vals.min()), "max": float(vals.max()),
                              "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
                             if len(vals) else {"n": 0})
        return out

    return {"seeds": seeds, "per_seed": per_seed, "spread": _spread(lambda k: k), "cv_spread": _spread(lambda k: f"cv_{k}")}


def seed_score(manifest: dict, on: str = "cv") -> float:
    """Mean AUROC over factors: out-of-fold CV on the training pool (``cv``) or TEST (``test``)."""
    if on == "cv":
        block = manifest.get("threshold_diagnostics") or {}
    elif on == "test":
        block = manifest["test_metrics"]
    else:
        raise ValueError(f"unknown selection metric {on!r}")
    vals = [v.get("auroc") for v in block.values() if v.get("auroc") is not None and not np.isnan(v["auroc"])]
    return float(np.mean(vals)) if vals else float("nan")


def select_seed(manifests: dict, rule: str = "best", on: str = "cv") -> int:
    seeds = list(manifests)
    if rule == "first":
        return seeds[0]
    scores = {s: seed_score(manifests[s], on) for s in seeds}
    if any(np.isnan(v) for v in scores.values()):
        missing = [s for s, v in scores.items() if np.isnan(v)]
        raise ValueError(f"seeds {missing} have no {on} AUROC to rank on"
                         + (" (trained with --no-thresholds?)" if on == "cv" else ""))
    if rule == "best":
        return max(seeds, key=lambda s: scores[s])
    if rule == "median":
        med = float(np.median(list(scores.values())))
        return min(seeds, key=lambda s: abs(scores[s] - med))
    raise ValueError(f"unknown selection rule {rule!r}")


def promote_bundle(src: Path, dest: Path, summary: dict) -> None:
    """Copy a seed's bundle files to ``dest`` and add the sweep summary to its manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    with open(src / MANIFEST_FILE, encoding="utf-8") as fh:
        manifest = json.load(fh)
    files = [encoder_file(f) for f in manifest["factor_order"]] + [RESIDUAL_FILE, PREPROCESSING_FILE, SPLITS_FILE]
    for name in files:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    manifest["seed_sweep"] = summary
    manifest["seed"] = summary["selected_seed"]
    with open(dest / MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


def write_sweep(summary: dict, out: Path) -> None:
    with open(out / "seed_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    rows = []
    for s, per_f in summary["per_seed"].items():
        for f, vals in per_f.items():
            rows.append({"seed": int(s), "factor": f, **vals, "selected": int(s) == summary["selected_seed"]})
    pd.DataFrame(rows).to_csv(out / "seed_sweep.tsv", sep="\t", index=False)


def within_spread(reference_metrics: dict, summary: dict) -> dict:
    """For each factor and metric: is the reference value inside [min, max] of the sweep?"""
    out = {}
    for f, st in summary["spread"].items():
        ref = reference_metrics.get(f, {})
        out[f] = {}
        for k in METRICS:
            if k in ref and st.get(k, {}).get("n", 0):
                out[f][k] = {"reference": ref[k], "min": st[k]["min"], "max": st[k]["max"],
                             "within": bool(st[k]["min"] <= ref[k] <= st[k]["max"])}
    return out
