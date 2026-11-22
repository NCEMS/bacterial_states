"""Comparison of metrics made for the figures with those stored in a bundle's manifest."""
from __future__ import annotations

KEYS = ("balanced_acc", "auroc")


def compare_metrics(new: dict, saved: dict | None, tol: float = 1e-6) -> dict:
    """Per head: the values here and in the manifest, their largest difference, and whether they
    agree (same number of samples, differences within ``tol``)."""
    if not saved:
        return {"skipped": "the manifest has no such metrics"}
    out = {}
    for h in sorted(set(new) | set(saved)):
        a, b = new.get(h), saved.get(h)
        if a is None or b is None:
            out[h] = {"agree": False, "missing": "here" if a is None else "manifest"}
            continue
        diff = max(abs(a[k] - b[k]) for k in KEYS)
        out[h] = {"n": a["n"], "n_manifest": b["n"], "max_abs_diff": float(diff), "agree": bool(a["n"] == b["n"] and diff <= tol)}
        out[h] |= {k: a[k] for k in KEYS} | {f"{k}_manifest": b[k] for k in KEYS}
    return out


def all_agree(cmp: dict) -> bool:
    return "skipped" not in cmp and all(v["agree"] for v in cmp.values())
