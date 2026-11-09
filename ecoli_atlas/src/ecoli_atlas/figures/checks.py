"""Compare regenerated numbers with the reference manifest of the bundle."""
from __future__ import annotations

TOLERANCES = {
    "test_n": 0, "test_auroc": 1e-3, "test_balanced_acc": 0.01, "test_ba_ci": 0.02,
    "oof_n": 0, "oof_auroc": 0.03, "oof_balanced_acc": 0.03,
    "atlas_embedding_max_abs_diff": 1e-3,   # the shipped atlas was embedded on a GPU; CPU float32 differs by ~3e-4
}


def _check(value, reference, tol):
    if value is None or reference is None:
        return {"value": value, "reference": reference, "tolerance": tol, "pass": None}
    return {"value": value, "reference": reference, "tolerance": tol, "pass": bool(abs(value - reference) <= tol)}


def _in_sweep(value, reference, spread: dict | None):
    """Out-of-fold refits vary with the seed; when a seed sweep's CV spread is known, a value inside
    its [min, max] counts as reproduced."""
    if spread is None or value is None or not spread.get("n"):
        return None
    lo, hi = spread["min"], spread["max"]
    return {"value": value, "reference": reference, "tolerance": f"seed sweep range [{lo:.3f}, {hi:.3f}] (n={spread['n']})",
            "pass": bool(lo <= value <= hi)}


def compare_to_manifest(metrics: dict, manifest: dict | None, tolerances: dict | None = None,
                        sweep: dict | None = None) -> dict:
    """metrics: {"test_metrics": {f: {...}}, "threshold_diagnostics": {f: {n, balanced_acc, auroc}},
    "atlas_embedding_max_abs_diff": float, "leakage": {"rates": {...}}} (all optional).
    ``sweep``: a ``seed_sweep`` block of a sweep manifest; its ``cv_spread`` replaces the fixed
    tolerance for the out-of-fold metrics."""
    tol = dict(TOLERANCES)
    tol.update(tolerances or {})
    manifest = manifest or {}
    cv_spread = (sweep or {}).get("cv_spread", {})
    checks = {}
    ref_test = manifest.get("test_metrics", {})
    for f, m in (metrics.get("test_metrics") or {}).items():
        r = ref_test.get(f)
        if not r:
            continue
        checks[f"test.{f}.n"] = _check(m.get("n"), r.get("n"), tol["test_n"])
        checks[f"test.{f}.auroc"] = _check(m.get("auroc"), r.get("auroc"), tol["test_auroc"])
        checks[f"test.{f}.balanced_acc"] = _check(m.get("balanced_acc"), r.get("balanced_acc"), tol["test_balanced_acc"])
        if m.get("ba_ci") and r.get("ba_ci"):
            checks[f"test.{f}.ba_ci_lo"] = _check(m["ba_ci"][0], r["ba_ci"][0], tol["test_ba_ci"])
            checks[f"test.{f}.ba_ci_hi"] = _check(m["ba_ci"][1], r["ba_ci"][1], tol["test_ba_ci"])
    ref_oof = manifest.get("threshold_diagnostics", {})
    for f, m in (metrics.get("threshold_diagnostics") or {}).items():
        r = ref_oof.get(f)
        if not r:
            continue
        checks[f"oof.{f}.n"] = _check(m.get("n"), r.get("n"), tol["oof_n"])
        checks[f"oof.{f}.auroc"] = (_in_sweep(m.get("auroc_refit"), r.get("auroc"), cv_spread.get(f, {}).get("auroc"))
                                    or _check(m.get("auroc_refit"), r.get("auroc"), tol["oof_auroc"]))
        checks[f"oof.{f}.balanced_acc"] = (_in_sweep(m.get("balanced_acc_refit"), r.get("balanced_acc"), cv_spread.get(f, {}).get("balanced_acc"))
                                           or _check(m.get("balanced_acc_refit"), r.get("balanced_acc"), tol["oof_balanced_acc"]))
    if metrics.get("atlas_embedding_max_abs_diff") is not None:
        checks["atlas_embedding_max_abs_diff"] = _check(metrics["atlas_embedding_max_abs_diff"], 0.0,
                                                        tol["atlas_embedding_max_abs_diff"])
    rates = (metrics.get("leakage") or {}).get("rates") or {}
    if "top_ae" in rates and "input_zcpm" in rates:
        checks["leakage.top_ae_below_input"] = {"value": rates["top_ae"]["mean"], "reference": rates["input_zcpm"]["mean"],
                                                "tolerance": "ordinal", "pass": bool(rates["top_ae"]["mean"] < rates["input_zcpm"]["mean"])}
    return checks


def all_pass(checks: dict) -> bool:
    return all(c["pass"] is not False for c in checks.values())
