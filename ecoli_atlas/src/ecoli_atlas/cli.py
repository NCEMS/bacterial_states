"""Command-line entry point: `ecoli_atlas <subcommand>`."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .utils import pick_device, setup_logging


def _info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _write_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


# ---------------------------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    from .prepare import prepare
    counts, _ = prepare(args.ingest_results, args.counts, args.metadata, args.out)
    _info(f"wrote {Path(args.out) / 'counts.tsv'} ({counts.shape[0]} features x {counts.shape[1]} samples) and samples.tsv")
    if args.models:
        from .artifacts import ModelBundle
        from .preprocess import align_counts
        bundle = ModelBundle.load(args.models)
        cov = align_counts(counts, bundle.fs)[2]
        cov.to_frame(bundle.fs.ids).to_csv(Path(args.out) / "feature_coverage.tsv", sep="\t")
        _info(f"model features present: {cov.n_present}/{cov.n_model_features} ({cov.fraction:.1%})")
    return 0


def _training_inputs(args, cfg):
    """Tables, feature set and splits for `train` / `evaluate` / `figures`."""
    from .data import load_tables
    from .features import FeatureSet, read_feature_table
    from .labels import head_labels, is_reference
    from .splits import check_no_leakage, load_splits, make_splits
    tables = load_tables(args.counts, args.metadata, args.qc, getattr(args, "model_name", None))
    fs = FeatureSet.from_table(read_feature_table(args.features))
    if getattr(args, "splits", None):
        splits = load_splits(args.splits)
        missing = sorted(set(tables.ids) - set(splits.index))
        if missing:
            raise SystemExit(f"{args.splits}: no split for {len(missing)} model samples ({missing[:5]})")
        splits = splits.loc[tables.ids]
        check_no_leakage(splits, cfg)
    else:
        splits = make_splits(tables.meta["GSE"], head_labels(tables.meta), is_reference(tables.meta), cfg, log=_info)
    return tables, fs, splits


def _fit_and_save(tables, fs, cfg, splits, device, seed: int, out: Path, cv: bool) -> dict:
    from .artifacts import save_bundle
    from .data import build_training
    from .evaluate import evaluate
    from .train import cross_validate, fit
    data = build_training(tables, fs, cfg, splits)
    _info(f"[seed {seed}] training pool {len(data.tr_rows)} samples, TEST {len(data.te_rows)}, "
          f"{fs.n} features, {data.C.shape[1]} covariates, study cap {data.study_cap}")
    n_missing = data.M.sum(dim=1).numpy()
    if cfg.mask_missing:
        _info(f"[seed {seed}] genes called missing: median {int(np.median(n_missing))} per sample, "
              f"{int((n_missing >= 100).sum())} samples with 100 or more")
    model, history = fit(data, cfg, device, seed=seed, log=_info)
    ev = evaluate(model, data, cfg, device)
    extra = {"phase_order": ev["report"]["phase_order"]}
    if cv:
        cvr = cross_validate(tables, fs, cfg, splits, device, seed=seed, log=_info)
        extra |= {"cv_metrics": cvr["metrics"], "cv_phase_order": cvr["phase_order"]}
    manifest = save_bundle(out, model, data, cfg, device, seed, metrics=ev["report"]["heads"].get("test", {}), extra=extra)
    _write_json(ev["report"] | extra, out / "evaluation.json")
    ev["embedding"].to_csv(out / "embedding.tsv", sep="\t", float_format="%.6g")
    ev["predictions"].assign(n_missing_genes=n_missing).to_csv(out / "predictions.tsv", sep="\t", float_format="%.6g")
    ev["phase_lfc"].to_csv(out / "phase_lfc.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(history).to_csv(out / "history.tsv", sep="\t", index=False, float_format="%.6g")
    if cv:
        cvr["predictions"].to_csv(out / "cv_predictions.tsv", sep="\t", float_format="%.6g")
    for h, m in manifest["test_metrics"].items():
        _info(f"[seed {seed}] TEST {h}: balanced acc {m['balanced_acc']:.3f}, AUROC {m['auroc']:.3f} (n={m['n']})")
    return manifest


def _mean_auroc(block: dict) -> float:
    vals = [v["auroc"] for v in (block or {}).values() if v.get("auroc") is not None and not np.isnan(v["auroc"])]
    return float(np.mean(vals)) if vals else float("nan")


def cmd_train(args) -> int:
    import torch
    from .config import TrainConfig
    from .splits import SPLIT_COLUMNS
    cfg = TrainConfig(seed=args.seed, fit=args.fit.replace("-", "_"), phase_head=args.phase_head)
    for name in ("epochs", "study_cap", "label_cap", "test_frac"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    cfg.mask_missing = not args.no_mask_missing
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables, fs, splits = _training_inputs(args, cfg)
    if args.split_only:
        splits[SPLIT_COLUMNS].to_csv(out / "splits.csv")
        from .splits import bucket_labels
        _info(f"wrote {out / 'splits.csv'}: " + ", ".join(f"{k} {v}" for k, v in bucket_labels(splits).value_counts().sort_index().items()))
        return 0
    device = pick_device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [cfg.seed]
    if len(seeds) == 1:
        _fit_and_save(tables, fs, cfg, splits, device, seeds[0], out, args.cv)
        _info(f"wrote the model bundle and its evaluation to {out}")
        return 0
    # several seeds on one split: each in seed_<s>/, one of them copied to the output directory
    rows, manifests = [], {}
    for s in seeds:
        manifests[s] = _fit_and_save(tables, fs, cfg, splits, device, s, out / f"seed_{s}", args.cv)
        for h, m in manifests[s]["test_metrics"].items():
            cvm = (manifests[s].get("cv_metrics") or {}).get(h, {})
            rows.append({"seed": s, "head": h, "test_balanced_acc": m["balanced_acc"], "test_auroc": m["auroc"],
                         "cv_balanced_acc": cvm.get("balanced_acc"), "cv_auroc": cvm.get("auroc")}
                        | ({"test_order": manifests[s]["phase_order"]["test"]["share_in_order_mean_over_studies"],
                            "cv_order": (manifests[s].get("cv_phase_order") or {}).get("share_in_order_mean_over_studies")}
                           if h == "phase" else {}))
    # ranked on the out-of-fold scores, never on TEST; without --cv the first seed is taken
    chosen = max(seeds, key=lambda s: _mean_auroc(manifests[s].get("cv_metrics"))) if args.cv else seeds[0]
    table = pd.DataFrame(rows)
    table["selected"] = table["seed"] == chosen
    table.to_csv(out / "seed_sweep.tsv", sep="\t", index=False, float_format="%.4f")
    for f in (out / f"seed_{chosen}").iterdir():
        shutil.copy2(f, out / f.name)
    _info(f"{len(seeds)} seeds; seed {chosen} copied to {out} "
          + ("(best mean out-of-fold AUROC)" if args.cv else "(the first; give --cv to rank the seeds)"))
    return 0


def cmd_evaluate(args) -> int:
    from .artifacts import SPLITS_FILE, ModelBundle
    from .data import build_training
    from .evaluate import evaluate
    device = pick_device(args.device)
    bundle = ModelBundle.load(args.models, device)
    cfg = bundle.config
    args.splits = str(Path(args.models) / SPLITS_FILE)
    tables, _, splits = _training_inputs(args, cfg)
    data = build_training(tables, bundle.fs, cfg, splits)
    ev = evaluate(bundle.model, data, cfg, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(ev["report"], out / "evaluation.json")
    ev["embedding"].to_csv(out / "embedding.tsv", sep="\t", float_format="%.6g")
    ev["predictions"].to_csv(out / "predictions.tsv", sep="\t", float_format="%.6g")
    ev["phase_lfc"].to_csv(out / "phase_lfc.tsv", sep="\t", index=False, float_format="%.4f")
    _info(f"wrote evaluation.json, embedding.tsv, predictions.tsv and phase_lfc.tsv to {out}")
    return 0


def cmd_drivers(args) -> int:
    from .artifacts import MANIFEST_FILE, SPLITS_FILE, ModelBundle
    from .config import HEADS
    from .data import build_training
    from .drivers import combine_seeds, default_contrasts, expression_units, gene_drivers, unit_table
    from .features import read_feature_table
    from .figures.drivers import fig_drivers
    from .figures.plotting import apply_style
    if args.head not in HEADS:
        _info(f"error: --head must be one of {list(HEADS)}")
        return 2
    device = pick_device(args.device)
    root = Path(args.models)
    seed_dirs = sorted((d for d in root.glob("seed_*") if (d / MANIFEST_FILE).exists()), key=lambda d: int(d.name.split("_")[1]))
    paths = seed_dirs if (seed_dirs and not args.one_model) else [root]
    bundles = [ModelBundle.load(p, device) for p in paths]
    cfg = bundles[0].config
    if any((b.config.fit, b.config.phase_head) != (cfg.fit, cfg.phase_head) for b in bundles):
        _info("error: the bundles differ in fit or phase head")
        return 2
    args.splits = str(paths[0] / SPLITS_FILE)
    tables, _, splits = _training_inputs(args, cfg)
    data = build_training(tables, bundles[0].fs, cfg, splits)           # the same for every seed of a sweep: one split
    classes = data.fit_classes[args.head]
    _info(f"{len(bundles)} model(s) from {root}" + (" (the seeds of the sweep)" if len(bundles) > 1 else ""))
    try:
        contrasts = [tuple(c.split(":")) for c in args.contrast] if args.contrast else default_contrasts(classes)
        if any(len(c) != 2 for c in contrasts):
            raise ValueError("--contrast takes LOW:HIGH")
        feat = read_feature_table(args.features)
        logx = data.X.numpy() * data.zscore.sd + data.zscore.mu
        unit = expression_units(feat, data.fs.ids, logx[data.tr_rows], data.splits["gse"].to_numpy()[data.tr_rows], min_r=args.unit_r)
        n_units = len(np.unique(unit))
        _info(f"{data.fs.n} features in {n_units} expression units ({int((np.bincount(np.unique(unit, return_inverse=True)[1]) > 1).sum())} with more than one gene)")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        formats = tuple(f for f in args.formats.split(",") if f)
        apply_style()
        summaries = []
        for low, high in contrasts:
            per_gene, per_unit, per_summary = [], [], []
            for b in bundles:
                table, summary = gene_drivers(b.model, data, args.head, low, high, device, steps=args.steps)
                for col, name in (("Symbol", "symbol"), ("Type", "type"), ("Product", "product")):
                    if col in feat.columns:
                        table[name] = feat[col].reindex(table["feature"]).fillna("").to_numpy()
                per_gene.append(table)
                per_unit.append(unit_table(table, unit, data.fs.ids))
                per_summary.append(summary)
            genes = combine_seeds(per_gene, "feature", args.top_genes) if len(bundles) > 1 else per_gene[0]
            # a unit is identified by its first gene; its lead gene may differ between seeds
            units = combine_seeds(per_unit, "unit_id", args.top_units) if len(bundles) > 1 else per_unit[0]
            if len(bundles) > 1:
                units = units[["unit"] + [c for c in units.columns if c != "unit"]]
            name = f"{args.head}_{low}_to_{high}"
            genes.to_csv(out / f"drivers_{name}.tsv", sep="\t", index=False, float_format="%.5g")
            units.to_csv(out / f"units_{name}.tsv", sep="\t", index=False, float_format="%.5g")
            fig_drivers(genes, args.head, low, high, out, formats, top=args.top)
            fig_drivers(units, args.head, low, high, out, formats, top=args.top, label="unit", name="units")
            summary = dict(per_summary[0])
            for k in ("score_gap_high_minus_baseline", "sum_of_attributions"):
                summary[k] = float(np.mean([s[k] for s in per_summary]))
            pos = units["attribution"] > 0
            robust = (units["seeds_in_top"] >= 0.75 * len(bundles)) if len(bundles) > 1 else pd.Series(True, index=units.index)
            summary |= {"n_models": len(bundles), "n_units": int(n_units),
                        "share_of_attribution": {f"top_{k}_units": float(units["share"].head(k).sum()) for k in (10, 50, 200)},
                        "share_of_attribution_towards_high": float(units.loc[pos, "share"].sum()),
                        "towards_high_up_vs_down": {d: float(units.loc[pos & (units["direction"] == d), "share"].sum()) for d in ("up", "down")},
                        "top_units_up": list(units.loc[pos & robust & (units["direction"] == "up"), "unit"].head(args.top)),
                        "top_units_down": list(units.loc[pos & robust & (units["direction"] == "down"), "unit"].head(args.top)),
                        "top_units_against": list(units.loc[~pos & robust, "unit"].head(args.top))}
            if len(bundles) > 1:
                summary["units_in_top_of_three_quarters_of_seeds"] = int(robust.sum())
                by_seed = pd.concat([u.set_index("unit_id")["attribution"] for u in per_unit], axis=1).corr(method="spearman").to_numpy()
                summary["rank_correlation_between_seeds"] = float(by_seed[np.triu_indices(len(per_unit), 1)].mean())
                by_seed = pd.concat([g.set_index("feature")["attribution"] for g in per_gene], axis=1).corr(method="spearman").to_numpy()
                summary["rank_correlation_between_seeds_genes"] = float(by_seed[np.triu_indices(len(per_gene), 1)].mean())
            summaries.append(summary)
            _info(f"{name}: {summary['n_samples'][high]} {high} samples in {summary['n_studies'][high]} studies; "
                  f"top 50 units carry {summary['share_of_attribution']['top_50_units']:.0%} of the attribution"
                  + (f"; {int(robust.sum())} units are among the top {args.top_units} in at least 3 of 4 seeds, "
                     f"rank correlation between seeds {summary['rank_correlation_between_seeds']:.2f}" if len(bundles) > 1 else ""))
    except ValueError as e:
        _info(f"error: {e}")
        return 2
    _write_json({"models": [str(p) for p in paths], "phase_head": cfg.phase_head, "fit": cfg.fit, "classes": list(classes),
                 "unit_min_correlation": args.unit_r, "contrasts": summaries}, out / f"drivers_{args.head}.json")
    _info(f"wrote the driver tables and figures to {out}")
    return 0


def cmd_embed(args) -> int:
    from .artifacts import ModelBundle
    from .data import read_table
    from .embed import embed_counts
    from .io import read_counts
    device = pick_device(args.device)
    bundle = ModelBundle.load(args.models, device)
    try:
        res = embed_counts(bundle, read_counts(args.counts), read_table(args.qc) if args.qc else None, device,
                           args.min_feature_coverage)
    except ValueError as exc:
        _info(f"error: {exc}")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["embedding"].to_csv(out / "embedding.tsv", sep="\t", float_format="%.6g")
    res["predictions"].to_csv(out / "predictions.tsv", sep="\t", float_format="%.6g")
    res["samples"].to_csv(out / "samples.tsv", sep="\t", float_format="%.6g")
    res["coverage"].to_frame(bundle.fs.ids).to_csv(out / "feature_coverage.tsv", sep="\t")
    _info(f"embedded {len(res['embedding'])} samples ({res['coverage'].fraction:.1%} of the model features present); "
          f"wrote embedding.tsv, predictions.tsv, samples.tsv and feature_coverage.tsv to {out}"
          + ("" if args.qc else "; no --qc table: the QC covariates were set to the training means"))
    return 0


def _figure_context(args):
    """The training tables put through the bundle in --models: (context, tables, splits)."""
    from .artifacts import SPLITS_FILE, ModelBundle
    from .data import build_training
    from .features import read_feature_table
    from .figures.context import FigureContext
    if getattr(args, "threads", None):
        import torch
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    bundle = ModelBundle.load(args.models, device)
    cfg = bundle.config
    args.splits = str(Path(args.models) / SPLITS_FILE)
    tables, _, splits = _training_inputs(args, cfg)
    data = build_training(tables, bundle.fs, cfg, splits)
    return FigureContext.build(bundle, data, device, read_feature_table(args.features)["Symbol"], log=_info), tables, splits


def _model_figures(args, names: list, out: Path, formats: tuple) -> tuple[dict, bool]:
    """Draw the figures that need the model. Returns (metrics, all checks passed)."""
    from .figures.checks import all_agree, compare_metrics
    from .figures.gene_correlates import fig_factor_gene_correlates
    from .figures.leakage import fig_study_leakage
    from .figures.pca import fig_batch_genes, fig_pca_batch
    from .figures.test_eval import fig_factor_test_eval, misclassified_table
    from .figures.thresholds import CV_FILE, cv_metrics, fig_threshold_curves, load_cv_predictions
    from .figures.umaps import fig_factor_umaps
    ctx, tables, splits = _figure_context(args)
    manifest = ctx.bundle.manifest
    metrics = {"models": str(args.models), "embedding_check": ctx.embedding_check(args.models)}
    ok = metrics["embedding_check"].get("max_abs_diff", 0.0) <= 1e-4
    _info(f"model figures for {len(ctx.ids)} samples; embedding against the bundle's: {metrics['embedding_check']}")

    def test_eval():
        res = fig_factor_test_eval(ctx, out, formats)
        metrics["test_metrics_vs_manifest"] = compare_metrics(res, manifest.get("test_metrics"))
        metrics["phase_misclassified"] = misclassified_table(ctx, out)
        return res

    def thresholds():
        cv, src = load_cv_predictions(out, args.models)
        if cv is None:
            from .train import cross_validate
            _info("no cv_predictions.tsv in the bundle or the output directory: fitting the study folds once")
            cv = cross_validate(tables, ctx.bundle.fs, ctx.cfg, splits, ctx.device, seed=manifest["seed"], log=_info)["predictions"]
            src = out / CV_FILE
            cv.to_csv(src, sep="\t", float_format="%.6g")
            cv, _ = load_cv_predictions(out)
        metrics["cv_predictions"] = str(src)
        metrics["cv_metrics_vs_manifest"] = compare_metrics(cv_metrics(cv), manifest.get("cv_metrics"))
        return fig_threshold_curves(cv, out, formats)

    draw = {"pca_batch": lambda: fig_pca_batch(ctx, out, formats), "batch_genes": lambda: fig_batch_genes(ctx, out, formats),
            "factor_test_eval": test_eval, "factor_umaps": lambda: fig_factor_umaps(ctx, out, formats),
            "factor_gene_correlates": lambda: fig_factor_gene_correlates(ctx, out, formats),
            "study_leakage": lambda: fig_study_leakage(ctx, out, formats), "threshold_curves": thresholds}
    for name in names:
        try:
            metrics[name] = draw[name]()
            _info(f"  {name}: done")
        except ValueError as e:
            metrics[name], ok = {"error": str(e)}, False
            _info(f"  {name}: error: {e}")
    if "test_metrics_vs_manifest" in metrics and not all_agree(metrics["test_metrics_vs_manifest"]):
        ok = False
        _info(f"TEST metrics differ from the manifest: {metrics['test_metrics_vs_manifest']}")
    return metrics, ok


def cmd_figures(args) -> int:
    from .data import read_table
    from .figures import DATA_FIGURES, MODEL_FIGURES
    from .figures.plotting import apply_style
    from .labels import head_labels
    from .splits import load_splits
    names = args.only.split(",") if args.only else list(DATA_FIGURES) + (list(MODEL_FIGURES) if args.models else [])
    unknown = sorted(set(names) - set(DATA_FIGURES) - set(MODEL_FIGURES))
    if unknown:
        _info(f"error: unknown figure(s) {unknown}; available: {sorted(DATA_FIGURES) + list(MODEL_FIGURES)}")
        return 2
    model_names = [n for n in names if n in MODEL_FIGURES]
    if model_names and not (args.models and args.counts and args.qc and args.features):
        _info(f"error: {model_names} need --models, --counts, --qc and --features")
        return 2
    if not (args.splits or args.models):
        _info("error: give --splits (a splits.csv or a bundle directory) or --models")
        return 2
    splits_path = Path(args.splits or args.models)
    splits = load_splits(splits_path / "splits.csv" if splits_path.is_dir() else splits_path)
    meta = read_table(args.metadata)
    missing = sorted(set(splits.index) - set(meta.index))
    if missing:
        _info(f"error: {len(missing)} samples of the split are not in the metadata ({missing[:5]})")
        return 1
    meta = meta.loc[splits.index]
    labels = head_labels(meta)
    apply_style()
    out, formats = Path(args.out), tuple(args.formats.split(","))
    out.mkdir(parents=True, exist_ok=True)
    if "split_balance" in names:
        DATA_FIGURES["split_balance"](labels, splits, out, formats).to_csv(out / "split_balance.tsv", sep="\t", index=False,
                                                                          float_format="%.4f")
    if "corpus_composition" in names:
        _write_json(DATA_FIGURES["corpus_composition"](meta.join(labels), out, formats), out / "corpus_composition.json")
    ok = True
    if model_names:
        metrics, ok = _model_figures(args, model_names, out, formats)
        path = out / "figure_metrics.json"          # a run of some figures (--only) keeps the entries of the others
        earlier = json.loads(path.read_text()) if path.exists() else {}
        _write_json((earlier if earlier.get("models") == metrics["models"] else {}) | metrics, path)
    _info(f"wrote {', '.join(names)} to {out}" + ("" if ok else "; a figure failed or a check against the bundle did not pass (figure_metrics.json)"))
    return 0 if ok or not args.strict else 1


def cmd_casestudy(args) -> int:
    from .casestudy import CASE_STUDIES
    from .casestudy.glu_cas import run_glu_cas
    from .casestudy.oxygen import run_oxygen
    from .casestudy.uti import run_uti
    from .figures.plotting import apply_style
    ctx, _, _ = _figure_context(args)
    apply_style()
    out, formats = Path(args.out), tuple(args.formats.split(","))
    out.mkdir(parents=True, exist_ok=True)
    run = {"uti": lambda: run_uti(ctx, out, formats, n_perm=args.n_perm),
           "oxygen": lambda: run_oxygen(ctx, out, formats, studies=args.oxygen_studies.split(",")),
           "glu-cas": lambda: run_glu_cas(ctx, out, formats, n_perm=args.n_perm)}
    path = out / "casestudy_metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    failed = False
    for name in (CASE_STUDIES if args.study == "all" else (args.study,)):
        try:
            metrics[name] = run[name]() | {"models": str(args.models)}
        except ValueError as e:
            metrics[name], failed = {"error": str(e), "models": str(args.models)}, True
            _info(f"[{name}] error: {e}")
    _write_json(metrics, path)
    _info(f"wrote the case-study figures and tables to {out}")
    return 1 if failed else 0


def cmd_axes(args) -> int:
    from .axes import AxisSpec, axis_scores, category_frame, load_embedding, summarize
    from .data import read_table
    from .figures.axes import fig_axes_map, fig_axis_distributions, fig_axis_series
    from .figures.plotting import apply_style
    try:
        specs = [AxisSpec.parse(a, args.holdout or ()) for a in args.axis]
        emb, splits, _ = load_embedding(args.models, args.blocks)
    except (ValueError, FileNotFoundError) as e:
        _info(f"error: {e}")
        return 1
    meta = read_table(args.metadata)
    missing = sorted(set(emb.index) - set(meta.index))
    if missing:
        _info(f"error: {len(missing)} samples of the embedding are not in the metadata ({missing[:5]})")
        return 1
    meta = meta.loc[emb.index]
    where: dict = {}
    for w in args.where or ():
        col, _, values = w.partition("=")
        where[col.strip()] = [v.strip() for v in values.split(",")]
    series = None
    if args.series:
        series = pd.read_csv(args.series, sep="\t", dtype={"sample": str, "series": str})
        lacking = {"sample", "series", "x"} - set(series.columns)
        if lacking:
            _info(f"error: {args.series} lacks columns {sorted(lacking)}")
            return 1
    out, formats = Path(args.out), tuple(args.formats.split(","))
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    done = {}
    for spec in specs:
        try:
            frame = category_frame(meta, splits, spec, where)
            res = axis_scores(emb.to_numpy(dtype=np.float64), frame, spec, n_folds=args.folds, seed=args.seed, l2=args.l2)
        except ValueError as e:
            _info(f"error: {e}")
            return 1
        scores = res["scores"]
        scores.index.name = "sample"
        scores.to_csv(out / f"axis_{spec.name}_scores.tsv", sep="\t", float_format="%.5g")
        summary = summarize(scores, spec) | {"column": spec.column, "blocks": args.blocks, "where": where, "thresholds": res["thresholds"],
                                             "weights": res["weights"], "embedding_columns": list(emb.columns)}
        _write_json(summary, out / f"axis_{spec.name}_summary.json")
        fig_axis_distributions(scores, spec, res["thresholds"], out, formats)
        if series is not None:
            fig_axis_series(scores, series, spec, out, formats)
        done[spec.name] = (scores, spec)
        m = summary["methods"]["ordinal"]["all"]
        ws = m["within_study"]
        _info(f"axis {spec.name}: {m['n']} samples with a category; ordinal score: Spearman {m['spearman']:.3f}; within-study pairs "
              f"in order: {100 * (ws['share_in_order_mean_over_studies'] or float('nan')):.1f}% (mean of {ws['n_studies']} studies)")
    names = list(done)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fig_axes_map(done[names[i]][0], done[names[i]][1], done[names[j]][0], done[names[j]][1], out, formats)
    _info(f"wrote the axis tables and figures to {out}")
    return 0


# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from .casestudy import CASE_STUDIES
    from .casestudy.oxygen import OXYGEN_CASE_STUDIES
    from .figures import MODEL_FIGURES
    p = argparse.ArgumentParser(prog="ecoli_atlas",
                                description="Ecoli_ATLAS: embed E. coli RNA-seq samples with the atlas model.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"ecoli_atlas {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_device(sp):
        sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    def add_tables(sp):
        sp.add_argument("--counts", required=True, help="joined count table of the atlas (features x samples)")
        sp.add_argument("--metadata", required=True, help="sample metadata (metadata_v2.tsv)")
        sp.add_argument("--qc", required=True, help="QC table (sample_qc.tsv); its in_model column selects the samples")
        sp.add_argument("--features", required=True, help="feature table of the counting annotation (*_counting_v2.features.tsv)")
        sp.add_argument("--model-name", help="use only the samples whose in_model has this value (default: any)")

    sp = sub.add_parser("prepare", help="convert Ecoli_inGEST results into a count table for `embed`")
    sp.add_argument("--ingest-results", metavar="DIR", help="an inGEST results directory")
    sp.add_argument("--counts", metavar="FILE", help="a count table instead of --ingest-results")
    sp.add_argument("--metadata", metavar="TSV", help="optional sample metadata (first column = sample id)")
    sp.add_argument("--models", metavar="DIR", help="model bundle; if given, also write feature_coverage.tsv")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train", help="train the atlas model and evaluate it on the held-out studies")
    add_tables(sp)
    sp.add_argument("--out", required=True, help="output directory for the bundle")
    sp.add_argument("--splits", help="reuse a splits.csv instead of making the split")
    sp.add_argument("--split-only", action="store_true", help="make the split, write splits.csv and stop")
    sp.add_argument("--seed", type=int, default=77, help="seed of the split and, without --seeds, of the model")
    sp.add_argument("--seeds", help="comma-separated model seeds, all trained on the same split (each in seed_<s>/)")
    sp.add_argument("--cv", action="store_true", help="also fit the study folds for out-of-fold metrics (ranks --seeds)")
    sp.add_argument("--fit", default="two-stage", choices=["two-stage", "one-pass", "unstructured"],
                    help="two-stage: block encoders first, then residual and decoder with the blocks frozen (default); "
                         "one-pass: the same blocks, all fitted together; unstructured: one embedding without blocks")
    sp.add_argument("--phase-head", default="binary", choices=["binary", "ordinal"],
                    help="binary: exponential against stationary (default); ordinal: one score over "
                         "exponential < transition < stationary, the call being stationary against the rest")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--study-cap", type=int, help="samples of one study used per epoch (default: size of the second-largest study)")
    sp.add_argument("--label-cap", type=int, help="samples of one class that one study may count for in a call's loss "
                                                  "(default: for every class the size of the second-largest study in it)")
    sp.add_argument("--test-frac", type=float)
    sp.add_argument("--no-mask-missing", action="store_true", help="treat genes a strain does not have as measured zeros "
                    "(default: they are called from the counts and left out of the fit)")
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_device(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="evaluate a trained bundle again on the training tables")
    add_tables(sp)
    sp.add_argument("--models", required=True, help="model bundle")
    sp.add_argument("--out", required=True)
    add_device(sp)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("drivers", help="the genes that move a head's call (integrated gradients), per pair of its classes")
    add_tables(sp)
    sp.add_argument("--models", required=True, help="model bundle; when it holds the seed_<n> bundles of a sweep, all of them are used")
    sp.add_argument("--head", default="phase", help="phase (default), medium or lab_strain")
    sp.add_argument("--contrast", action="append", metavar="LOW:HIGH",
                    help="a pair of the head's classes; repeatable (default: first against last class, and for the "
                         "ordinal phase head also exponential:transition and transition:stationary)")
    sp.add_argument("--one-model", action="store_true", help="use only the bundle in --models, not the seed_<n> bundles beside it")
    sp.add_argument("--top", type=int, default=25, help="rows per direction in the figures and the summary")
    sp.add_argument("--top-genes", type=int, default=200, help="a gene counts for seeds_in_top when among this many genes of a seed")
    sp.add_argument("--top-units", type=int, default=100, help="the same for expression units")
    sp.add_argument("--unit-r", type=float, default=0.6, help="within-study correlation at which neighbouring genes are joined into a unit")
    sp.add_argument("--steps", type=int, default=32, help="points along the path of the integrated gradients")
    sp.add_argument("--out", required=True)
    sp.add_argument("--formats", default="pdf,png")
    add_device(sp)
    sp.set_defaults(func=cmd_drivers)

    sp = sub.add_parser("embed", help="embed new samples with a trained bundle")
    sp.add_argument("--counts", required=True, help="inGEST count table (or the counts.tsv of `prepare`)")
    sp.add_argument("--models", required=True, help="model bundle")
    sp.add_argument("--qc", help="QC table of the samples (inGEST sample_qc.tsv layout); optional")
    sp.add_argument("--min-feature-coverage", type=float, default=0.9)
    sp.add_argument("--out", required=True)
    add_device(sp)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("axes", help="place the samples on ordered axes of the embedding (growth phase, medium, ...)")
    sp.add_argument("--models", required=True, help="bundle directory (embedding.tsv, manifest.json, splits.csv)")
    sp.add_argument("--metadata", required=True)
    sp.add_argument("--axis", action="append", required=True, metavar="NAME:COLUMN:CAT1,CAT2,...",
                    help="an axis: a metadata column and its values in order; may be given several times")
    sp.add_argument("--holdout", action="append", metavar="CATEGORY",
                    help="a category left out of the fits and only scored (extrapolation or interpolation test); repeatable")
    sp.add_argument("--where", action="append", metavar="COLUMN=VALUE[,VALUE]",
                    help="only samples with these metadata values get a category; repeatable")
    sp.add_argument("--blocks", default="all", help="embedding blocks to use, comma-separated (default: the whole embedding)")
    sp.add_argument("--series", help="table (sample, series, x[, x_label]) of ordered series, e.g. time courses, drawn as trajectories")
    sp.add_argument("--folds", type=int, default=5, help="study folds of the out-of-fold scores")
    sp.add_argument("--l2", type=float, default=1.0, help="ridge penalty of the ordinal model")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True)
    sp.add_argument("--formats", default="pdf,png")
    sp.set_defaults(func=cmd_axes)

    sp = sub.add_parser("figures", help="figures of the training data (split_balance, corpus_composition) and, with --models, "
                                        "of the model: " + ", ".join(MODEL_FIGURES))
    sp.add_argument("--metadata", required=True, help="sample metadata (metadata_v2.tsv)")
    sp.add_argument("--splits", help="splits.csv, or a bundle directory that holds one (default: --models)")
    sp.add_argument("--models", help="model bundle; with it and the three tables below the model figures are drawn too")
    sp.add_argument("--counts", help="joined count table of the atlas")
    sp.add_argument("--qc", help="QC table (sample_qc.tsv)")
    sp.add_argument("--features", help="feature table of the counting annotation")
    sp.add_argument("--model-name", help="use only the samples whose in_model has this value (default: any)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--only", help="comma-separated figure names")
    sp.add_argument("--formats", default="pdf,png", help="comma-separated figure formats (default pdf,png)")
    sp.add_argument("--strict", action="store_true", help="exit 1 when a figure fails or the embedding or the TEST metrics "
                                                          "differ from those stored in the bundle")
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_device(sp)
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("casestudy", help="case studies on the embedding: uti, oxygen, glu-cas")
    sp.add_argument("study", choices=list(CASE_STUDIES) + ["all"])
    add_tables(sp)
    sp.add_argument("--models", required=True, help="model bundle")
    sp.add_argument("--oxygen-studies", default=",".join(OXYGEN_CASE_STUDIES), help="comma-separated studies of the oxygen case study")
    sp.add_argument("--n-perm", type=int, default=1000, help="permutations of the paired tests")
    sp.add_argument("--out", required=True)
    sp.add_argument("--formats", default="pdf,png")
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_device(sp)
    sp.set_defaults(func=cmd_casestudy)
    return p


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
