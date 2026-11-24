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
def cmd_download(args) -> int:
    from . import download as dl
    rc = 0
    if "modules" in args.bundles:
        rc = _download_modules(args)
        args.bundles = [b for b in args.bundles if b != "modules"]
        if not args.bundles:
            return rc
    bundles = list(dl.BUNDLES) if "all" in args.bundles else args.bundles
    for b in bundles:
        rc = max(rc, dl.download(b, Path(args.dest) if args.dest else None, dry_run=args.dry_run))
    return rc


def _download_modules(args) -> int:
    """`download modules`: fetch the module sources and build modules.tsv (gene symbols of the counting annotation)."""
    import datetime
    from . import download as dl
    from .features import read_feature_table
    from .modules import MODULES_FILE, build_module_table, write_module_table
    dest = Path(args.dest) if args.dest else dl.default_data_dir() / "modules"
    features = Path(args.features) if args.features else (dl.data_root() / "resources" / "annotation"
                                                          / "GCF_000005845.2_ASM584v2_counting_v2.features.tsv")
    _info(f"module sources -> {dest}")
    for src, info in dl.MODULE_SOURCES.items():
        _info(f"  {src}: {info['description']}")
    if args.dry_run:
        return 0
    if not features.exists():
        _info(f"error: the counting feature table {features} is needed to map genes (give --features)")
        return 2
    try:
        record = dl.download_module_sources(dest / "sources", log=sys.stderr)
    except (OSError, ValueError) as e:
        _info(f"error: {e}")
        return 1
    table, stats = build_module_table(dest / "sources", read_feature_table(features))
    when = datetime.date.today().isoformat()
    header = [f"curated gene modules for ecoli_atlas drivers --modules; built {when} by `ecoli_atlas download modules`",
              f"precise1k (RegulonDB regulons, iModulons): SBRG/precise1k commit {dl.PRECISE1K_COMMIT}, MIT licence",
              f"KEGG pathways and modules: rest.kegg.jp, retrieved {record['kegg']['link_pathway_eco.txt']['retrieved']}; "
              "free for academic use, do not redistribute"]
    write_module_table(table, dest / MODULES_FILE, header)
    _write_json({"sources": record, "stats": stats, "features": str(features)}, dest / "modules_manifest.json")
    for src, st in stats.items():
        _info(f"  {src}: {st['modules']} modules, {st['members_matched']}/{st['members_listed']} members matched to the model's genes")
    _info(f"wrote {len(table)} modules to {dest / MODULES_FILE}")
    return 0


def cmd_prepare(args) -> int:
    from .prepare import prepare
    try:
        counts, _ = prepare(args.ingest_results, args.counts, args.metadata, args.out)
    except (ValueError, FileNotFoundError) as exc:
        _info(f"error: {exc}")
        return 1
    written = [n for n in ("counts.tsv", "antisense_counts.tsv", "qc.tsv", "samples.tsv") if (Path(args.out) / n).exists()]
    _info(f"wrote {', '.join(written)} to {args.out} ({counts.shape[0]} features x {counts.shape[1]} samples)")
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
    tables = load_tables(args.counts, args.metadata, args.qc, getattr(args, "model_name", None), getattr(args, "antisense", None),
                         getattr(args, "time_series", None))
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


def _fit_and_save(tables, fs, cfg, splits, device, seed: int, out: Path, cv: bool, missing_out: Path | None = None,
                  cache: dict | None = None) -> dict:
    """``missing_out``: where to write the missing-gene calls of the samples (they are the same for every seed).
    ``cache``: a dict shared by the seeds of a sweep, in which the training data is kept (``data.prepared``)."""
    from .artifacts import save_bundle
    from .data import prepared
    from .evaluate import evaluate
    from .train import cross_validate, fit
    data = prepared(cache, "all", tables, fs, cfg, splits)
    _info(f"[seed {seed}] training pool {len(data.tr_rows)} samples, TEST {len(data.te_rows)}, "
          f"{fs.n} features, {data.C.shape[1]} covariates, study cap {data.study_cap}")
    n_missing = data.M.sum(dim=1).numpy()
    if cfg.mask_missing:
        _info(f"[seed {seed}] genes called missing: median {int(np.median(n_missing))} per sample, "
              f"{int((n_missing >= 100).sum())} samples with 100 or more")
    if cfg.dna_decoder or cfg.dna_inputs:
        _info(f"[seed {seed}] genomic DNA: {int(data.extra.get('dna_samples', 0))} samples with an estimate above 0")
    order_term = None
    if cfg.phase_order_weight > 0:
        from .train import order_pair_sampler
        smp = order_pair_sampler(data, data.tr_rows)
        if smp is None:
            raise SystemExit("--phase-order-weight: no ordered pairs among the training samples")
        order_term = {"weight": cfg.phase_order_weight, "pairs": smp["n_pairs"], "studies": smp["n_studies"],
                      "time_series": bool(tables.time_series is not None)}
        _info(f"[seed {seed}] phase-order term, weight {cfg.phase_order_weight}: {smp['n_pairs']['levels']} pairs of phase levels in "
              f"{smp['n_studies']['levels']} studies, {smp['n_pairs']['series']} time pairs in {smp['n_studies']['series']} studies")
    if missing_out is not None:
        rows, cols = np.nonzero(data.M.numpy())
        pd.DataFrame({"sample": np.array(data.ids, dtype=object)[rows], "feature": np.array(fs.ids, dtype=object)[cols]}).to_csv(
            missing_out, sep="\t", index=False)
    model, history = fit(data, cfg, device, seed=seed, log=_info)
    ev = evaluate(model, data, cfg, device)
    extra = {"phase_order": ev["report"]["phase_order"]} | ({"series_order": ev["report"]["series_order"]} if "series_order" in ev["report"] else {})
    if order_term:
        extra["phase_order_term"] = order_term
    if cv:
        cvr = cross_validate(tables, fs, cfg, splits, device, seed=seed, log=_info, cache=cache)
        extra |= {"cv_metrics": cvr["metrics"], "cv_phase_order": cvr["phase_order"]}
        if "counts" in cvr:
            extra["cv_counts"] = cvr["counts"]
        if "series_order" in cvr:
            extra["cv_series_order"] = cvr["series_order"]
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
    from .metrics import mean_auroc
    return mean_auroc(block)


# training fields of TrainConfig that `--hparams` may set beside the HParams fields
HPARAMS_TRAINING_FIELDS = ("epochs", "block_epochs", "batch", "block_batch")


def read_hparams(path) -> dict:
    """The `--hparams` JSON: an object over HParams fields and HPARAMS_TRAINING_FIELDS. Unknown keys are an error."""
    from dataclasses import fields
    from .config import HParams
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--hparams {path}: {exc}") from None
    if not isinstance(d, dict):
        raise SystemExit(f"--hparams {path}: not a JSON object")
    known = {f.name for f in fields(HParams)} | set(HPARAMS_TRAINING_FIELDS)
    unknown = sorted(set(d) - known)
    if unknown:
        raise SystemExit(f"--hparams {path}: unknown keys {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    return d


def apply_hparams(cfg, d: dict) -> None:
    """Set the values of a `--hparams` object on ``cfg`` (HParams fields on cfg.hparams, the rest on cfg)."""
    from dataclasses import replace
    hp = {k: v for k, v in d.items() if k not in HPARAMS_TRAINING_FIELDS}
    if "block_dims" in hp:
        hp["block_dims"] = cfg.hparams.block_dims | dict(hp["block_dims"])
    cfg.hparams = replace(cfg.hparams, **hp)
    for k in HPARAMS_TRAINING_FIELDS:
        if k in d:
            setattr(cfg, k, int(d[k]))


def _train_config(args):
    """The TrainConfig of `train` (and `tune`) from the command line."""
    from .config import TrainConfig
    cfg = TrainConfig(seed=args.seed, fit=args.fit.replace("-", "_"), phase_head=args.phase_head)
    if getattr(args, "hparams", None):
        apply_hparams(cfg, read_hparams(args.hparams))
    for name in ("epochs", "study_cap", "label_cap", "test_frac"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    cfg.mask_missing = not args.no_mask_missing
    cfg.mask_pool = not args.no_pool
    cfg.dna_decoder, cfg.dna_inputs = args.dna_decoder, args.dna_inputs
    cfg.gc_offset, cfg.gc_scale, cfg.feature_gc = args.gc_offset, args.gc_scale, args.feature_gc
    if args.gc_scale != "learned" and not args.gc_offset:
        raise SystemExit("--gc-scale applies to the GC term: give --gc-offset too")
    cfg.clean_inputs, cfg.encoder_covariates, cfg.qc_module = args.clean_inputs, args.encoder_covariates, args.qc_module
    cfg.apply_qc_module()
    cfg.dna_penalty_w = args.dna_penalty
    cfg.phase_order_weight = args.phase_order_weight
    if args.block_dims:
        try:
            cfg.hparams = cfg.hparams.with_block_dims(args.block_dims)
        except ValueError as err:
            raise SystemExit(str(err)) from None
        _info("embedding blocks: " + ", ".join(f"{h} {d}" for h, d in cfg.hparams.block_dims.items()) + f", residual {cfg.hparams.res_dim}")
    return cfg


def cmd_train(args) -> int:
    import torch
    from .splits import SPLIT_COLUMNS
    cfg = _train_config(args)
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
        _fit_and_save(tables, fs, cfg, splits, device, seeds[0], out, args.cv, out / "missing_genes.tsv")
        _info(f"wrote the model bundle and its evaluation to {out}")
        return 0
    # several seeds on one split: each in seed_<s>/, one of them copied to the output directory
    rows, manifests, cache = [], {}, {}
    for s in seeds:
        manifests[s] = _fit_and_save(tables, fs, cfg, splits, device, s, out / f"seed_{s}", args.cv,
                                     out / "missing_genes.tsv" if s == seeds[0] else None,      # once: the calls do not depend on the seed
                                     cache)
        for h, m in manifests[s]["test_metrics"].items():
            cvm = (manifests[s].get("cv_metrics") or {}).get(h, {})
            rows.append({"seed": s, "head": h, "test_balanced_acc": m["balanced_acc"], "test_auroc": m["auroc"],
                         "cv_balanced_acc": cvm.get("balanced_acc"), "cv_auroc": cvm.get("auroc"),
                         "cv_nll": (manifests[s].get("cv_counts") or {}).get("nll_per_feature")}
                        | ({"test_order": manifests[s]["phase_order"]["test"]["share_in_order_mean_over_studies"],
                            "cv_order": (manifests[s].get("cv_phase_order") or {}).get("share_in_order_mean_over_studies")}
                           | ({"test_series_order": ((manifests[s].get("series_order") or {}).get("test") or {}).get("share_in_order"),
                               "withheld_series_order": ((manifests[s].get("series_order") or {}).get("withheld") or {}).get("share_in_order"),
                               "cv_series_order": (manifests[s].get("cv_series_order") or {}).get("share_in_order")}
                              if "series_order" in manifests[s] else {})
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


def cmd_tune(args) -> int:
    import torch
    from .tune import run_tune
    if args.hparams:
        raise SystemExit("tune: --hparams sets what the search varies; give the fixed settings by their own flags")
    cfg = _train_config(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    tables, fs, splits = _training_inputs(args, cfg)
    device = pick_device(args.device)
    seeds = tuple(int(s) for s in args.trial_seeds.split(","))
    run_tune(tables, fs, cfg, splits, device, args.out, args.trials, max_total=args.max_total_trials, trial_seeds=seeds,
             sampler_seed=args.sampler_seed, max_epochs=args.max_epochs, reset_running=args.fail_running, log=_info)
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
    from .drivers import (axis_score, combine_seeds, contrast_drivers, default_contrasts, expression_units, gene_drivers, read_axis,
                          refit_axis, study_bootstrap, unit_matrix, unit_table)
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
    axis, axis_by_path = None, {}
    if args.axis:   # the axis as `axes` fitted it on --models; with the seeds of a sweep, fitted again on each seed's embedding
        try:
            def embedding_of(d):
                e = pd.read_csv(d / "embedding.tsv", sep="\t", index_col=0)
                e.index = e.index.astype(str)
                return e
            axis = read_axis(args.axis, embedding_of(root), args.axis_score)
            axis_by_path = {p: (axis if p == root else refit_axis(axis, embedding_of(p))) for p in paths}
        except (OSError, ValueError, KeyError) as e:
            _info(f"error: --axis: {e}")
            return 2
    bundles = [ModelBundle.load(p, device) for p in paths]
    cfg = bundles[0].config
    if any((b.config.fit, b.config.phase_head) != (cfg.fit, cfg.phase_head) for b in bundles):
        _info("error: the bundles differ in fit or phase head")
        return 2
    args.splits = str(paths[0] / SPLITS_FILE)
    tables, _, splits = _training_inputs(args, cfg)
    data = build_training(tables, bundles[0].fs, cfg, splits)           # the same for every seed of a sweep: one split
    label = f"axis_{axis['name']}" if axis else args.head
    classes = axis["categories"] if axis else data.fit_classes[args.head]
    if axis:
        category = axis["category"].reindex([str(i) for i in data.ids]).fillna("").to_numpy()
        _info(f"axis {axis['name']} ({axis['method']} score), fitted categories {classes}")
    _info(f"{len(bundles)} model(s) from {root}" + (" (the seeds of the sweep)" if len(bundles) > 1 else ""))
    try:
        contrasts = [tuple(c.split(":")) for c in args.contrast] if args.contrast else default_contrasts(classes)
        if any(len(c) != 2 for c in contrasts):
            raise ValueError("--contrast takes LOW:HIGH")
        if axis:
            for c in {x for pair in contrasts for x in pair} - set(axis["category"]):
                raise ValueError(f"axis {axis['name']}: no sample has the category {c!r}")
        feat = read_feature_table(args.features)
        logx = data.X.numpy() * data.zscore.sd + data.zscore.mu
        unit = expression_units(feat, data.fs.ids, logx[data.tr_rows], data.splits["gse"].to_numpy()[data.tr_rows], min_r=args.unit_r)
        if args.modules:                # curated modules: read once, matched to the model features by symbol or locus tag
            from .modules import membership, read_module_table
            f_sym = feat["Symbol"].reindex(data.fs.ids).fillna("").astype(str).to_numpy()
            f_lt = (feat["Locus_Tag"].reindex(data.fs.ids).fillna("").astype(str).to_numpy() if "Locus_Tag" in feat.columns
                    else np.array([""] * data.fs.n))
            mod_table, mod_G = membership(read_module_table(args.modules), f_sym, f_lt, args.module_min, args.module_max)
            if not len(mod_table):
                raise ValueError(f"{args.modules}: no module has {args.module_min}-{args.module_max} genes of the model")
            _info(f"{len(mod_table)} modules of {args.modules} with {args.module_min}-{args.module_max} model features")
        n_units = len(np.unique(unit))
        _info(f"{data.fs.n} features in {n_units} expression units ({int((np.bincount(np.unique(unit, return_inverse=True)[1]) > 1).sum())} with more than one gene)")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        formats = tuple(f for f in args.formats.split(",") if f)
        apply_style()
        summaries = []
        for low, high in contrasts:
            per_gene, per_unit, per_summary, per_study = [], [], [], []
            for p, b in zip(paths, bundles):
                if axis:
                    ax = axis_by_path[p]
                    lo, hi = np.where(category == low)[0], np.where(category == high)[0]
                    table, summary, study_attr = contrast_drivers(b.model, data, axis_score(ax["weights"], ax["mean"], ax["sd"]), lo, hi,
                                                                  label, low, high, device, steps=args.steps)
                else:
                    table, summary, study_attr = gene_drivers(b.model, data, args.head, low, high, device, steps=args.steps)
                per_study.append(study_attr)
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
            if args.study_boot > 0:                                  # the seed mean, with the high-class studies drawn again
                study_attr = np.mean(per_study, axis=0)
                unit_ids = [data.fs.ids[int(np.where(unit == u)[0][0])] for u in np.unique(unit)]
                for tbl, key, P, ids, top in ((genes, "feature", study_attr, data.fs.ids, args.top_genes),
                                              (units, "unit_id", study_attr @ unit_matrix(unit), unit_ids, args.top_units)):
                    boot = study_bootstrap(P, top, n_boot=args.study_boot)
                    for col, v in boot.items():
                        tbl[col] = pd.Series(v, index=ids).reindex(tbl[key]).to_numpy()
            name = f"{label}_{low}_to_{high}"
            genes.to_csv(out / f"drivers_{name}.tsv", sep="\t", index=False, float_format="%.5g")
            units.to_csv(out / f"units_{name}.tsv", sep="\t", index=False, float_format="%.5g")
            fig_drivers(genes, label, low, high, out, formats, top=args.top)
            fig_drivers(units, label, low, high, out, formats, top=args.top, label="unit", name="units")
            summary = dict(per_summary[0])
            if args.gsea:   # GO terms, apart for the genes higher and lower in the high class, each ranked by signed attribution
                from .figures.gsea import fig_gsea_bars
                from .gsea import ranking_by_symbol, run as run_gsea
                sym = genes["symbol"] if "symbol" in genes.columns else genes["feature"]
                summary["gsea"] = {}
                for d, word in (("up", "higher"), ("down", "lower")):
                    sub = genes["direction"] == d
                    try:
                        res, info = run_gsea(ranking_by_symbol(genes.loc[sub, "attribution"], sym[sub].where(sym[sub] != "", "")),
                                             args.gene_sets, permutations=args.gsea_permutations, sides=args.gsea_sides or "two")
                    except ValueError as e:                     # too few genes of one direction for any set
                        summary["gsea"][d] = {"skipped": str(e)}
                        continue
                    res.to_csv(out / f"gsea_{name}_{d}.tsv", sep="\t", index=False, float_format="%.5g")
                    fig_gsea_bars(res, f"GO terms of the {label} drivers, {low} → {high}: genes {word} in {high} ({int(sub.sum())})",
                                  out, f"fig_gsea_{name}_{d}", formats, top=args.top, both=True,
                                  up_label=f"{word} in {high}, change moves the score towards {high}",
                                  down_label=f"{word} in {high}, change moves the score towards {low}")
                    summary["gsea"][d] = info
            if args.modules:                # curated modules scored with the attributions of every seed and study
                from .figures.modules import fig_modules
                from .modules import score_modules, summary as module_summary
                fc = genes.set_index("feature")["log2fc"].reindex(data.fs.ids).to_numpy()
                mres = score_modules(np.stack(per_study), unit, mod_table, mod_G, f_sym, fc, n_null=args.module_null,
                                     top_k=args.top_units // 4 or 1)
                mres.to_csv(out / f"modules_{name}.tsv", sep="\t", index=False, float_format="%.5g")
                fig_modules(mres, f"Modules of the {label} drivers, {low} → {high}", high, low, out, f"fig_modules_{name}", formats)
                summary["modules"] = {"table": str(args.modules), "null_draws": args.module_null, "namespaces": module_summary(mres)}
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
            if args.study_boot > 0:
                summary["study_bootstraps"] = args.study_boot
                summary["units_in_top_of_three_quarters_of_study_bootstraps"] = int((units["study_in_top"] >= 0.75).sum())
            if len(bundles) > 1:
                summary["units_in_top_of_three_quarters_of_seeds"] = int(robust.sum())
                by_seed = pd.concat([u.set_index("unit_id")["attribution"] for u in per_unit], axis=1).corr(method="spearman").to_numpy()
                summary["rank_correlation_between_seeds"] = float(by_seed[np.triu_indices(len(per_unit), 1)].mean())
                by_seed = pd.concat([g.set_index("feature")["attribution"] for g in per_gene], axis=1).corr(method="spearman").to_numpy()
                summary["rank_correlation_between_seeds_genes"] = float(by_seed[np.triu_indices(len(per_gene), 1)].mean())
            summaries.append(summary)
            _info(f"{name}: {summary['n_samples'][high]} {high} samples in {summary['n_studies'][high]} studies; "
                  f"top 50 units carry {summary['share_of_attribution']['top_50_units']:.0%} of the attribution"
                  + (f"; {int(robust.sum())} units are among the top {args.top_units} in at least {int(np.ceil(0.75 * len(bundles)))} of {len(bundles)} seeds, "
                     f"rank correlation between seeds {summary['rank_correlation_between_seeds']:.2f}" if len(bundles) > 1 else ""))
    except (ValueError, ImportError) as e:
        _info(f"error: {e}")
        return 2
    report = {"models": [str(p) for p in paths], "phase_head": cfg.phase_head, "fit": cfg.fit, "classes": list(classes),
              "unit_min_correlation": args.unit_r, "contrasts": summaries}
    if axis:
        report["axis"] = {"summary": str(args.axis), "name": axis["name"], "column": axis["column"], "score": axis["method"]}
    _write_json(report, out / f"drivers_{label}.json")
    _info(f"wrote the driver tables and figures to {out}")
    return 0


def _studies_arg(path):
    """sample -> study from a table whose first column is the sample and whose second (or `GSE` / `study`) column is the study."""
    if not path:
        return None
    from .compare import read_sample_table
    t = read_sample_table(path)
    col = next((c for c in ("GSE", "gse", "study") if c in t.columns), t.columns[0])
    return t[col]


def cmd_embed(args) -> int:
    from .artifacts import ModelBundle
    from .data import read_table
    from .embed import embed_counts
    from .io import read_counts
    from .qc_record import load_ingest_results
    if bool(args.ingest_results) == bool(args.counts):
        _info("error: give an inGEST run with --ingest-results DIR, or --counts with --qc")
        return 1
    if args.counts and not args.qc:
        _info("error: --counts needs --qc (the samples' QC table); with an inGEST run use --ingest-results DIR, "
              "which reads the counts, antisense counts and QC records itself")
        return 1
    device = pick_device(args.device)
    bundle = ModelBundle.load(args.models, device)
    try:
        if args.ingest_results:
            run = load_ingest_results(args.ingest_results)
            counts, qc = run["counts"], run["qc"]
            antisense = read_counts(args.antisense) if args.antisense else run["antisense"]
            _info(f"read {counts.shape[1]} samples from {run['results_dir']}"
                  + ("" if antisense is not None else " (no antisense count table there)"))
        else:
            counts, qc = read_counts(args.counts), read_table(args.qc)
            antisense = read_counts(args.antisense) if args.antisense else None
        products = tuple(p for p, on in (("corrected", args.corrected_counts), ("decoded", args.decoded_counts)) if on)
        feature_gc = None
        if products and args.feature_gc:
            from .technical import load_feature_gc
            feature_gc = load_feature_gc(args.feature_gc)
        res = embed_counts(bundle, counts, qc, device, args.min_feature_coverage, antisense,
                           studies=_studies_arg(args.studies), pool=not args.no_pool, products=products, feature_gc=feature_gc,
                           fill_missing_gc=args.fill_missing_gc)
    except (ValueError, FileNotFoundError) as exc:
        _info(f"error: {exc}")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.ingest_results:
        qc.to_csv(out / "qc.tsv", sep="\t")
    for w in res["warnings"]:
        _info(f"warning: {w}")
    res["embedding"].to_csv(out / "embedding.tsv", sep="\t", float_format="%.6g")
    res["predictions"].to_csv(out / "predictions.tsv", sep="\t", float_format="%.6g")
    res["samples"].to_csv(out / "samples.tsv", sep="\t", float_format="%.6g")
    high = res["samples"].index[res["samples"]["high_dna"] == 1]
    if len(high):
        limit = bundle.config.max_dna_fraction
        _info(f"warning: {len(high)} sample(s) have a genomic-DNA estimate above {limit:.0%}, the level above which samples were kept out of "
              f"the atlas; their calls and position may be affected: " + ", ".join(f"{s} ({res['samples'].loc[s, 'dna_fraction_est']:.0%})" for s in high[:10])
              + (" ..." if len(high) > 10 else ""))
    low = res["samples"].index[res["samples"]["low_depth"] == 1]
    if len(low):
        gate = bundle.config.min_cds_reads
        _info(f"warning: {len(low)} sample(s) have fewer than {gate:,} reads on CDS features, the depth below which no sample was part of "
              f"the atlas, so they are outside the depths the model was trained on; read their calls and position with care: "
              + ", ".join(f"{s} ({res['samples'].loc[s, 'cds_reads']:,})" for s in low[:10]) + (" ..." if len(low) > 10 else ""))
    res["missing"].to_csv(out / "missing_genes.tsv", sep="\t", index=False)
    res["covariates"].to_csv(out / "covariates.tsv", sep="\t", float_format="%.6g")
    res["coverage"].to_frame(bundle.fs.ids).to_csv(out / "feature_coverage.tsv", sep="\t")
    written = []
    for key in ("corrected_counts", "decoded_counts"):
        if key in res:
            res[key].to_csv(out / f"{key}.tsv", sep="\t", float_format="%.6g")
            written.append(f"{key}.tsv")
    if args.seeds:                               # every seed of the bundle: the spread of the calls (and of the axes, `axes --seeds`)
        from .seeds import SEED_PREDICTIONS_FILE, SEEDS_DIR, seed_dirs, seed_prediction_summary
        dirs = seed_dirs(args.models)
        if not dirs:
            _info(f"error: --seeds: {args.models} has no seed_<n> bundles (a bundle trained with one seed)")
            return 1
        per_seed = {}
        for d in dirs:
            try:
                r = embed_counts(ModelBundle.load(d, device), counts, qc, device, args.min_feature_coverage, antisense,
                                 studies=_studies_arg(args.studies), pool=not args.no_pool, fill_missing_gc=args.fill_missing_gc)
            except (ValueError, FileNotFoundError) as exc:
                _info(f"error: --seeds, {d.name}: {exc}")
                return 1
            sd = out / SEEDS_DIR / d.name
            sd.mkdir(parents=True, exist_ok=True)
            r["embedding"].to_csv(sd / "embedding.tsv", sep="\t", float_format="%.6g")
            r["predictions"].to_csv(sd / "predictions.tsv", sep="\t", float_format="%.6g")
            per_seed[d.name] = r["predictions"]
        seed_prediction_summary(per_seed).to_csv(out / SEED_PREDICTIONS_FILE, sep="\t", float_format="%.6g")
        written.append(f"{SEED_PREDICTIONS_FILE} ({len(dirs)} seeds; each in {SEEDS_DIR}/)")
    _info(f"embedded {len(res['embedding'])} samples ({res['coverage'].fraction:.1%} of the model features present); "
          f"wrote embedding.tsv, predictions.tsv, samples.tsv, covariates.tsv, missing_genes.tsv, feature_coverage.tsv"
          + "".join(f", {w}" for w in written) + (" and qc.tsv" if args.ingest_results else "") + f" to {out}")
    return 0


def _blocks_arg(value):
    return "all" if value in (None, "all") else tuple(value.split(","))


def cmd_compare(args) -> int:
    from .compare import Atlas, calls_frame, compare, read_embedding, read_sample_table
    atlas = Atlas.load(args.models, args.atlas_metadata)
    if args.exclude_study:                       # e.g. the query's own study, when the atlas holds its samples already
        n0 = len(atlas.embedding)
        try:
            atlas = atlas.without_studies(args.exclude_study)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        _info(f"left {n0 - len(atlas.embedding)} atlas samples of {', '.join(args.exclude_study)} out of the neighbours")
    query = read_embedding(args.embeddings)
    emb_dir = Path(args.embeddings) if Path(args.embeddings).is_dir() else Path(args.embeddings).parent
    pred_path = Path(args.predictions) if args.predictions else emb_dir / "predictions.tsv"
    preds = calls_frame(pd.read_csv(pred_path, sep="\t", index_col=0)) if pred_path.exists() else None
    metadata = read_sample_table(args.metadata) if args.metadata else None
    group_by = None
    if args.group_by:
        parts = tuple(c.strip() for c in args.group_by.split(",") if c.strip())
        group_by = parts[0] if len(parts) == 1 else parts
    try:
        summary = compare(query, atlas, args.out, k=args.k, metric=args.metric, blocks=_blocks_arg(args.blocks), query_predictions=preds,
                          do_umap=args.umap, seed=args.seed, log=_info, metadata=metadata, group_by=group_by,
                          order_by=args.order_by, color_by=args.color_by, plots=args.plots, trajectory=args.order_by is not None,
                          formats=tuple(args.formats.split(",")))
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    show = [c for c in summary.columns if c.endswith("_knn_vote") or c in ("nearest_atlas_sample", "top_study", "novelty_percentile")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        _info(summary[show].head(20).to_string())
    _info(f"wrote {Path(args.out) / 'summary.tsv'} and neighbors.tsv")
    return 0


def cmd_view(args) -> int:
    from .viewer.build import write_view
    try:
        out = write_view(args.out, plotly_js=args.plotly, log=_info, models=args.models, metadata=args.metadata,
                         studies=args.studies, sample_text=args.sample_text, embeddings=args.embeddings or None,
                         presets=args.presets, umap=not args.no_umap, k=args.k, metric=args.metric, seed=args.seed,
                         title=args.title, counts=args.counts, qc=args.qc, n_pcs=args.n_pcs)
    except (ValueError, FileNotFoundError) as exc:
        _info(f"error: {exc}")
        return 1
    _info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); open it in a web browser")
    return 0


def cmd_neighbors(args) -> int:
    from .neighbors import run_neighbors
    gses = list(args.gse or [])
    if args.gse_file:
        gses += [ln.strip().split()[0] for ln in Path(args.gse_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not gses:
        _info("error: give at least one GSE accession (positional or --gse-file)")
        return 2
    try:
        report = run_neighbors(gses, args.models, args.out, metadata=args.metadata, k=args.k, metric=args.metric,
                               blocks=_blocks_arg(args.blocks), exclude_same_study=not args.within_study, do_umap=args.umap,
                               seed=args.seed, allow_missing=args.allow_missing, log=_info)
    except ValueError as exc:
        _info(f"error: {exc}")
        return 1
    _info(f"wrote neighbors.tsv, summary.tsv ({report['n_query']} query samples) and study_summary.tsv "
          f"({len(report['found'])} studies) to {args.out}")
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
    from .figures.gene_correlates import fig_head_gene_correlates
    from .figures.batch_tracking import fig_batch_tracking
    from .figures.gsea import fig_gsea_batch
    from .figures.leakage import fig_study_leakage
    from .figures.pca import fig_batch_genes, fig_pca_batch
    from .figures.test_eval import fig_head_test_eval, misclassified_table
    from .figures.thresholds import CV_FILE, cv_metrics, fig_threshold_curves, load_cv_predictions
    from .figures.head_scores import fig_head_scores
    ctx, tables, splits = _figure_context(args)
    manifest = ctx.bundle.manifest
    metrics = {"models": str(args.models), "embedding_check": ctx.embedding_check(args.models)}
    ok = metrics["embedding_check"].get("max_abs_diff", 0.0) <= 1e-4
    _info(f"model figures for {len(ctx.ids)} samples; embedding against the bundle's: {metrics['embedding_check']}")

    def test_eval():
        res = fig_head_test_eval(ctx, out, formats)
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
            "head_test_eval": test_eval, "head_scores": lambda: fig_head_scores(ctx, out, formats),
            "head_gene_correlates": lambda: fig_head_gene_correlates(ctx, out, formats),
            "study_leakage": lambda: fig_study_leakage(ctx, out, formats), "threshold_curves": thresholds,
            "gsea": lambda: fig_gsea_batch(ctx, out, formats, permutations=args.gsea_permutations, gene_sets=args.gene_sets,
                                           sides=args.gsea_sides or "one", modules=args.modules,
                                           threads=args.threads or 4),
            "batch_tracking": lambda: fig_batch_tracking(ctx, out, formats, modules=args.modules, x=args.tracking_x)}
    for name in names:
        try:
            metrics[name] = draw[name]()
            _info(f"  {name}: done")
        except (ValueError, ImportError) as e:
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
    pw = {"gsea": args.gsea, "modules": args.modules, "gene_sets": args.gene_sets, "permutations": args.gsea_permutations,
          "sides": args.gsea_sides or "two"}
    run = {"uti": lambda: run_uti(ctx, out, formats, n_perm=args.n_perm, **pw),
           "oxygen": lambda: run_oxygen(ctx, out, formats, studies=args.oxygen_studies.split(","), **pw),
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


def _series_table(spec: str, query_study: str | None = None) -> pd.DataFrame:
    """--series: a table (sample, series, x[, x_label][, panel]), or ``PATH:SERIES_COLUMN:X_COLUMN[:PANEL_COLUMN]`` for
    any sample table (CSV or TSV, first column the sample id), e.g. ``samples.csv:replicate:time_h``; then the series are
    named after the query study (or the column) and its values. A panel column draws the series in one panel per value
    instead of one per study."""
    if Path(spec).exists() or spec.count(":") < 2:
        t = pd.read_csv(spec, sep="\t", dtype={"sample": str, "series": str, "panel": str})
        lacking = {"sample", "series", "x"} - set(t.columns)
        if lacking:
            raise ValueError(f"{spec} lacks columns {sorted(lacking)}")
        return t
    parts = spec.rsplit(":", 3)
    if len(parts) == 4 and Path(parts[0]).exists():
        path, scol, xcol, pcol = parts
    else:
        (path, scol, xcol), pcol = spec.rsplit(":", 2), None
    t = pd.read_csv(path, sep="," if path.endswith(".csv") else "\t", dtype=str, keep_default_na=False)
    t = t.set_index(t.columns[0])
    for c in (scol, xcol, pcol):
        if c is not None and c not in t.columns:
            raise KeyError(f"{path} has no column {c!r}")
    prefix = f"{query_study} " if query_study and pcol is None else ""
    out = pd.DataFrame({"sample": t.index.astype(str).str.strip(), "series": [f"{prefix}{scol} {v}" for v in t[scol]],
                        "x": pd.to_numeric(t[xcol], errors="coerce").to_numpy(), "x_label": xcol})
    if pcol is not None:
        out["panel"] = t[pcol].astype(str).to_numpy()
    return out.dropna(subset=["x"])


def cmd_extract(args) -> int:
    from .extract import extract_studies
    gses = list(args.gse)
    if args.gse_file:
        gses += [ln.strip() for ln in Path(args.gse_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not gses:
        _info("error: name at least one study (GSE)")
        return 2
    try:
        res = extract_studies(args.models, gses, args.out, seeds=args.seeds)
    except (ValueError, FileNotFoundError) as e:
        _info(f"error: {e}")
        return 1
    if res["missing_studies"]:
        _info(f"warning: not in the atlas of {args.models}: {', '.join(res['missing_studies'])}")
    _info(f"extracted {res['n_samples']} samples of {', '.join(res['studies'])}"
          + (f" with {len(res['seeds'])} seeds" if res["seeds"] else "") + f" to {args.out} (embedding.tsv, predictions.tsv"
          + (", predictions_seeds.tsv, seeds/" if res["seeds"] else "") + "); give the studies to --exclude-study of `axes` / `compare`")
    return 0


def cmd_axes(args) -> int:
    from .axes import AxisSpec, axis_scores, category_frame, load_embedding, summarize
    from .data import read_table
    from .figures.axes import fig_axes_map, fig_axis_distributions, fig_axis_series, fig_heads_map, fig_heads_seed_votes, fig_heads_series
    from .figures.plotting import apply_style
    try:
        specs = [AxisSpec.parse(a, args.holdout or ()) for a in args.axis]
        emb, splits, _ = load_embedding(args.models, args.blocks)
    except (ValueError, FileNotFoundError) as e:
        _info(f"error: {e}")
        return 1
    meta = read_table(args.metadata)
    if args.exclude_study:                       # atlas studies kept out of the fits and the tables (e.g. a held-out study)
        keep = ~splits["gse"].astype(str).isin(args.exclude_study).to_numpy()
        emb, splits = emb[keep], splits[keep]
    missing = sorted(set(emb.index) - set(meta.index))
    if missing:
        _info(f"error: {len(missing)} samples of the embedding are not in the metadata ({missing[:5]})")
        return 1
    meta = meta.loc[emb.index]
    query_ids: list = []
    query_heads = None
    atlas_ids = list(emb.index)
    if args.query:                               # new samples (`embed` output): only scored, never fitted
        qpath = Path(args.query)
        qpath = qpath / "embedding.tsv" if qpath.is_dir() else qpath
        try:
            q = pd.read_csv(qpath, sep="\t", index_col=0)
        except FileNotFoundError as e:
            _info(f"error: {e}")
            return 1
        q.index = q.index.astype(str)
        lacking = [c for c in emb.columns if c not in q.columns]
        clash = sorted(set(q.index) & set(emb.index))
        if lacking or clash:
            _info(f"error: --query {qpath}: " + (f"lacks the embedding columns {lacking[:5]} (embedded with another bundle?)" if lacking
                                                  else f"{len(clash)} sample names are also atlas samples ({clash[:5]})"))
            return 1
        query_ids = list(q.index)
        if (qpath.parent / "predictions.tsv").exists():       # the head calls of `embed`, drawn along with the axes
            query_heads = pd.read_csv(qpath.parent / "predictions.tsv", sep="\t", index_col=0)
            query_heads.index = query_heads.index.astype(str)
        emb = pd.concat([emb, q[list(emb.columns)]])
        splits = pd.concat([splits, pd.DataFrame({"gse": args.query_study, "split": 3}, index=q.index)])
        meta = pd.concat([meta, pd.DataFrame("", index=q.index, columns=meta.columns)])
    where: dict = {}
    for w in args.where or ():
        col, _, values = w.partition("=")
        where[col.strip()] = [v.strip() for v in values.split(",")]
    series = None
    if args.series:
        try:
            series = _series_table(args.series, args.query_study if query_ids else None)
        except (ValueError, FileNotFoundError, KeyError) as e:
            _info(f"error: --series: {e}")
            return 1
    seed_Z, seed_heads = {}, None
    if args.seeds:                               # the same axes fitted on every seed's embedding: the spread of the scores
        from .seeds import SEEDS_DIR, seed_dirs
        dirs = seed_dirs(args.models)
        if not dirs:
            _info(f"error: --seeds: {args.models} has no seed_<n> bundles (a bundle trained with one seed)")
            return 1
        qdir = (Path(args.query) if Path(args.query).is_dir() else Path(args.query).parent) if query_ids else None
        for d in dirs:
            e, _, _ = load_embedding(d, args.blocks)
            e = e.reindex(atlas_ids)
            if query_ids:
                qp = qdir / SEEDS_DIR / d.name / "embedding.tsv"
                if not qp.exists():
                    _info(f"error: --seeds: no {qp}; embed the --query samples with `embed --seeds`")
                    return 1
                qs = pd.read_csv(qp, sep="\t", index_col=0)
                qs.index = qs.index.astype(str)
                e = pd.concat([e, qs.reindex(query_ids)[list(e.columns)]])
            if e.isna().any().any():
                _info(f"error: --seeds: the embedding of {d.name} lacks samples of the bundle's embedding")
                return 1
            seed_Z[d.name] = e.to_numpy(dtype=np.float64)
        if query_ids:                            # each seed's calls of the query samples, for fig_heads_series
            seed_heads = {}
            for d in dirs:
                f = qdir / SEEDS_DIR / d.name / "predictions.tsv"
                if f.exists():
                    seed_heads[d.name] = pd.read_csv(f, sep="\t", index_col=0)
                    seed_heads[d.name].index = seed_heads[d.name].index.astype(str)
            seed_heads = seed_heads or None
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
        if query_ids:
            scores.loc[query_ids, "role"] = "query"
        if seed_Z:
            from .seeds import spread
            try:
                per = {n: axis_scores(Z, frame, spec, n_folds=args.folds, seed=args.seed, l2=args.l2)["scores"] for n, Z in seed_Z.items()}
            except ValueError as e:
                _info(f"error: --seeds: {e}")
                return 1
            for m in ("ordinal", "anchor"):
                sp_ = spread(pd.DataFrame({n: r[m] for n, r in per.items()}))
                for k in ("median", "min", "max"):
                    scores[f"{m}_seed_{k}"] = sp_[k]
            scores["n_seeds"] = len(per)
            seed_scores = pd.DataFrame({n: r["ordinal"] for n, r in per.items()}).rename_axis("sample")
            seed_scores.to_csv(out / f"axis_{spec.name}_seed_scores.tsv", sep="\t", float_format="%.5g")
        else:
            seed_scores = None
        if spec.members:                          # grouped categories: keep the column value of each sample
            scores.insert(scores.columns.get_loc("category") + 1, "value",
                          meta[spec.column].astype(str).str.strip().reindex(scores.index).where(scores["category"] != "", ""))
        scores.to_csv(out / f"axis_{spec.name}_scores.tsv", sep="\t", float_format="%.5g")
        summary = summarize(scores, spec) | {"column": spec.column, "blocks": args.blocks, "where": where, "l2": args.l2, "folds": args.folds,
                                             "thresholds": res["thresholds"],
                                             "weights": res["weights"], "embedding_columns": list(emb.columns)}
        if spec.members:
            summary["members"] = {c: list(spec.values_of(c)) for c in spec.categories}
            summary["value_medians"] = {m: {v: float(scores.loc[scores["value"] == v, m].median()) for c in spec.categories
                                            for v in spec.values_of(c) if (scores["value"] == v).any()} for m in ("ordinal", "anchor")}
        _write_json(summary, out / f"axis_{spec.name}_summary.json")
        fig_axis_distributions(scores, spec, res["thresholds"], out, formats)
        if series is not None:
            fig_axis_series(scores, series, spec, out, formats, seeds=seed_scores, x_axis=args.series_x)
        done[spec.name] = (scores, spec)
        m = summary["methods"]["ordinal"]["all"]
        ws = m["within_study"]
        _info(f"axis {spec.name}: {m['n']} samples with a category; ordinal score: Spearman {m['spearman']:.3f}; within-study pairs "
              f"in order: {100 * (ws['share_in_order_mean_over_studies'] or float('nan')):.1f}% (mean of {ws['n_studies']} studies)")
    names = list(done)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fig_axes_map(done[names[i]][0], done[names[i]][1], done[names[j]][0], done[names[j]][1], out, formats,
                         query=query_ids, series=series)
    if query_heads is not None:                  # the calls on their own 0-1 scale, next to the axis scores
        atlas_heads = None
        apath = Path(args.models) / "predictions.tsv"
        if apath.exists():
            atlas_heads = pd.read_csv(apath, sep="\t", index_col=0)
            atlas_heads.index = atlas_heads.index.astype(str)
            atlas_heads = atlas_heads.loc[atlas_heads.index.intersection(emb.index)]   # without --exclude-study studies
        if series is not None:
            fig_heads_series(query_heads, series, out, formats, seeds=seed_heads, x_axis=args.series_x)
            if seed_heads:
                fig_heads_seed_votes(query_heads, series, seed_heads, out, formats)
        fig_heads_map(atlas_heads, query_heads, out, formats, series=series)
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

    def add_gsea(sp):
        sp.add_argument("--gene-sets", metavar="TSV", help="gene-set table for the enrichment (term, namespace, name, genes; "
                        "default: the package's GO sets, reference/go_gene_sets_ecocyc.tsv.gz)")
        sp.add_argument("--gsea-permutations", type=int, default=10000, metavar="N", help="permutations of the enrichment test (10000)")
        sp.add_argument("--gsea-sides", choices=["one", "two"], help="one: are a set's genes at the top of the ranking (fgsea "
                        "scoreType pos, NES >= 0); two: top or bottom (gseapy, NES signed). Default: one for the `figures` "
                        "panel gsea, two for drivers and case studies")

    def add_tables(sp):
        sp.add_argument("--counts", required=True, help="joined count table of the atlas (features x samples)")
        sp.add_argument("--metadata", required=True, help="sample metadata (metadata_v2.tsv)")
        sp.add_argument("--qc", required=True, help="QC table (sample_qc.tsv); its in_model column selects the samples")
        sp.add_argument("--features", required=True, help="feature table of the counting annotation (*_counting_v2.features.tsv)")
        sp.add_argument("--model-name", help="use only the samples whose in_model has this value (default: any)")
        sp.add_argument("--antisense", help="antisense count table of the atlas (features x samples); a gene with antisense reads "
                        "is not called missing from a strain")
        sp.add_argument("--time-series", help="time-course table (atlas_data/time_courses.tsv: sample, series, x): time pairs for the "
                        "phase-order term, and the within-series order of the phase score in the evaluation")

    from .download import BUNDLES
    sp = sub.add_parser("download", help="download the data bundles of the release: training-data (into atlas_data/), "
                        "models (into outputs/)")
    sp.add_argument("bundles", nargs="+", choices=sorted(BUNDLES) + ["all", "modules"],
                    help="training-data, models, all (both), or modules (curated gene modules for drivers --modules: fetched "
                         "from their sources and built into atlas_data/modules/modules.tsv)")
    sp.add_argument("--features", help="modules: the counting feature table to map genes to (default: "
                    "resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv)")
    sp.add_argument("--dest", help="directory to unpack into (default: atlas_data/ or outputs/ under $BS_DATA_ROOT or here)")
    sp.add_argument("--dry-run", action="store_true", help="print what would be downloaded")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("prepare", help="write the input tables of `embed` (counts, antisense counts, QC) from Ecoli_inGEST results")
    sp.add_argument("--ingest-results", metavar="DIR", help="an inGEST run directory or its results directory")
    sp.add_argument("--counts", metavar="FILE", help="a count table instead of --ingest-results")
    sp.add_argument("--metadata", metavar="TSV", help="optional sample metadata (first column = sample id)")
    sp.add_argument("--models", metavar="DIR", help="model bundle; if given, also write feature_coverage.tsv")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_prepare)

    def add_train_config(sp):
        """The flags that make the TrainConfig (`_train_config`), shared by `train` and `tune`."""
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
        sp.add_argument("--dna-decoder", action="store_true", help="add a genomic-DNA term to the expected counts (off by default)")
        sp.add_argument("--dna-inputs", action="store_true", help="take the expected genomic-DNA reads off the counts the encoders see (off by default)")
        sp.add_argument("--gc-offset", action="store_true",
                        help="mechanistic GC bias in the decoder: each library's GC fit (gc_slope, gc_curv) at every feature's GC content, "
                        "with one learned scale; the two GC covariates then leave the linear covariate term (off by default)")
        sp.add_argument("--gc-scale", choices=["learned", "fixed"], default="learned",
                        help="with --gc-offset: fit the term's scale (learned, default) or hold it at 1, the gene-level value of the "
                        "measured fit (fixed), so the decoder cannot split the GC effect between the term and the embedding")
        sp.add_argument("--feature-gc", help="with --gc-offset or --clean-inputs: GC content of the features (Feature_ID, Length, GC); "
                        "default: the table for the inGEST counting annotation shipped with the package")
        sp.add_argument("--clean-inputs", action=argparse.BooleanOptionalAction, default=True,
                        help="take each library's measured GC offset off the counts the encoders see (default); "
                        "--no-clean-inputs: the raw counts, as before 2026-11-22")
        sp.add_argument("--encoder-covariates", action=argparse.BooleanOptionalAction, default=False,
                        help="give the technical covariates to the residual encoder too (--no-encoder-covariates, expression only, "
                        "is the default)")
        sp.add_argument("--dna-penalty", type=float, default=1.0,
                        help="weight of the genomic-DNA penalty on the residual block (default 1; 0 = off): the R2 of the DNA estimate "
                        "from the residual within biological replicate groups (metadata_v2 replicate_group), added to the loss; left out "
                        "when the metadata has no replicate groups")
        sp.add_argument("--qc-module", action="store_true",
                        help="all technical terms in one module: --gc-offset --gc-scale fixed --dna-decoder --clean-inputs "
                        "--no-encoder-covariates")
        sp.add_argument("--phase-order-weight", type=float, default=0.0,
                        help="weight of the phase-order term of the phase block (0 = off): a ranking loss on the phase score over pairs of samples "
                        "whose order along the growth curve is known, inside a study (phase levels) or a culture series (--time-series)")
        sp.add_argument("--block-dims", help="sizes of embedding blocks, e.g. phase=6,medium=12,lab_strain=4 or residual=16 (default: 1 per call, "
                        "residual 24; blocks not named keep their size). The bundle records the sizes")
        sp.add_argument("--no-pool", action="store_true", help="do not pool the missing-gene calls over a study's samples of one strain")
        sp.add_argument("--no-mask-missing", action="store_true", help="treat genes a strain does not have as measured zeros "
                        "(default: they are called from the counts and left out of the fit)")
        sp.add_argument("--hparams", metavar="JSON", help="network sizes, loss weights and epochs as a JSON object over the fields of "
                        "config.HParams and epochs, block_epochs, batch, block_batch (e.g. best/trial_<n>.json of `tune`); "
                        "--block-dims and --epochs override it")

    sp = sub.add_parser("train", help="train the atlas model and evaluate it on the held-out studies")
    add_tables(sp)
    sp.add_argument("--out", required=True, help="output directory for the bundle")
    sp.add_argument("--splits", help="reuse a splits.csv instead of making the split")
    sp.add_argument("--split-only", action="store_true", help="make the split, write splits.csv and stop")
    sp.add_argument("--seed", type=int, default=77, help="seed of the split and, without --seeds, of the model")
    sp.add_argument("--seeds", help="comma-separated model seeds, all trained on the same split (each in seed_<s>/)")
    sp.add_argument("--cv", action="store_true", help="also fit the study folds for out-of-fold metrics (ranks --seeds)")
    add_train_config(sp)
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_device(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("tune", help="hyper-parameter search (Optuna): network sizes, loss weights and epochs, scored out of fold "
                        "on the study folds (count NLL and mean head AUROC, a Pareto front); never on TEST")
    add_tables(sp)
    sp.add_argument("--splits", required=True, help="the splits.csv of the model (fixed for the whole study)")
    sp.add_argument("--out", required=True, help="study directory: study.journal, trials.tsv, pareto.tsv, best/trial_<n>.json; "
                    "an existing study there is resumed (several workers may share it)")
    sp.add_argument("--trials", type=int, default=40, help="trials run by this worker (40)")
    sp.add_argument("--max-total-trials", type=int, default=160,
                    help="stop once the study holds this many trials over all workers and runs (160)")
    sp.add_argument("--trial-seeds", default="1", help="comma-separated model seeds fitted per trial; the objectives are their mean (1)")
    sp.add_argument("--sampler-seed", type=int, help="seed of the TPE sampler (give each worker its own)")
    sp.add_argument("--fail-running", action="store_true",
                    help="first mark trials left running by a killed worker as failed (only when no other worker runs)")
    sp.add_argument("--max-epochs", type=int, help=argparse.SUPPRESS)          # smoke runs: caps epochs and block_epochs
    sp.add_argument("--seed", type=int, default=77, help=argparse.SUPPRESS)
    add_train_config(sp)
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_device(sp)
    sp.set_defaults(func=cmd_tune)

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
    sp.add_argument("--study-boot", type=int, default=200, metavar="N",
                    help="draws of the high-class studies for the study_sd / study_in_top / study_same_sign columns (0: none)")
    sp.add_argument("--axis", metavar="SUMMARY_JSON", help="attribute an `axes` dimension instead of a head: the "
                    "axis_<name>_summary.json that `axes` wrote for this bundle (classes = the axis's fitted categories); with "
                    "the seed_<n> bundles of a sweep, the axis is fitted again on each seed's embedding and the seeds are averaged")
    sp.add_argument("--axis-score", default="ordinal", choices=["ordinal", "anchor"], help="which score of the axis (ordinal)")
    sp.add_argument("--gsea", action="store_true", help="also the GO-term enrichment of the genes ranked by attribution "
                    "(gsea_<contrast>.tsv, fig_gsea_<contrast>; needs gseapy)")
    sp.add_argument("--modules", metavar="TSV", help="also score curated gene modules (modules_<contrast>.tsv, "
                    "fig_modules_<contrast>): a module table from `ecoli_atlas download modules` (regulons, iModulons, KEGG) "
                    "or any table with term, namespace, name, genes")
    sp.add_argument("--module-null", type=int, default=10000, metavar="N", help="null draws of the module scores (10000)")
    sp.add_argument("--module-min", type=int, default=3, help="smallest module scored, in model genes (3)")
    sp.add_argument("--module-max", type=int, default=500, help="largest module scored, in model genes (500)")
    add_gsea(sp)
    sp.add_argument("--out", required=True)
    sp.add_argument("--formats", default="pdf,png")
    add_device(sp)
    sp.set_defaults(func=cmd_drivers)

    sp = sub.add_parser("embed", help="embed new samples with a trained bundle",
                        description="Embed new samples. With an inGEST run: `ecoli_atlas embed --ingest-results RUN_DIR --models BUNDLE "
                        "--out DIR` reads the counts, antisense counts and the samples' QC records. Otherwise give --counts and --qc.")
    sp.add_argument("--ingest-results", metavar="DIR", help="an inGEST run directory (or its results directory): counts, "
                    "antisense counts and QC records are read from it")
    sp.add_argument("--counts", help="count table instead of --ingest-results (inGEST all_samples_raw_counts.txt, or the counts.tsv "
                    "of `prepare`); needs --qc")
    sp.add_argument("--models", required=True, help="model bundle")
    sp.add_argument("--qc", help="QC table of the samples, with --counts (the qc.tsv of `prepare`, or the sample_qc.tsv layout)")
    sp.add_argument("--antisense", help="antisense count table of the samples (inGEST all_samples_antisense_counts.txt; read from the "
                    "run with --ingest-results): a gene with antisense reads is not called missing")
    sp.add_argument("--studies", help="table sample -> study (first column the sample; column GSE or study, else the second column): "
                    "the missing-gene calls are pooled within a study. Default: all samples of the table are one study")
    sp.add_argument("--no-pool", action="store_true", help="call the genes missing from a strain for every sample on its own")
    sp.add_argument("--fill-missing-gc", choices=["study", "atlas"],
                    help="embed samples whose QC record has no GC bias fit (too few deep genes) with the median of their study's "
                    "samples (study; --studies, else the whole table) or the atlas training mean (atlas), instead of refusing them; "
                    "samples.tsv says which (gc_filled). Such libraries are usually below the depth gate too")
    sp.add_argument("--min-feature-coverage", type=float, default=0.9)
    sp.add_argument("--seeds", action="store_true",
                    help="also embed with every seed_<n> model of the bundle: predictions_seeds.tsv (median, min, max and the number "
                    "of seeds making each call) and seeds/seed_<n>/ (embedding and predictions, for `axes --seeds`). The spread "
                    "is the dependence on training randomness only, not the sampling noise of the counts")
    sp.add_argument("--corrected-counts", action="store_true",
                    help="write corrected_counts.tsv: the observed counts with the technical part swapped for that of a reference "
                    "library (paired-end, the atlas's mean technical values, no GC bias): the decoder's covariate term moved to the "
                    "reference and the GC bias of the library taken off by its measured fit. Batch effects the covariates do not "
                    "describe stay")
    sp.add_argument("--decoded-counts", action="store_true",
                    help="write decoded_counts.tsv: the decoder's expected counts at each sample's embedding with the same swap; "
                    "denoised, but they lose real within-sample detail through the embedding (a summary, not corrected counts)")
    sp.add_argument("--feature-gc", help="GC content of the features (Feature_ID, Length, GC; ecoli_atlas/scripts/feature_gc.py); "
                    "default: the table for the inGEST counting annotation shipped with the package")
    sp.add_argument("--out", required=True)
    add_device(sp)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("compare", help="new samples against the atlas: nearest atlas samples, their labels, novelty, optional UMAP")
    sp.add_argument("--embeddings", required=True, help="output directory of `embed` for the new samples (or its embedding.tsv)")
    sp.add_argument("--models", required=True, help="model bundle the samples were embedded with; its embedding is the atlas")
    sp.add_argument("--atlas-metadata", help="sample metadata of the atlas (metadata_v2.tsv); adds strain, medium, ... of the neighbours")
    sp.add_argument("--predictions", help="predictions.tsv of the new samples (default: next to --embeddings)")
    sp.add_argument("--out", required=True)
    sp.add_argument("-k", type=int, default=15)
    sp.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    sp.add_argument("--blocks", default="all", help="all or comma-separated block names, e.g. phase,residual")
    sp.add_argument("--umap", action="store_true")
    sp.add_argument("--seed", type=int, default=77)
    sp.add_argument("--metadata", help="table of the new samples (first column = sample id; CSV or TSV, e.g. the samples.tsv of `prepare`)")
    sp.add_argument("--group-by", help="metadata column(s), comma-separated: write group_summary.tsv, group_neighbors.tsv (the pooled "
                    "neighbours of each group by atlas study, strain, genotype, treatment, ...) and the group figures")
    sp.add_argument("--exclude-study", action="append", metavar="GSE",
                    help="leave the atlas samples of this study out of the neighbours (repeatable), e.g. the study the new samples "
                    "come from when the atlas holds it already, so that its own copies are not their nearest neighbours")
    sp.add_argument("--order-by", help="metadata column that orders the groups (numeric when it parses, e.g. a time)")
    sp.add_argument("--color-by", help="metadata column that colours the new samples in the UMAP (default: the groups)")
    sp.add_argument("--plots", action="store_true", help="write the figures even without --group-by")
    sp.add_argument("--formats", default="pdf,png", help="comma-separated figure formats (default pdf,png)")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("neighbors", help="nearest atlas neighbours of whole GEO studies already in the atlas")
    sp.add_argument("gse", nargs="*", help="GSE accessions")
    sp.add_argument("--gse-file", help="text file with one GSE per line")
    sp.add_argument("--models", required=True, help="model bundle; its embedding is the atlas, its splits.csv gives each sample's split bucket")
    sp.add_argument("--metadata", help="sample metadata of the atlas (metadata_v2.tsv); adds strain, medium, ... of the samples")
    sp.add_argument("--out", required=True)
    sp.add_argument("-k", type=int, default=15)
    sp.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    sp.add_argument("--blocks", default="all", help="all or comma-separated block names, e.g. phase,residual")
    sp.add_argument("--within-study", action="store_true", help="allow neighbours from the query's own study")
    sp.add_argument("--umap", action="store_true", help="also fit a UMAP of the whole atlas and plot the query studies")
    sp.add_argument("--seed", type=int, default=77)
    sp.add_argument("--allow-missing", action="store_true", help="continue when some GSEs are not in the atlas")
    sp.set_defaults(func=cmd_neighbors)

    sp = sub.add_parser("view", help="write an interactive 3-D map of the atlas (one HTML file, opens in a web browser)")
    sp.add_argument("--models", required=True, help="model bundle; its embedding.tsv / predictions.tsv are the atlas")
    sp.add_argument("--metadata", help="sample metadata of the atlas (metadata_v2.tsv): labels shown and offered for colouring")
    sp.add_argument("--studies", help="study text (metadata_v2_studies.tsv from atlas_build/metadata/study_text.py): GEO title, summary, design")
    sp.add_argument("--sample-text", help="GEO sample text (metadata_v2_sample_text.tsv): sample titles, characteristics, SRA runs")
    sp.add_argument("--embeddings", action="append", metavar="DIR[=NAME]",
                    help="output directory of `embed` with your samples, drawn among the atlas samples; repeatable")
    sp.add_argument("--counts", help="count table of the atlas samples (atlas_v2_counts.tsv): adds the composition covariates "
                    "(mRNA reads, rRNA and tRNA ratios)")
    sp.add_argument("--qc", help="QC table of the atlas samples (sample_qc.tsv): adds the QC covariates of the model")
    sp.add_argument("--presets", help="JSON list of example views (default: the examples shipped with the package)")
    sp.add_argument("--no-umap", action="store_true", help="skip the UMAP layouts (they need umap-learn and take a few minutes)")
    sp.add_argument("--n-pcs", type=int, default=10, help="principal components kept per PCA layout; the page draws any three (default 10)")
    sp.add_argument("--plotly", default="auto", metavar="auto|inline|cdn|PATH",
                    help="where plotly.js comes from: inlined from the plotly Python package (auto falls back to the CDN when it "
                         "is not installed), the cdnjs CDN (smaller file, needs a network connection), or a local plotly.min.js")
    sp.add_argument("-k", type=int, default=10, help="nearest atlas samples listed for each of your samples")
    sp.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    sp.add_argument("--seed", type=int, default=77)
    sp.add_argument("--title", help="page title")
    sp.add_argument("--out", required=True, help="HTML file to write")
    sp.set_defaults(func=cmd_view)

    sp = sub.add_parser("extract", help="the samples of atlas studies as an `embed` output (embedding, calls, optionally every "
                                         "seed), for `axes --query`, `compare` and the call figures")
    sp.add_argument("gse", nargs="*", help="GSE accessions")
    sp.add_argument("--gse-file", help="text file with one GSE per line")
    sp.add_argument("--models", required=True, help="model bundle")
    sp.add_argument("--seeds", action="store_true", help="also every seed_<n> model of the bundle: seeds/ and predictions_seeds.tsv")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("axes", help="place the samples on ordered axes of the embedding (growth phase, medium, ...)")
    sp.add_argument("--models", required=True, help="bundle directory (embedding.tsv, manifest.json, splits.csv)")
    sp.add_argument("--metadata", required=True)
    sp.add_argument("--axis", action="append", required=True, metavar="NAME:COLUMN:CAT1,CAT2,...",
                    help="an axis: a metadata column and its values in order; may be given several times. A category may "
                    "group values as label=value1+value2")
    sp.add_argument("--holdout", action="append", metavar="CATEGORY",
                    help="a category left out of the fits and only scored (extrapolation or interpolation test); repeatable")
    sp.add_argument("--where", action="append", metavar="COLUMN=VALUE[,VALUE]",
                    help="only samples with these metadata values get a category; repeatable")
    sp.add_argument("--blocks", default="all", help="embedding blocks to use, comma-separated (default: the whole embedding)")
    sp.add_argument("--series", help="ordered series, e.g. time courses, drawn as trajectories: a table (sample, series, x[, x_label]), or "
                    "PATH:SERIES_COLUMN:X_COLUMN[:PANEL_COLUMN] for any sample table (first column the sample id), e.g. "
                    "samples.csv:replicate:time_h; a panel column (also a column `panel` of the table) draws one panel per value")
    sp.add_argument("--series-x", default="auto", choices=["auto", "categorical", "linear", "log"],
                    help="x axis of the series figures: the positions as categories (labelled with their values, the series side "
                    "by side), linear or log; auto (default): categorical with at most 12 positions in a panel, else log when "
                    "they span more than 20-fold, else linear")
    sp.add_argument("--query", help="new samples to place on the axes: an `embed` output directory (or its embedding.tsv) made with the "
                    "same bundle; they are only scored, by the model of all fitted samples, and drawn over the map of two axes. With the "
                    "directory's predictions.tsv the calls (probabilities, 0-1) are drawn too: fig_heads_series, fig_heads_phase_by_medium")
    sp.add_argument("--query-study", default="query", help="study label of the --query samples (default: query)")
    sp.add_argument("--exclude-study", action="append", metavar="GSE",
                    help="leave the atlas samples of this study out of the fits and the tables (repeatable), e.g. the study the "
                    "--query samples come from, so that it stays unseen")
    sp.add_argument("--seeds", action="store_true",
                    help="also fit every axis on the embedding of each seed_<n> model of the bundle: <method>_seed_median/min/max in "
                    "the score tables, axis_<name>_seed_scores.tsv, and every seed's score as small dots in the series figures. --query samples must have been "
                    "embedded with `embed --seeds`. The spread is the dependence on training randomness only")
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
    sp.add_argument("--modules", metavar="TSV", help="gsea panel: also the sets of a module table (atlas_data/modules/modules.tsv "
                    "from `ecoli_atlas download modules`), one scatter per database (RegulonDB, KEGG pathway, KEGG module)")
    sp.add_argument("--tracking-x", choices=("ribosome", "phase"), default="ribosome",
                    help="batch_tracking panel: x axis, the mean log2 expression of the ribosomal proteins (default) or "
                    "the study's mean phase ordinal score")
    sp.add_argument("--counts", help="joined count table of the atlas")
    sp.add_argument("--qc", help="QC table (sample_qc.tsv)")
    sp.add_argument("--antisense", help="antisense count table of the atlas, when the bundle was trained with one")
    sp.add_argument("--features", help="feature table of the counting annotation")
    sp.add_argument("--model-name", help="use only the samples whose in_model has this value (default: any)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--only", help="comma-separated figure names")
    sp.add_argument("--formats", default="pdf,png", help="comma-separated figure formats (default pdf,png)")
    sp.add_argument("--strict", action="store_true", help="exit 1 when a figure fails or the embedding or the TEST metrics "
                                                          "differ from those stored in the bundle")
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_gsea(sp)
    add_device(sp)
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("casestudy", help="case studies on the embedding: uti, oxygen, glu-cas")
    sp.add_argument("study", choices=list(CASE_STUDIES) + ["all"])
    add_tables(sp)
    sp.add_argument("--models", required=True, help="model bundle")
    sp.add_argument("--oxygen-studies", default=",".join(OXYGEN_CASE_STUDIES), help="comma-separated studies of the oxygen case study")
    sp.add_argument("--gsea", action="store_true", help="oxygen, uti: preranked GSEA of the genes along the axis against GO terms "
                    "(<name>_gsea_go.tsv, fig_<name>_gsea_go)")
    sp.add_argument("--modules", metavar="TSV", help="oxygen, uti: the same against a curated module table (atlas_data/modules/modules.tsv "
                    "from `ecoli_atlas download modules`: KEGG pathways and modules, RegulonDB regulons, iModulons; <name>_gsea_modules.tsv, one figure per database)")
    sp.add_argument("--n-perm", type=int, default=1000, help="permutations of the paired tests")
    sp.add_argument("--out", required=True)
    sp.add_argument("--formats", default="pdf,png")
    sp.add_argument("--threads", type=int, help="torch CPU threads")
    add_gsea(sp)
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
