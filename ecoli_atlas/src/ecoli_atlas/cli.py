"""Command-line entry point: `ecoli_atlas <subcommand>`."""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__, download as dl
from .utils import log, pick_device, setup_logging


def _info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _feature_table(path_arg):
    from .io import read_feature_table
    p = Path(path_arg) if path_arg else dl.default_feature_table()
    if p is None:
        return None
    return read_feature_table(p)


# ---------------------------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    from .prepare import prepare
    counts, samples = prepare(args.ingest_results, args.counts, args.metadata, args.out)
    _info(f"wrote {Path(args.out) / 'counts.tsv'} ({counts.shape[0]} genes x {counts.shape[1]} samples) and samples.tsv")
    if args.models:
        from .artifacts import ModelBundle
        from .preprocess import GeneMap, align_to_genes, clean_counts
        bundle = ModelBundle.load(args.models)
        feat = _feature_table(args.feature_table)
        X = clean_counts(counts, GeneMap.from_feature_table(feat)) if feat is not None else counts.T
        _, cov = align_to_genes(X, bundle.genes)
        cov.to_frame(bundle.genes).to_csv(Path(args.out) / "gene_coverage.tsv", sep="\t")
        _info(f"gene coverage: {cov.n_present}/{cov.n_model_genes} model genes present ({cov.fraction:.1%})")
    return 0


def cmd_embed(args) -> int:
    from .artifacts import ModelBundle
    from .embed import embed_counts
    from .io import read_counts
    from .preprocess import GeneMap
    device = pick_device(args.device)
    bundle = ModelBundle.load(args.models, device)
    counts = read_counts(args.counts, strip_sample_names=not args.keep_sample_names)
    feat = _feature_table(args.feature_table)
    if feat is None:
        _info("warning: no feature table given or found; rRNA/tRNA are not removed and locus tags are not "
              "renamed (fine if the table already uses gene symbols without rRNA/tRNA)")
    gm = GeneMap.from_feature_table(feat) if feat is not None else None
    res = embed_counts(counts, bundle, gm, apply_thresholds=tuple(args.apply_thresholds or ()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cov = res.coverage
    _info(f"gene alignment: {cov.n_present}/{cov.n_model_genes} model genes present "
          f"({len(cov.missing)} filled with 0)")
    cov.to_frame(bundle.genes).to_csv(out / "gene_coverage.tsv", sep="\t")
    if cov.fraction < args.min_gene_coverage:
        _info(f"error: gene coverage {cov.fraction:.3f} below --min-gene-coverage {args.min_gene_coverage}; "
              f"see {out / 'gene_coverage.tsv'}. Are the gene identifiers symbols or b-numbers of E. coli K-12?")
        return 1
    res.embedding.to_csv(out / "final_embedding.csv")
    res.predictions.to_csv(out / "sample_predictions.csv")
    res.block_layout.to_csv(out / "embedding_block_layout.csv", index=False)
    _info(f"wrote {out / 'final_embedding.csv'} {res.embedding.shape}, sample_predictions.csv, embedding_block_layout.csv")
    for f in bundle.factor_order:
        counts_by = res.predictions[f"{f}_pred"].value_counts().to_dict()
        _info(f"  {f}: {counts_by}")
    return 0


def cmd_compare(args) -> int:
    from .compare import Atlas, compare
    atlas = Atlas.load(args.atlas or dl.default_atlas_dir())
    query = pd.read_csv(args.embeddings, index_col=0)
    preds = None
    pred_path = Path(args.predictions) if args.predictions else Path(args.embeddings).parent / "sample_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path, index_col=0)
    blocks = "all" if args.blocks in (None, "all") else tuple(args.blocks.split(","))
    metadata = None
    if args.metadata:
        from .io import read_metadata
        metadata = read_metadata(args.metadata)
    group_by = None
    if args.group_by:
        parts = tuple(c.strip() for c in args.group_by.split(",") if c.strip())
        group_by = parts[0] if len(parts) == 1 else parts
    try:
        summary = compare(query, atlas, args.out, k=args.k, metric=args.metric, blocks=blocks, query_predictions=preds,
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


def _write_atlas(bundle, counts, meta, feat, atlas_dir: Path, apply_thresholds, device_name: str) -> None:
    from .embed import embed_counts
    from .preprocess import GeneMap
    gm = GeneMap.from_feature_table(feat)
    res = embed_counts(counts, bundle, gm, apply_thresholds=apply_thresholds)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    res.embedding.to_csv(atlas_dir / "final_embedding.csv")
    res.predictions.to_csv(atlas_dir / "sample_predictions.csv")
    res.block_layout.to_csv(atlas_dir / "embedding_block_layout.csv", index=False)
    am = meta.copy()
    am.index = am.index.astype(str).str.strip()
    am = am.reindex(res.embedding.index)
    am.index.name = "sample"
    am.to_csv(atlas_dir / "atlas_metadata.csv")
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "ecoli_atlas_version": __version__, "atlas_version": dl.ATLAS_DATA_VERSION,
                "n_samples": int(len(res.embedding)), "n_with_metadata": int(am["gse"].notna().sum()) if "gse" in am else None,
                "embedding_dim": int(res.embedding.shape[1]), "device": device_name,
                "model_manifest_created_utc": (bundle.manifest or {}).get("created_utc")}
    (atlas_dir / "atlas_manifest.json").write_text(json.dumps(manifest, indent=2))
    _info(f"wrote atlas ({len(res.embedding)} samples) to {atlas_dir}")


def cmd_train(args) -> int:
    import copy
    from .artifacts import ModelBundle
    from .config import TrainConfig
    from .data import build_training_data
    from .io import read_counts, read_metadata
    from .splits import load_splits
    from .sweep import promote_bundle, seed_score, select_seed, summarize_sweep, write_sweep
    from .train import reload_check, run_training

    device = pick_device(args.device)
    if args.hparams:
        cfg = TrainConfig.from_manifest(args.hparams)
        _info(f"hyper-parameters and split configuration taken from {args.hparams}")
    else:
        cfg = TrainConfig()
    for name in ("seed", "factor_epochs", "top_epochs", "min_class_n", "withheld_study", "test_frac", "n_folds"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if args.factors:
        cfg.factors = tuple(args.factors.split(","))
    if args.case_studies is not None:
        cfg.case_studies = tuple(s for s in args.case_studies.split(",") if s)
    cfg.tune = args.tune
    if args.n_trials is not None:
        cfg.n_trials_factor = cfg.n_trials_top = args.n_trials
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else [cfg.seed]
    if len(set(seeds)) != len(seeds):
        _info("error: --seeds contains duplicates")
        return 1
    if len(seeds) > 1 and args.select != "first" and args.select_on == "cv" and args.no_thresholds:
        _info("error: ranking seeds on CV needs the out-of-fold refits; drop --no-thresholds or use --select first")
        return 1

    counts = read_counts(args.counts, strip_sample_names=False)
    meta = read_metadata(args.metadata)
    feat = _feature_table(args.feature_table)
    if feat is None:
        _info("error: --feature-table is required (or put it in $ECOLI_ATLAS_DATA)")
        return 1
    splits = load_splits(args.splits) if args.splits else None
    # the split is built once (with cfg.seed) and shared by every seed of the sweep
    data = build_training_data(counts, meta, feat, cfg, splits=splits, max_samples=args.max_samples, log=_info)
    _info(f"device: {device}")
    out = Path(args.out)

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    sweep_dirs, manifests = {}, {}
    for seed in seeds:
        dest = out / f"seed_{seed}" if len(seeds) > 1 else out
        if args.resume and (dest / "manifest.json").exists():
            reloaded = ModelBundle.load(dest, device)
            _info(f"===== seed {seed}: already trained, reusing {dest} =====")
            sweep_dirs[seed] = dest
            manifests[seed] = reloaded.manifest
            continue
        cfg_s = copy.deepcopy(cfg)
        cfg_s.seed = seed
        if len(seeds) > 1:
            _info(f"===== seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) =====")
        result = run_training(data, cfg_s, device, log=_info, compute_thresholds=not args.no_thresholds,
                              training_data_version=args.training_data_version)
        result.bundle.save(dest, splits=data.splits_frame())
        reloaded = ModelBundle.load(dest, device)
        dev = reload_check(reloaded, data, result.embedding)
        _info(f"saved bundle to {dest} (reload check max deviation {dev:.2e})")
        sweep_dirs[seed] = dest
        manifests[seed] = reloaded.manifest
    if len(seeds) > 1:
        summary = summarize_sweep(manifests)
        try:
            selected = select_seed(manifests, args.select, args.select_on)
        except ValueError as exc:
            _info(f"error: {exc}")
            return 1
        summary["selected_seed"] = selected
        summary["selection_rule"] = args.select
        summary["selection_metric"] = None if args.select == "first" else f"mean {args.select_on} AUROC"
        promote_bundle(sweep_dirs[selected], out, summary)
        write_sweep(summary, out)
        _info(f"seed sweep: selected seed {selected} ({args.select}"
              + ("" if args.select == "first" else f" by mean {args.select_on.upper()} AUROC "
                 f"= {seed_score(manifests[selected], args.select_on):.3f}") + ")")
        _info("per-factor balanced accuracy spread (CV = out-of-fold on the training pool):")
        for f in summary["spread"]:
            for label, block in (("CV", summary["cv_spread"]), ("TEST", summary["spread"])):
                ba = block[f]["balanced_acc"]
                if ba.get("n"):
                    _info(f"  {f:14s} {label:4s} min={ba['min']:.3f} mean={ba['mean']:.3f} max={ba['max']:.3f} sd={ba['sd']:.3f}")
        _info(f"promoted {sweep_dirs[selected]} -> {out}; details in {out / 'seed_sweep.tsv'}")
        reloaded = ModelBundle.load(out, device)

    if args.write_atlas:
        atlas_dir = Path(args.atlas_out) if args.atlas_out else out / "atlas"
        _write_atlas(reloaded, counts, meta, feat, atlas_dir, tuple(cfg.apply_thresholds), str(device))
    return 0


def cmd_verify(args) -> int:
    from .io import read_counts, read_metadata
    from .verify import verify_models
    counts = read_counts(args.counts, strip_sample_names=False)
    feat = _feature_table(args.feature_table)
    if feat is None:
        _info("error: --feature-table is required (or put it in $ECOLI_ATLAS_DATA)")
        return 1
    meta = read_metadata(args.metadata) if args.metadata else None
    report = verify_models(args.old, args.new, counts, feat, meta, device=pick_device(args.device), k=args.k, log=_info)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    return 0


def cmd_download(args) -> int:
    rc = 0
    bundles = list(dl.BUNDLES) if "all" in args.bundles else args.bundles
    for b in bundles:
        rc = max(rc, dl.download(b, Path(args.dest) if args.dest else None, dry_run=args.dry_run))
    return rc


def _training_tables(args):
    """(counts, metadata, feature table) for commands that work on the training corpus."""
    from .io import read_counts, read_metadata
    counts = read_counts(args.counts, strip_sample_names=False)
    meta = read_metadata(args.metadata)
    feat = _feature_table(args.feature_table)
    if feat is None:
        raise SystemExit("error: --feature-table is required (or put it in $ECOLI_ATLAS_DATA)")
    return counts, meta, feat


def _set_threads(n):
    if n:
        import torch
        torch.set_num_threads(int(n))


def cmd_figures(args) -> int:
    from .figures import FIGURES
    from .figures.run import run_figures
    if args.list:
        for name, spec in FIGURES.items():
            flag = "  [refits fold models; skipped with --no-retrain]" if spec.needs_retrain else ""
            print(f"{name:24s} {spec.doc}{flag}\n{'':24s} -> {', '.join(spec.outputs)}")
        return 0
    for req in ("counts", "metadata", "out"):
        if getattr(args, req) is None:
            _info(f"error: --{req} is required")
            return 2
    _set_threads(args.threads)
    counts, meta, feat = _training_tables(args)
    only = args.only.split(",") if args.only else None
    skip = args.skip.split(",") if args.skip else None
    formats = tuple(f for f in args.formats.split(",") if f)
    side, n_failed = run_figures(counts, meta, feat, args.models, args.out, only=only, skip=skip, formats=formats,
                                 seed=args.seed, device=pick_device(args.device), retrain=not args.no_retrain,
                                 force_retrain=args.force_retrain, recombat=args.recombat, atlas_dir=args.atlas,
                                 reference_sweep=args.reference_sweep, log=_info)
    for k, c in side["checks"].items():
        status = "pass" if c["pass"] else ("n/a" if c["pass"] is None else "FAIL")
        _info(f"  check {k:36s} {status:4s} value={c['value']} reference={c['reference']} tol={c['tolerance']}")
    _info(f"wrote {len(side['figures']) - n_failed} figure sets to {args.out} ({side['total_seconds']}s); "
          f"details in {Path(args.out) / 'figure_metrics.json'}")
    if n_failed:
        _info(f"error: {n_failed} figure(s) failed: {side['failed']}")
        return 1
    if args.strict and not side["all_checks_pass"]:
        _info("error: --strict and at least one check against the manifest failed")
        return 1
    return 0


def cmd_casestudy(args) -> int:
    from .casestudy.run import run_casestudies
    _set_threads(args.threads)
    rc = run_casestudies(args, log=_info)
    return rc


def cmd_neighbors(args) -> int:
    from .neighbors import run_neighbors
    gses = list(args.gse or [])
    if args.gse_file:
        gses += [ln.strip().split()[0] for ln in Path(args.gse_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not gses:
        _info("error: give at least one GSE accession (positional or --gse-file)")
        return 2
    blocks = "all" if args.blocks in (None, "all") else tuple(args.blocks.split(","))
    try:
        report = run_neighbors(gses, args.atlas or dl.default_atlas_dir(), args.out, models_dir=args.models, k=args.k,
                               metric=args.metric, blocks=blocks, exclude_same_study=not args.within_study,
                               collapse_runs=not args.no_collapse_runs, do_umap=args.umap, seed=args.seed,
                               allow_missing=args.allow_missing, log=_info)
    except ValueError as exc:
        _info(f"error: {exc}")
        return 1
    _info(f"wrote neighbors.tsv, summary.tsv ({report['n_query']} query samples) and study_summary.tsv "
          f"({len(report['found'])} studies) to {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecoli_atlas",
                                description="Ecoli_ATLAS: embed E. coli RNA-seq samples and compare them with the atlas.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"ecoli_atlas {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_device(sp):
        sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    sp = sub.add_parser("prepare", help="convert Ecoli_inGEST results into the atlas count-table format")
    sp.add_argument("--ingest-results", metavar="DIR", help="an <experiment>_results directory")
    sp.add_argument("--counts", metavar="FILE", help="a count table instead of --ingest-results")
    sp.add_argument("--metadata", metavar="CSV", help="optional sample metadata (first column = sample id)")
    sp.add_argument("--models", metavar="DIR", help="model bundle; if given, also write gene_coverage.tsv")
    sp.add_argument("--feature-table", metavar="FILE")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train", help="train the factor encoders and the residual autoencoder")
    sp.add_argument("--counts", required=True, help="genes x samples count table (b-numbers or symbols)")
    sp.add_argument("--metadata", required=True, help="metadata_revised.csv-style table")
    sp.add_argument("--feature-table", help="NCBI feature table (default: $ECOLI_ATLAS_DATA/...)")
    sp.add_argument("--out", required=True, help="models directory to write")
    sp.add_argument("--hparams", metavar="manifest.json", help="take hyper-parameters and split config from a manifest")
    sp.add_argument("--splits", metavar="splits.csv", help="reuse a saved study-level split instead of searching")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--tune", action="store_true", help="run the Optuna searches (slow; GPU recommended)")
    g.add_argument("--no-tune", dest="tune", action="store_false")
    sp.set_defaults(tune=False)
    sp.add_argument("--n-trials", type=int)
    sp.add_argument("--factor-epochs", type=int)
    sp.add_argument("--top-epochs", type=int)
    sp.add_argument("--factors", help="comma-separated (default growth_phase,medium_type,is_lab_strain)")
    sp.add_argument("--withheld-study")
    sp.add_argument("--case-studies", help="comma-separated GSEs forced into TEST")
    sp.add_argument("--test-frac", type=float)
    sp.add_argument("--n-folds", type=int)
    sp.add_argument("--min-class-n", type=int)
    sp.add_argument("--seed", type=int, help="seed for the split search and for a single fit (default 77)")
    sp.add_argument("--seeds", help="comma-separated seeds: fit each on the same split, save them under <out>/seed_<s>/, "
                                    "promote one to <out>/ and record the spread of CV and TEST metrics in its manifest")
    sp.add_argument("--select", default="best", choices=["best", "median", "first"],
                    help="which seed to promote in a sweep: best = highest mean AUROC (see --select-on), "
                         "median = closest to the median, first")
    sp.add_argument("--select-on", default="cv", choices=["cv", "test"],
                    help="metric the seeds are ranked on: cv = study-level out-of-fold AUROC on the training pool "
                         "(default; keeps TEST untouched), test = TEST AUROC (biased; only to reproduce old sweeps)")
    sp.add_argument("--resume", action="store_true",
                    help="in a sweep, reuse seeds whose bundle already exists under <out>/seed_<s>/ instead of "
                         "retraining (continue a sweep after a job hit its wall-time limit)")
    sp.add_argument("--threads", type=int, help="torch CPU threads (default: all cores)")
    sp.add_argument("--max-samples", type=int, help="(development) random subset of samples")
    sp.add_argument("--no-thresholds", action="store_true", help="skip the out-of-fold threshold refits (use 0.5)")
    sp.add_argument("--write-atlas", action="store_true", help="also embed the whole count table as the atlas")
    sp.add_argument("--atlas-out", help="atlas directory (default: <out>/atlas; never overwrites an existing sibling)")
    sp.add_argument("--training-data-version", default=dl.ATLAS_DATA_VERSION)
    add_device(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("embed", help="embed new samples with a trained bundle")
    sp.add_argument("--counts", required=True, help="count table (inGEST output or genes x samples TSV)")
    sp.add_argument("--models", default=None, help=f"model bundle directory (default: ${dl.ENV_MODELS} or ./outputs/models)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--feature-table", help="NCBI feature table (default: $ECOLI_ATLAS_DATA/...)")
    sp.add_argument("--min-gene-coverage", type=float, default=0.9)
    sp.add_argument("--apply-thresholds", nargs="*", metavar="FACTOR",
                    help="use the saved operating threshold instead of argmax for these factors")
    sp.add_argument("--keep-sample-names", action="store_true", help="do not strip BAM paths from column names")
    add_device(sp)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("compare", help="nearest atlas neighbours, label votes, novelty, optional UMAP")
    sp.add_argument("--embeddings", required=True, help="final_embedding.csv of the new samples")
    sp.add_argument("--atlas", help=f"atlas directory (default: ${dl.ENV_MODELS}/../atlas or ./outputs/atlas)")
    sp.add_argument("--predictions", help="sample_predictions.csv of the new samples (default: next to --embeddings)")
    sp.add_argument("--out", required=True)
    sp.add_argument("-k", type=int, default=15)
    sp.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    sp.add_argument("--blocks", default="all", help="all or comma-separated block names, e.g. growth_phase,residual")
    sp.add_argument("--umap", action="store_true")
    sp.add_argument("--seed", type=int, default=77)
    sp.add_argument("--metadata", help="table of the new samples (first column = sample id; CSV, or the samples.tsv "
                                       "of `prepare`); its columns are added to summary.tsv")
    sp.add_argument("--group-by", help="metadata column(s), comma-separated: write group_summary.tsv and the group figures")
    sp.add_argument("--order-by", help="metadata column that orders the groups (numeric when it parses, e.g. a time)")
    sp.add_argument("--color-by", help="metadata column that colours the new samples in the UMAP (default: the groups)")
    sp.add_argument("--plots", action="store_true", help="write the figures even without --group-by")
    sp.add_argument("--formats", default="pdf,png", help="comma-separated figure formats (default pdf,png)")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("verify-models", help="compare two model bundles (e.g. original vs. retrained)")
    sp.add_argument("--old", required=True)
    sp.add_argument("--new", required=True)
    sp.add_argument("--counts", required=True)
    sp.add_argument("--feature-table")
    sp.add_argument("--metadata")
    sp.add_argument("-k", type=int, default=15)
    sp.add_argument("--report", help="write the JSON report here")
    add_device(sp)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("download", help="download data bundles")
    sp.add_argument("bundles", nargs="+", choices=sorted(dl.BUNDLES) + ["all"])
    sp.add_argument("--dest")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_download)

    def add_training_inputs(sp, required=True):
        sp.add_argument("--counts", required=required, help="training count table (merged_count_table.txt)")
        sp.add_argument("--metadata", required=required, help="metadata_revised.csv")
        sp.add_argument("--feature-table", help="NCBI feature table (default: $ECOLI_ATLAS_DATA/...)")
        sp.add_argument("--models", default=None, help=f"model bundle (default: ${dl.ENV_MODELS} or ./outputs/models)")
        sp.add_argument("--out", required=required, help="output directory (never outputs/figures or the bundle)")
        sp.add_argument("--formats", default="pdf,png", help="comma-separated figure formats (default pdf,png)")
        sp.add_argument("--seed", type=int, default=None, help="seed for UMAP / PCA / refits (default: the bundle's seed)")
        sp.add_argument("--threads", type=int, help="torch CPU threads")
        add_device(sp)

    sp = sub.add_parser("figures", help="regenerate the paper figures from a bundle and the training tables")
    add_training_inputs(sp, required=False)
    sp.add_argument("--only", help="comma-separated figure names (see --list)")
    sp.add_argument("--skip", help="comma-separated figure names to skip")
    sp.add_argument("--list", action="store_true", help="list the figures and their outputs, then exit")
    sp.add_argument("--no-retrain", action="store_true", help="skip figures that refit fold models (threshold_curves)")
    sp.add_argument("--force-retrain", action="store_true", help="recompute oof_scores.csv even if it exists in --out")
    sp.add_argument("--recombat", action="store_true", help="include the reComBat bar in study_leakage (needs reComBat installed)")
    sp.add_argument("--atlas", help="atlas directory to compare the regenerated embedding with (e.g. outputs/atlas)")
    sp.add_argument("--reference-sweep", metavar="manifest.json",
                    help="manifest of a seed sweep (e.g. outputs/models_repro_sweep/manifest.json); its CV spread "
                         "is the acceptance range for the out-of-fold refits of threshold_curves")
    sp.add_argument("--strict", action="store_true", help="exit 1 when a check against the bundle's manifest fails")
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("casestudy", help="case studies: uti, oxygen, glu-cas or all (uti + oxygen)")
    sp.add_argument("which", choices=["uti", "oxygen", "glu-cas", "all"])
    add_training_inputs(sp, required=False)
    sp.add_argument("--case-metadata", help="glu-cas: the metadata of the inference count table (metadata_paper1.csv layout)")
    sp.add_argument("--oxygen-studies", help="oxygen: comma-separated GSEs (default GSE189154,GSE78756)")
    sp.set_defaults(func=cmd_casestudy)

    sp = sub.add_parser("neighbors", help="nearest atlas neighbours of whole GEO studies already in the atlas")
    sp.add_argument("gse", nargs="*", help="GSE accessions")
    sp.add_argument("--gse-file", help="text file with one GSE per line")
    sp.add_argument("--atlas", help=f"atlas directory (default: ${dl.ENV_MODELS}/../atlas or ./outputs/atlas)")
    sp.add_argument("--models", help="model bundle; its splits.csv / manifest.json give each sample's split bucket")
    sp.add_argument("--out", required=True)
    sp.add_argument("-k", type=int, default=15)
    sp.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    sp.add_argument("--blocks", default="all", help="all or comma-separated block names, e.g. growth_phase,residual")
    sp.add_argument("--within-study", action="store_true", help="allow neighbours from the query's own study")
    sp.add_argument("--no-collapse-runs", action="store_true", help="keep one row per SRR run instead of averaging runs per GSM")
    sp.add_argument("--umap", action="store_true", help="also fit a UMAP of the whole atlas and plot the query studies")
    sp.add_argument("--seed", type=int, default=77)
    sp.add_argument("--allow-missing", action="store_true", help="continue when some GSEs are not in the atlas")
    sp.set_defaults(func=cmd_neighbors)
    return p


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.command in ("embed", "figures", "casestudy") and getattr(args, "models", None) is None:
        args.models = str(dl.default_models_dir())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
