"""Dispatcher behind ``ecoli_atlas casestudy``."""
from __future__ import annotations

import json
from pathlib import Path

from ..figures.plotting import apply_style
from ..figures.run import refuse_reference_dir

METRICS_FILE = "casestudy_metrics.json"


def run_casestudies(args, log=None) -> int:
    log = log or (lambda *_: None)
    which = args.which
    formats = tuple(f for f in args.formats.split(",") if f)
    out = Path(args.out) if args.out else None
    if out is None:
        log("error: --out is required")
        return 2
    models_dir = Path(args.models)
    refuse_reference_dir(out, models_dir)
    apply_style()
    results = {}
    if which == "glu-cas":
        if not (args.counts and args.case_metadata):
            log("error: glu-cas needs --counts (the inference count table) and --case-metadata")
            return 2
        from ..artifacts import ModelBundle
        from ..io import read_counts, read_metadata
        from ..utils import pick_device
        from .glu_cas import run_glu_cas
        from ..cli import _feature_table
        bundle = ModelBundle.load(models_dir, pick_device(args.device))
        counts = read_counts(args.counts)   # BAM paths stripped to the SRR run ids used by the metadata
        results["glu_cas"] = run_glu_cas(counts, read_metadata(args.case_metadata), bundle, _feature_table(args.feature_table),
                                         out, formats, seed=args.seed if args.seed is not None else 77, log=log)
    else:
        if not (args.counts and args.metadata):
            log(f"error: {which} needs --counts and --metadata (the training tables)")
            return 2
        from ..cli import _training_tables
        from ..figures.context import build_context
        from ..utils import pick_device
        counts, meta, feat = _training_tables(args)
        ctx = build_context(counts, meta, feat, models_dir, device=pick_device(args.device), seed=args.seed, log=log)
        if which in ("uti", "all"):
            from .uti import run_uti
            results["uti"] = run_uti(ctx, out, formats, log=log)
        if which in ("oxygen", "all"):
            from .oxygen import OXYGEN_CASE_STUDIES, run_oxygen
            studies = tuple(s for s in args.oxygen_studies.split(",") if s) if getattr(args, "oxygen_studies", None) else OXYGEN_CASE_STUDIES
            results["oxygen"] = run_oxygen(ctx, out, formats, studies=studies, log=log)
    p = out / METRICS_FILE
    existing = json.loads(p.read_text()) if p.exists() else {}
    existing.update(results)
    p.write_text(json.dumps(existing, indent=2, default=str))
    log(f"wrote case-study figures and {p}")
    return 0
