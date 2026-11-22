#!/usr/bin/env python3
"""Remove what killed jobs left behind in an inGEST run directory, before the run is resumed.

A job that is killed (scancel, time limit, node failure) leaves its partly written outputs on disk.
Snakemake records such outputs in `.snakemake/incomplete/` (file name = URL-safe base64 of the
output path), but in the atlas recount it took them for finished files on resume in spite of
`--rerun-incomplete`: truncated .gam files were surjected and went on down the workflow.

Deleting the marked file alone is not enough, because most intermediates are temp(): once a
temporary file is gone, Snakemake does not redo it as long as the files made from it exist. So for
each marked output:

    results/<sample>/{vg,rseqc,coverage,feature_overlap,qc}/...
        the sample's alignment and everything made from it is removed (those directories; in qc/
        the pre-alignment `_species_early.json` is kept)
    results/<sample>/<anything else>
        the whole sample directory is removed (the sample starts again from the download)
    any other path (results/<file>, checkpoints/..., sra/...)
        that path is removed

The markers are removed as well. Paths that are absolute or leave the run directory are skipped.

    clear_incomplete.py RUN_DIR [--results-dir results] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import sys

ALIGNMENT_DIRS = ("vg", "rseqc", "coverage", "feature_overlap", "qc")
KEEP_SUFFIX = "_species_early.json"


def marked_outputs(run_dir: str) -> list[tuple[str, str]]:
    """(marker file, output path) for every incomplete marker; long ids are split over
    directories whose names start with '@'."""
    root = os.path.join(run_dir, ".snakemake", "incomplete")
    found = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            marker = os.path.join(dirpath, name)
            parts = os.path.relpath(marker, root).split(os.sep)
            b64 = "".join(p[1:] if p.startswith("@") else p for p in parts)
            try:
                found.append((marker, base64.urlsafe_b64decode(b64).decode()))
            except (ValueError, UnicodeDecodeError):
                continue  # not a marker (e.g. `migration_underway`)
    return sorted(found, key=lambda x: x[1])


def targets(path: str, results_dir: str) -> list[str]:
    """Paths (relative to the run directory) to remove for one marked output."""
    parts = path.split("/")
    if parts[0] != results_dir or len(parts) < 3:
        return [path]
    sample_dir = "/".join(parts[:2])
    if parts[2] in ALIGNMENT_DIRS and len(parts) > 3:
        return [f"{sample_dir}/{d}" for d in ALIGNMENT_DIRS]
    return [sample_dir]


def remove(run_dir: str, rel: str, dry_run: bool) -> int:
    """Remove a file or directory; in a qc/ directory keep the pre-alignment record. Returns the
    number of files removed."""
    full = os.path.join(run_dir, rel)
    if not os.path.lexists(full):
        return 0
    if os.path.isdir(full) and not os.path.islink(full):
        if os.path.basename(rel) != "qc":
            n = sum(len(f) for _, _, f in os.walk(full))
            if not dry_run:
                shutil.rmtree(full)
            return n
        n = 0
        for name in os.listdir(full):
            if not name.endswith(KEEP_SUFFIX):
                n += remove(run_dir, f"{rel}/{name}", dry_run)
        return n
    if not dry_run:
        os.remove(full)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true", help="report only")
    args = ap.parse_args()

    marks = marked_outputs(args.run_dir)
    todo: dict[str, None] = {}
    for _, path in marks:
        norm = os.path.normpath(path)
        if os.path.isabs(norm) or norm.startswith(".."):
            print(f"clear_incomplete: skipped (outside the run directory): {path}", file=sys.stderr)
            continue
        for t in targets(norm, args.results_dir):
            todo[t] = None
    n_files = 0
    for rel in todo:
        n = remove(args.run_dir, rel, args.dry_run)
        n_files += n
        if n:
            print(f"clear_incomplete: {'would remove' if args.dry_run else 'removed'} {rel} ({n} file(s))")
    if not args.dry_run:
        for marker, _ in marks:
            os.remove(marker)
    print(f"clear_incomplete: {len(marks)} incomplete output(s) marked, {n_files} file(s) "
          f"{'to remove' if args.dry_run else 'removed'} in {os.path.abspath(args.run_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
