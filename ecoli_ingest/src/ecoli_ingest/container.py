"""Execution mode (conda env vs. container), bind mounts, and command assembly."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from .paths import ENV_RUNTIME

# Every executable the Snakefile and its scripts invoke.
TOOLS_REQUIRED = (
    "snakemake", "python", "vg", "fastp", "fastqc", "centrifuge", "centrifuge-kreport",
    "featureCounts", "infer_experiment.py", "geneBody_coverage.py", "bedtools", "samtools",
    "multiqc", "Rscript",
)
MODES = ("auto", "always", "never")


def find_runtime() -> str | None:
    env = os.environ.get(ENV_RUNTIME)
    if env:
        return env
    for name in ("apptainer", "singularity"):
        found = shutil.which(name)
        if found:
            return found
    return None


def missing_tools(tools: Iterable[str] = TOOLS_REQUIRED) -> list[str]:
    return [t for t in tools if shutil.which(t) is None]


def resolve_mode(mode: str) -> str:
    """Turn auto/always/never into always/never."""
    if mode not in MODES:
        raise ValueError(f"unknown container mode {mode!r}")
    if mode != "auto":
        return mode
    return "never" if not missing_tools(("snakemake", "vg")) else "always"


def compute_binds(paths: Iterable[str | os.PathLike]) -> list[str]:
    """Canonical, de-duplicated bind list. Files are replaced by their parent directory and
    paths nested inside another bind are dropped."""
    dirs: list[Path] = []
    for p in paths:
        if p is None:
            continue
        rp = Path(os.path.realpath(p))
        if rp.is_file():
            rp = rp.parent
        if not rp.exists():
            continue
        if rp not in dirs:
            dirs.append(rp)
    dirs.sort(key=lambda d: len(d.parts))
    kept: list[Path] = []
    for d in dirs:
        if not any(d == k or k in d.parents for k in kept):
            kept.append(d)
    return [str(d) for d in kept]


def snakemake_command(*, snakefile: Path, rundir: Path, configfile: Path, resources: Path,
                      cores: str = "all", targets: Sequence[str] = ("all",), dry_run: bool = False,
                      extra: Sequence[str] = (), resource_limits: dict[str, int] | None = None) -> list[str]:
    cmd = [
        "snakemake",
        "--snakefile", str(snakefile),
        "--directory", str(rundir),
        "--configfile", str(configfile),
        "--config", f"resources_dir={resources}",
    ]
    if resource_limits:
        # e.g. download_slots=N caps concurrent `sra_download` jobs (see the Snakefile)
        cmd.append("--resources")
        cmd.extend(f"{k}={v}" for k, v in resource_limits.items())
    if not any(a == "--rerun-triggers" or a.startswith("--rerun-triggers=") for a in extra):
        # Rerun a step only when its output is missing or older than its input. Snakemake's default
        # also reruns steps whose rule code or parameters changed, so upgrading ecoli_ingest (or
        # editing an unrelated config key) would re-download and re-align a finished run directory.
        # Pass `-- --rerun-triggers ...` to choose other triggers, `-- --forcerun RULE` to redo a step.
        cmd += ["--rerun-triggers", "mtime"]
    cmd += ["--cores", str(cores), "--rerun-incomplete"]
    if dry_run:
        cmd.append("--dry-run")
    # targets go before any pass-through options: several snakemake options take a variable
    # number of values (e.g. --rerun-triggers) and would otherwise swallow the target name
    cmd.extend(targets)
    cmd.extend(extra)
    return cmd


def container_command(runtime: str, image: Path, binds: Sequence[str], pwd: Path,
                      inner: Sequence[str]) -> list[str]:
    cmd = [runtime, "exec"]
    for b in binds:
        cmd += ["--bind", b]
    cmd += ["--pwd", str(pwd), str(image)]
    cmd += list(inner)
    return cmd
