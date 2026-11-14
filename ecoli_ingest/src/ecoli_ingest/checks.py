"""Environment and input checks run by `ecoli_ingest check` and before `ecoli_ingest run`."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import container, paths, runconfig

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _exists(p: Path) -> bool:
    return p.exists()


def resource_path(cfg: dict, key: str, resources: Path | None, rundir: Path) -> Path | None:
    """Config value (relative to run dir) if given, else the default inside resources."""
    if cfg.get(key):
        return runconfig.resolve_path(cfg[key], rundir)
    if resources is None:
        return None
    return resources / paths.RESOURCE_FILES[key]


def check_mode(mode: str, resources: Path | None) -> list[Check]:
    out: list[Check] = []
    if mode == "never":
        missing = container.missing_tools()
        if missing:
            out.append(Check("tools on PATH", FAIL,
                             "missing: " + ", ".join(missing) + "  (activate the ecoli_ingest conda env "
                             "or use --container always)"))
        else:
            out.append(Check("tools on PATH", OK, f"{len(container.TOOLS_REQUIRED)} executables found"))
            try:
                r = subprocess.run(["Rscript", "-e", "suppressPackageStartupMessages(library(DESeq2))"],
                                   capture_output=True, text=True, timeout=180)
                err_lines = r.stderr.strip().splitlines()
                detail = "" if r.returncode == 0 else (err_lines[-1] if err_lines else "Rscript failed")
                out.append(Check("R: DESeq2 loads", OK if r.returncode == 0 else FAIL, detail))
            except (OSError, subprocess.TimeoutExpired) as e:
                out.append(Check("R: DESeq2 loads", FAIL, str(e)))
    else:
        rt = container.find_runtime()
        out.append(Check("container runtime", OK if rt else FAIL,
                         rt or "apptainer/singularity not found; set ECOLI_INGEST_APPTAINER or install "
                               "(see docs/cyverse.md for an unprivileged install)"))
        if resources is None:
            out.append(Check("container image", FAIL, "no resources directory"))
        else:
            img, legacy = paths.container_image(resources)
            if img.exists():
                out.append(Check("container image", WARN if legacy else OK,
                                 f"{img}" + ("  (legacy build/ layout; rename to container/)" if legacy else "")))
            else:
                out.append(Check("container image", FAIL,
                                 f"{img} not found; run `ecoli_ingest download container` or use --container never"))
    return out


def check_resources(cfg: dict, resources: Path | None, rundir: Path) -> list[Check]:
    out: list[Check] = []
    if resources is None:
        out.append(Check("resources directory", FAIL,
                         "not found; looked in: " + ", ".join(str(c) for c in paths.resources_candidates())
                         + "  (run `ecoli_ingest download core` or set ECOLI_INGEST_RESOURCES)"))
    else:
        out.append(Check("resources directory", OK, str(resources)))

    vg = resource_path(cfg, "vg_index", resources, rundir)
    if vg is not None:
        missing = [vg.name + s for s in paths.VG_SUFFIXES if not _exists(Path(str(vg) + s))]
        out.append(Check("vg graph index", FAIL if missing else OK,
                         f"{vg}.d2.*" if not missing else f"missing {', '.join(missing)} under {vg.parent}"))
        if not _exists(Path(str(vg) + paths.VG_OPTIONAL_SUFFIXES[0])):
            out.append(Check("vg short-read minimizer (k=15)", WARN,
                             "absent; samples with reads < 50 bp will fail at vg giraffe"))

    for key, label in (("annotation_gff", "annotation GFF (featureCounts)"),
                       ("annotation_source_gff", "annotation GFF (RefSeq source, for gene products)"),
                       ("annotation_bed", "annotation BED")):
        p = resource_path(cfg, key, resources, rundir)
        if p is not None:
            out.append(Check(label, OK if _exists(p) else FAIL, str(p)))

    if runconfig.centrifuge_enabled(cfg):
        cf = resource_path(cfg, "centrifuge_index_path", resources, rundir)
        if cf is not None:
            missing = [s for s in paths.CENTRIFUGE_SUFFIXES if not _exists(Path(str(cf) + s))]
            out.append(Check("centrifuge index", FAIL if missing else OK,
                             f"{cf}.*.cf" if not missing else
                             f"missing {', '.join(cf.name + s for s in missing)}; run `ecoli_ingest download` "
                             "or skip Centrifuge with `centrifuge: false` in config.yaml"))
    else:
        out.append(Check("centrifuge index", OK, "not needed (centrifuge: false)"))
        if cfg.get("centrifuge_min_percentage") is not None:
            out.append(Check("centrifuge gate", WARN,
                             "centrifuge_min_percentage is ignored because centrifuge: false"))
    return out


def check_run_dir(cfg: dict, rundir: Path, configfile: Path, mode: str = "never") -> list[Check]:
    out: list[Check] = []
    out.append(Check("run directory writable", OK if os.access(rundir, os.W_OK) else FAIL, str(rundir)))
    out.append(Check("config file", OK, str(configfile)))
    try:
        ss = runconfig.samples(cfg)
    except ValueError as e:
        return out + [Check("samples", FAIL, str(e))]
    if not ss:
        return out + [Check("samples", FAIL, "config.yaml has no samples")]
    missing, mixed_compression = [], []
    for s in ss:
        for files in (s.r1, s.r2 or []):
            for f in files:
                if not runconfig.resolve_path(f, rundir).exists():
                    missing.append(f)
            if len({f.lower().endswith((".gz", ".gzip")) for f in files}) > 1:
                mixed_compression.append(s.name)
    n_files = sum(len(s.r1) + len(s.r2 or []) for s in ss)
    n_local = sum(1 for s in ss if not s.accessions)
    detail = f"{n_local} samples, {n_files} files" if n_local else "none (all samples use accessions)"
    out.append(Check("sample reads", FAIL if missing else OK,
                     detail if not missing else "missing: " + ", ".join(missing)))
    if mixed_compression:
        out.append(Check("sample reads", FAIL, "gzipped and plain FASTQ mixed within one mate (files are "
                                               "concatenated as-is): " + ", ".join(mixed_compression)))
    layouts: dict[str, bool] = {}
    if any(s.accessions for s in ss):
        out.extend(_check_accessions(cfg, rundir, layouts, mode))
    paired, mixed = runconfig.is_paired(cfg, layouts)
    if mixed:
        out.append(Check("read layout", WARN, "mixed single- and paired-end samples: the workflow runs "
                                               "everything single-end and ignores r2"))
    else:
        out.append(Check("read layout", OK, "paired-end" if paired else "single-end"))
    if not cfg.get("deseq2", {}).get("contrasts"):
        out.append(Check("deseq2 contrasts", WARN, "none configured; DESeq2 outputs will be placeholders"))
    return out


def _check_accessions(cfg: dict, rundir: Path, layouts: dict[str, bool], mode: str = "never") -> list[Check]:
    """Resolve accession samples (network once; cached under <sra_dir>/manifest) and summarise the
    download. Fills ``layouts`` (sample -> paired?) for the read-layout check."""
    sra = paths.load_script("sra_reads")
    sdir = runconfig.sra_dir(cfg, rundir)
    n_acc = sum(len(s.accessions) for s in runconfig.samples(cfg))
    try:
        reads = sra.resolve_samples(cfg.get("samples") or {}, str(sdir), str(sdir / "manifest"))
    except ValueError as e:
        return [Check("sra accessions", FAIL, str(e))]
    layouts.update({name: spec["r2"] is not None for name, spec in reads.items() if spec["accessions"]})
    summ = sra.download_summary(reads, str(sdir))
    gb = summ["pending_bytes"] / 1e9
    detail = (f"{n_acc} accession(s) -> {summ['runs']} run(s), {summ['files']} file(s); "
              f"{summ['present']} already in {sdir}, {gb:.2f} GB to download")
    out = [Check("sra accessions", OK, detail)]
    if summ["present"] < summ["files"]:
        out.append(_check_sra_method(sra, cfg, mode, summ.get("no_ena") or []))
    return out


def _check_sra_method(sra, cfg: dict, mode: str, no_ena: list) -> Check:
    """Which source the missing runs will come from: sra-tools (prefetch + fasterq-dump) first and
    ENA as the backup, unless ``sra_method`` names one source."""
    try:
        method = sra.check_method(cfg.get("sra_method"))
    except ValueError as e:
        return Check("read download method", FAIL, str(e))
    only_sra = f"; {len(no_ena)} run(s) have no ENA FASTQ (e.g. {no_ena[0]})" if no_ena else ""
    if method == "ena":
        if no_ena:
            return Check("read download method", FAIL, f"sra_method: ena, but ENA has no FASTQ files for {', '.join(no_ena[:5])}")
        return Check("read download method", OK, "ENA FASTQ files (sra_method: ena)")
    if mode != "never":
        return Check("read download method", OK, f"{method}: sra-tools when the image provides them"
                     + ("" if method == "sra-tools" else ", else ENA") + only_sra)
    have = sra.sra_tools_available()
    if have:
        return Check("read download method", OK, "sra-tools (prefetch + fasterq-dump)"
                     + ("" if method == "sra-tools" else ", ENA FASTQ files as the backup") + only_sra)
    if method == "sra-tools" or no_ena:
        return Check("read download method", FAIL, "prefetch / fasterq-dump not on PATH (conda install -c bioconda sra-tools)"
                     + only_sra)
    return Check("read download method", WARN, "prefetch / fasterq-dump not on PATH: downloading from ENA only "
                                               "(slower; `conda install -c bioconda sra-tools` enables the primary method)")


def run_all(*, cfg: dict, rundir: Path, configfile: Path, resources: Path | None, mode: str) -> list[Check]:
    return check_mode(mode, resources) + check_resources(cfg, resources, rundir) + check_run_dir(cfg, rundir, configfile, mode)


def report(checks: list[Check], out=None) -> bool:
    import sys
    out = out or sys.stderr
    width = max(len(c.name) for c in checks) if checks else 10
    for c in checks:
        print(f"[{c.status:>4}] {c.name:<{width}}  {c.detail}", file=out)
    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    print(f"{len(checks)} checks: {n_fail} failed, {n_warn} warnings", file=out)
    return n_fail == 0
