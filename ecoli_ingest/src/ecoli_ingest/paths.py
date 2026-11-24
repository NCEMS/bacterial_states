"""Locations of the packaged workflow and of the external resource bundle."""
from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

# Relative paths inside a resources directory. Mirrors the defaults in workflow/Snakefile.
RESOURCE_FILES = {
    "vg_index": "vg/ecoli_graph_test",
    "annotation_gff": "annotation/GCF_000005845.2_ASM584v2_counting_v2.gff",
    "annotation_bed": "annotation/GCF_000005845.2_ASM584v2_counting_v2.bed",
    "genome_fasta": "annotation/GCF_000005845.2_ASM584v2_genomic.fna",     # NC_000913.3; GC bias in the QC record
    "centrifuge_index_path": "centrifuge/p_compressed+h+v",
}
CONTAINER_IMAGE = "container/ecoli_ingest.sif"   # written by `ecoli_ingest download container`

VG_SUFFIXES = (".d2.gbz", ".d2.dist", ".d2.min", ".d2.shortread.zipcodes")
VG_OPTIONAL_SUFFIXES = (".d2.k15.min",)  # used automatically for reads shorter than 50 bp
CENTRIFUGE_SUFFIXES = (".1.cf", ".2.cf", ".3.cf")

ENV_RESOURCES = "ECOLI_INGEST_RESOURCES"
ENV_DATA_ROOT = "BS_DATA_ROOT"
ENV_RUNTIME = "ECOLI_INGEST_APPTAINER"


def package_root() -> Path:
    return Path(str(files("ecoli_ingest")))


def package_commit() -> str:
    """Short git commit of the package source: from git when it is a checkout, else from the REVISION file that
    release/build_container.sh writes into the image's copy; "" when neither is there."""
    import subprocess
    root = package_root()
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return (root / "REVISION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def workflow_dir() -> Path:
    return package_root() / "workflow"


def snakefile() -> Path:
    return workflow_dir() / "Snakefile"


def scripts_dir() -> Path:
    return workflow_dir() / "scripts"


def load_script(name: str):
    """Import ``workflow/scripts/<name>.py`` as a module (they are plain scripts, not a package)."""
    import importlib.util
    path = scripts_dir() / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ecoli_ingest_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def config_template() -> Path:
    return package_root() / "config" / "config.template.yaml"


def packaged_manifest() -> Path:
    return package_root() / "resources.manifest.tsv"


def resources_candidates(explicit: str | os.PathLike | None = None) -> list[Path]:
    """Ordered places to look for the resources directory."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get(ENV_RESOURCES):
        cands.append(Path(os.environ[ENV_RESOURCES]))
    if os.environ.get(ENV_DATA_ROOT):
        cands.append(Path(os.environ[ENV_DATA_ROOT]) / "resources")
    # ./resources, then a resources/ directory with the expected layout in any parent directory
    # (so run directories under a checkout or a project folder find it without configuration)
    cwd = Path.cwd()
    for d in [cwd] + list(cwd.parents):
        cand = d / "resources"
        if d == cwd or (cand / "annotation").is_dir() or (cand / "vg").is_dir():
            cands.append(cand)
    cands.append(Path.home() / ".local" / "share" / "ecoli_ingest" / "resources")
    seen: set[Path] = set()
    out: list[Path] = []
    for c in cands:
        c = c.expanduser()
        key = Path(os.path.abspath(c))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def find_resources(explicit: str | os.PathLike | None = None) -> Path | None:
    for c in resources_candidates(explicit):
        if c.is_dir():
            return c
    return None


def default_resources_target(explicit: str | os.PathLike | None = None) -> Path:
    """Where `download` should put things when no resources directory exists yet."""
    return resources_candidates(explicit)[0]


def container_image(resources: Path) -> Path:
    """The container image inside a resources directory."""
    return resources / CONTAINER_IMAGE
