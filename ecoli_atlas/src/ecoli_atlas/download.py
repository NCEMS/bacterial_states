"""Data bundles for Ecoli_ATLAS: assets of the GitHub release `resources-v2` of NCEMS/bacterial_states (built by
`release/make_release.py`). The URLs below are the only place that names the download location.

Each bundle pins the SHA-256 of its archive. The archives are built reproducibly, so the same files
always give the same checksum; a download that does not match is rejected before anything is
unpacked. A new archive needs a new version (ATLAS_DATA_VERSION) and a new checksum here.

The Apptainer image (CPU) is the environment of a major version, not of one code release: the packages of
locks/linux-64.txt, `ecoli_atlas_env<IMAGE_ENV>.sif` in the GitHub release `containers-v<major>` (`curl -LO <IMAGE_URL>`,
then `sha256sum` against IMAGE_SHA256). It also holds the package as of its build, for `apptainer run` without an
install; with ECOLI_ATLAS_SRC bound in, it runs that code instead (see container/Singularity.def)."""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ATLAS_DATA_VERSION = "v2"
# assets of the GitHub data release `resources-<version>` (code releases are tagged vX.Y.Z)
RELEASE_URL = f"https://github.com/NCEMS/bacterial_states/releases/download/resources-{ATLAS_DATA_VERSION}"
# Apptainer image: the environment of locks/linux-64.txt (release/build_container.sh, which refuses a lock whose
# SHA-256 is not IMAGE_LOCK_SHA256). One image per environment, normally one per major version: a change of the lock
# needs a new IMAGE_ENV ("2.1", or the next major) and a new image, never a replaced one. CPU torch only: the CUDA
# stack is ~5 GB compressed, over the 2 GiB limit of a release asset; GPU work uses the conda environment
# (requirements-cuda.txt). Check the download against IMAGE_SHA256, None until the image is published.
IMAGE_ENV = "2"
IMAGE_LOCK_SHA256 = "8c0447c27d7bb0905ebcdcdc7083063f88703beaee7b0f51460dcbbeb0c91545"   # locks/linux-64.txt
IMAGE_URL = (f"https://github.com/NCEMS/bacterial_states/releases/download/containers-v{IMAGE_ENV.split('.')[0]}/"
             f"ecoli_atlas_env{IMAGE_ENV}.sif")
IMAGE_SHA256: str | None = None
ENV_DATA = "ECOLI_ATLAS_DATA"
ENV_MODELS = "ECOLI_ATLAS_MODELS"
ENV_DATA_ROOT = "BS_DATA_ROOT"

BUNDLES = {
    "training-data": {
        "url": f"{RELEASE_URL}/ecoli_atlas_training_data_{ATLAS_DATA_VERSION}.tar.gz",
        "sha256": "2d1fa16677f2fec09b2d316021b1024188270c5f6d09b3f331db2a503dd03443",
        "size": "100 MB download, 301 MB unpacked",
        "contents": ["atlas_v2_counts.tsv", "atlas_v2_antisense_counts.tsv", "metadata_v2.tsv", "sample_qc.tsv",
                     "time_courses.tsv", "atlas_v2_missing_genes.tsv (genes called missing, per sample)",
                     "metadata_v2_studies.tsv, metadata_v2_sample_text.tsv (GEO text)",
                     "metadata_v2_evidence.tsv (the text behind each label)", "metadata_v2_label_scheme.yaml"],
        "subdir": "atlas_data",
    },
    "models": {
        "url": f"{RELEASE_URL}/ecoli_atlas_models_{ATLAS_DATA_VERSION}.tar.gz",
        "sha256": "195e5177d39843264087cc3b557d4b6b44f5bbc92002794bc8bb7797c7569779",
        "size": "312 MB download, 378 MB unpacked",
        "contents": ["models/ (the selected bundle: model.pt, preprocessing.pt, manifest.json, splits.csv, the atlas's "
                     "embedding.tsv, predictions.tsv, missing_genes.tsv, evaluation.json)",
                     "models/seed_1 ... seed_8 (every seed of the sweep, for drivers and seed spreads)"],
        "subdir": "outputs",
    },
}


# Curated gene modules for `drivers --modules` (built into modules.tsv by `ecoli_atlas download modules`). They are not
# shipped: each source is fetched from its own site. precise1k (SBRG, MIT licence) is pinned to one commit and checked
# by SHA-256; KEGG is fetched from its REST service (free for academic use, not redistributed), whose content changes
# with KEGG releases, so the retrieval date is recorded instead of a checksum.
PRECISE1K_COMMIT = "4829b83ace51ee7980de6950077f1cf8f3746f88"     # SBRG/precise1k, 2023-08-29
PRECISE1K_RAW = f"https://raw.githubusercontent.com/SBRG/precise1k/{PRECISE1K_COMMIT}"
MODULE_SOURCES = {
    "precise1k": {
        "description": "RegulonDB regulons (gene-level TRN) and the PRECISE-1K iModulons; SBRG/precise1k, MIT licence",
        "files": {
            "TRN.csv": (f"{PRECISE1K_RAW}/data/annotation/TRN.csv",
                        "d91a2caf32807cc03332cf4f52692ac7ff41027f7b9bf58d50e9b999f5a2a462"),
            "M.csv": (f"{PRECISE1K_RAW}/data/precise1k/M.csv",
                      "6ad63e36378b9da8e7c5b501cca81a5db7f65872684f4a870428e37380c06afa"),
            "imodulon_table.csv": (f"{PRECISE1K_RAW}/data/precise1k/imodulon_table.csv",
                                   "da98e18363014ab64dca198ebf4f5410887e519ba8d98a5bbc88143ac477b063"),
            "LICENSE": (f"{PRECISE1K_RAW}/LICENSE", "b974c4b7ee69ee9d26e0148b38b4766f3e149c4ec12fc1be3ce1e4e0f8f83014"),
        },
    },
    "kegg": {
        "description": "KEGG pathways and modules of E. coli K-12 MG1655 (organism eco); KEGG REST, free for academic use",
        "files": {
            "list_pathway_eco.txt": ("https://rest.kegg.jp/list/pathway/eco", None),
            "link_pathway_eco.txt": ("https://rest.kegg.jp/link/pathway/eco", None),
            "list_module.txt": ("https://rest.kegg.jp/list/module", None),
            "link_module_eco.txt": ("https://rest.kegg.jp/link/module/eco", None),
            "brite_pathways.keg": ("https://rest.kegg.jp/get/br:br08901", None),     # classes A/B of the pathway maps
            "brite_modules.keg": ("https://rest.kegg.jp/get/br:ko00002", None),      # classes A/B/C of the modules
        },
    },
}


def download_module_sources(dest: Path, sources=tuple(MODULE_SOURCES), log=None) -> dict:
    """Fetch the module source files into ``dest/<source>/``; a pinned file whose checksum does not match is
    rejected (ValueError). Returns {source: {file: {url, sha256, retrieved}}} for the manifest."""
    import datetime
    log = log or sys.stderr
    record = {}
    for src in sources:
        info = MODULE_SOURCES[src]
        d = Path(dest) / src
        d.mkdir(parents=True, exist_ok=True)
        record[src] = {}
        for name, (url, sha) in info["files"].items():
            req = urllib.request.Request(url, headers={"User-Agent": "ecoli-atlas"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            got = hashlib.sha256(data).hexdigest()
            if sha and got != sha:
                raise ValueError(f"{src}/{name}: checksum mismatch (got {got}, expected {sha}); nothing was kept")
            (d / name).write_bytes(data)
            record[src][name] = {"url": url, "sha256": got,
                                 "retrieved": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            print(f"  {src}/{name}: {len(data):,} bytes" + (" (checksum ok)" if sha else ""), file=log)
    return record


def data_root() -> Path:
    return Path(os.environ.get(ENV_DATA_ROOT, Path.cwd()))


def default_data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA, data_root() / "atlas_data"))


def default_models_dir() -> Path:
    return Path(os.environ.get(ENV_MODELS, data_root() / "outputs" / "models"))


def download(bundle: str, dest: Path | None = None, dry_run: bool = False, log=None) -> int:
    log = log or sys.stderr
    info = BUNDLES[bundle]
    dest = Path(dest) if dest else data_root() / info["subdir"]
    print(f"bundle   : {bundle} ({info['size']})", file=log)
    print(f"url      : {info['url']}", file=log)
    print(f"target   : {dest}", file=log)
    print("contents : " + ", ".join(info["contents"]), file=log)
    if "TBD" in info["url"]:
        print("\nThe download location for this bundle has not been published yet. Until then copy the "
              "files listed above into the target directory by hand.", file=log)
        return 2
    if dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        h = hashlib.sha256()
        with urllib.request.urlopen(info["url"]) as resp, open(tmp_path, "wb") as out:
            for block in iter(lambda: resp.read(1 << 24), b""):
                h.update(block)
                out.write(block)
        if h.hexdigest() != info["sha256"]:
            print(f"error: checksum mismatch for {bundle}: got {h.hexdigest()}, expected {info['sha256']}; "
                  "nothing was unpacked", file=log)
            return 1
        print("checksum ok", file=log)
        with tarfile.open(tmp_path, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                _check_member(m, dest)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(dest, members=members, filter="data")
            else:
                tar.extractall(dest, members=members)
    finally:
        tmp_path.unlink(missing_ok=True)
    return 0


def _check_member(member: tarfile.TarInfo, target: Path) -> None:
    """Refuse members that would land outside ``target`` or are not plain files/directories.
    (Same rule as ecoli_ingest.resources; duplicated so the packages stay independent.)"""
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"refusing to unpack {member.name!r}: not a regular file or directory")
    resolved = (target / member.name).resolve()
    if resolved != target.resolve() and target.resolve() not in resolved.parents:
        raise ValueError(f"refusing to unpack {member.name!r}: outside {target}")
