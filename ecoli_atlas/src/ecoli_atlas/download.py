"""Data bundles for Ecoli_ATLAS: assets of the GitHub release `resources-v2` of NCEMS/bacterial_states (built by
`release/make_release.py`). The URLs below are the only place that names the download location.

Each bundle pins the SHA-256 of its archive. The archives are built reproducibly, so the same files
always give the same checksum; a download that does not match is rejected before anything is
unpacked. A new archive needs a new version (ATLAS_DATA_VERSION) and a new checksum here."""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ATLAS_DATA_VERSION = "v2"
# assets of the GitHub data release `resources-<version>` (code releases are tagged X.X.X)
RELEASE_URL = f"https://github.com/NCEMS/bacterial_states/releases/download/resources-{ATLAS_DATA_VERSION}"
ENV_DATA = "ECOLI_ATLAS_DATA"
ENV_MODELS = "ECOLI_ATLAS_MODELS"
ENV_DATA_ROOT = "BS_DATA_ROOT"

BUNDLES = {
    "training-data": {
        "url": f"{RELEASE_URL}/ecoli_atlas_training_data_{ATLAS_DATA_VERSION}.tar.gz",
        "sha256": "a1edcd18a016fbb2cdbb4a407027d5acef01b86c70b9e2ccdd51df6beb4ebbe9",
        "size": "100 MB download, 301 MB unpacked",
        "contents": ["atlas_v2_counts.tsv", "atlas_v2_antisense_counts.tsv", "metadata_v2.tsv", "sample_qc.tsv",
                     "time_courses.tsv", "atlas_v2_missing_genes.tsv (genes called missing, per sample)",
                     "metadata_v2_studies.tsv, metadata_v2_sample_text.tsv (GEO text)",
                     "metadata_v2_evidence.tsv (the text behind each label)", "metadata_v2_label_scheme.yaml"],
        "subdir": "atlas_data",
    },
    "models": {
        "url": f"{RELEASE_URL}/ecoli_atlas_models_{ATLAS_DATA_VERSION}.tar.gz",
        "sha256": "ca953af2624c3e485d9e31af83ff30c90c4094de33937062be9afd5b0c0aae25",
        "size": "312 MB download, 378 MB unpacked",
        "contents": ["models/ (the selected bundle: model.pt, preprocessing.pt, manifest.json, splits.csv, the atlas's "
                     "embedding.tsv, predictions.tsv, missing_genes.tsv, evaluation.json)",
                     "models/seed_1 ... seed_8 (every seed of the sweep, for drivers and seed spreads)"],
        "subdir": "outputs",
    },
}


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
