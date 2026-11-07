"""Data bundles for Ecoli_ATLAS. Hosting is not decided yet; the URLs below are the only place that
needs to change once it is."""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ATLAS_DATA_VERSION = "v1"
ENV_DATA = "ECOLI_ATLAS_DATA"
ENV_MODELS = "ECOLI_ATLAS_MODELS"
ENV_DATA_ROOT = "BS_DATA_ROOT"

BUNDLES = {
    "training-data": {
        "url": f"https://TBD/ncems/bacterial_states/ecoli_atlas_training_data_{ATLAS_DATA_VERSION}.tar.gz",
        "size": "~56 MB",
        "contents": ["merged_count_table.txt", "metadata_revised.csv", "GCF_000005845.2_ASM584v2_feature_table.txt"],
        "subdir": "atlas_data",
    },
    "models": {
        "url": f"https://TBD/ncems/bacterial_states/ecoli_atlas_models_{ATLAS_DATA_VERSION}.tar.gz",
        "size": "~20 MB",
        "contents": ["models/{growth_phase,medium_type,is_lab_strain}_encoder.pt", "models/residual_ae.pt",
                     "models/preprocessing.pt", "models/manifest.json", "models/splits.csv",
                     "atlas/final_embedding.csv", "atlas/sample_predictions.csv", "atlas/embedding_block_layout.csv",
                     "atlas/atlas_metadata.csv", "atlas/atlas_manifest.json"],
        "subdir": "outputs",
    },
}
FEATURE_TABLE = "GCF_000005845.2_ASM584v2_feature_table.txt"


def data_root() -> Path:
    return Path(os.environ.get(ENV_DATA_ROOT, Path.cwd()))


def default_data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA, data_root() / "atlas_data"))


def default_models_dir() -> Path:
    return Path(os.environ.get(ENV_MODELS, data_root() / "outputs" / "models"))


def default_atlas_dir() -> Path:
    return default_models_dir().parent / "atlas"


def default_feature_table() -> Path | None:
    p = default_data_dir() / FEATURE_TABLE
    return p if p.exists() else None


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
        with urllib.request.urlopen(info["url"]) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 24)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return 0
