"""Resource bundles: manifest generation/verification and the download stub.

Hosting is not decided yet. Every URL lives in BUNDLE_URLS below and nowhere else; until they are
filled in, `download` prints the interim CyVerse instructions instead.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from . import paths

RESOURCES_VERSION = "v1"

# bundle name -> top-level directories (relative to the resources dir) that belong to it
BUNDLES = {
    "core": ("vg", "annotation"),
    "centrifuge": ("centrifuge",),
    "container": ("container",),
}
BUNDLE_SIZES = {"core": "~650 MB", "centrifuge": "~6.9 GB", "container": "~5.3 GB"}
BUNDLE_URLS = {
    "core": f"https://TBD/ncems/bacterial_states/ingest_resources_core_{RESOURCES_VERSION}.tar.gz",
    "centrifuge": f"https://TBD/ncems/bacterial_states/ingest_resources_centrifuge_{RESOURCES_VERSION}.tar.gz",
    "container": f"https://TBD/ncems/bacterial_states/ingest_container_{RESOURCES_VERSION}.tar.gz",
}
CYVERSE_LEGACY_PATH = "/iplant/home/shared/NCEMS/working-groups/bacterial-states/resources.tar.gz"

MANIFEST_COLUMNS = ("path", "sha256", "bytes", "bundle")


def bundle_of(relpath: str) -> str | None:
    top = relpath.split("/", 1)[0]
    for bundle, tops in BUNDLES.items():
        if top in tops:
            return bundle
    return None


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def generate_manifest(resources: Path, log=None) -> list[tuple[str, str, int, str]]:
    rows = []
    for root, _dirs, files in os.walk(resources):
        for name in sorted(files):
            p = Path(root) / name
            rel = p.relative_to(resources).as_posix()
            bundle = bundle_of(rel)
            if bundle is None:
                continue
            if log:
                print(f"hashing {rel} ...", file=log, flush=True)
            rows.append((rel, sha256_of(p), p.stat().st_size, bundle))
    rows.sort()
    return rows


def write_manifest(rows, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for rel, digest, size, bundle in rows:
            fh.write(f"{rel}\t{digest}\t{size}\t{bundle}\n")


def read_manifest(path: Path) -> list[tuple[str, str, int, str]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if tuple(header) != MANIFEST_COLUMNS:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in fh:
            if not line.strip():
                continue
            rel, digest, size, bundle = line.rstrip("\n").split("\t")
            rows.append((rel, digest, int(size), bundle))
    return rows


def verify(resources: Path, rows, bundles: set[str] | None = None, log=None) -> list[tuple[str, str]]:
    """Return (path, status) with status in ok / missing / size / sha256."""
    results = []
    for rel, digest, size, bundle in rows:
        if bundles and bundle not in bundles:
            continue
        p = resources / rel
        if not p.exists():
            results.append((rel, "missing"))
        elif p.stat().st_size != size:
            results.append((rel, "size"))
        else:
            if log:
                print(f"verifying {rel} ...", file=log, flush=True)
            results.append((rel, "ok" if sha256_of(p) == digest else "sha256"))
    return results


def download(bundle: str, dest: Path, dry_run: bool = False, log=None) -> int:
    log = log or sys.stderr
    url = BUNDLE_URLS[bundle]
    print(f"bundle   : {bundle} ({BUNDLE_SIZES[bundle]})", file=log)
    print(f"url      : {url}", file=log)
    print(f"target   : {dest}", file=log)
    if "TBD" in url:
        print("\nThe download location for this bundle has not been published yet.", file=log)
        print("Interim: fetch the complete resources tarball from CyVerse with GoCommands", file=log)
        print(f"    gocmd get --progress -f --icat {CYVERSE_LEGACY_PATH}", file=log)
        print(f"    tar -xvzf resources.tar.gz -C {dest.parent}", file=log)
        print("then `mv resources/build resources/container` (the image lives in container/ now).", file=log)
        return 2
    if dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print("downloading ...", file=log, flush=True)
        with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 24)
        print("extracting ...", file=log, flush=True)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    manifest = paths.packaged_manifest()
    if manifest.exists():
        bad = [(r, s) for r, s in verify(dest, read_manifest(manifest), {bundle}) if s != "ok"]
        if bad:
            for rel, status in bad:
                print(f"  {status:8s} {rel}", file=log)
            return 1
        print("checksums ok", file=log)
    return 0
