"""Resource bundles (manifest generation/verification, download) and the container image.

The core bundle is an asset of the GitHub release `resources-v2` of NCEMS/bacterial_states; the Centrifuge
index is fetched from its authors' public bucket. Both are checked against the packaged manifest
after unpacking. The Apptainer image is the environment of a major version, not of one code release: the
tools of locks/linux-64.txt, `ecoli_ingest_env<CONTAINER_ENV>.sif` in the GitHub release `containers-v<major>`,
downloaded over HTTPS and checked against the SHA-256 pinned below. The host launcher runs its own Snakefile and
scripts in it, so every 2.x release uses the same image as long as the lock does not change. Every URL lives in
BUNDLE_URLS / CONTAINER_URL below and nowhere else.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

from . import paths

RESOURCES_VERSION = "v2"
# assets of the GitHub data release `resources-<version>` (code releases are tagged vX.Y.Z)
RELEASE_URL = f"https://github.com/NCEMS/bacterial_states/releases/download/resources-{RESOURCES_VERSION}"

# bundle name -> top-level directories (relative to the resources dir) that belong to it
BUNDLES = {
    "core": ("vg", "annotation"),
    "centrifuge": ("centrifuge",),
}
BUNDLE_SIZES = {"core": "178 MB download, 688 MB unpacked", "centrifuge": "5.4 GB download, 6.9 GB unpacked"}
BUNDLE_URLS = {
    "core": f"{RELEASE_URL}/ecoli_ingest_core_{RESOURCES_VERSION}.tar.gz",
    # upstream prebuilt index from the Centrifuge authors (Kim et al. 2016; built 2016). The file name
    # is not versioned, so pin the S3 object version; the manifest checksums are verified after unpacking
    "centrifuge": "https://genome-idx.s3.amazonaws.com/centrifuge/p_compressed%2Bh%2Bv.tar.gz"
                  "?versionId=KoqnkfTuATv3a5Cq9hBzQxwFWYRx90aj",
}
# directory (relative to the resources dir) a bundle's tarball is unpacked into; our own tarballs carry
# their top-level directories, the upstream Centrifuge tarball has its .cf files at the root
BUNDLE_SUBDIR = {"centrifuge": "centrifuge"}
# downloaded by `ecoli_ingest download` when no bundle is named; the container needs --with-apptainer
DEFAULT_BUNDLES = ("core", "centrifuge")

# Apptainer image: the environment of locks/linux-64.txt (built by release/build_container.sh, which refuses a lock
# whose SHA-256 is not CONTAINER_LOCK_SHA256). One image per environment, normally one per major version: a change of
# the lock needs a new CONTAINER_ENV ("2.1", or the next major) and a new image, never a replaced one, so that every
# release keeps the image it pins. Release assets are limited to 2 GiB; the image is ~1.1 GiB. The download is
# checked against CONTAINER_SHA256, None until the image of this environment is published.
CONTAINER_ENV = "2"
CONTAINER_LOCK_SHA256 = "52f727342d6ffdce0add4f030c930b980a065817c716c2abec7cca2b5c616930"   # locks/linux-64.txt
CONTAINER_RELEASE_URL = ("https://github.com/NCEMS/bacterial_states/releases/download/"
                         f"containers-v{CONTAINER_ENV.split('.')[0]}")
CONTAINER_URL = f"{CONTAINER_RELEASE_URL}/ecoli_ingest_env{CONTAINER_ENV}.sif"
CONTAINER_SHA256: str | None = "19be5ba51343cc4b1c2788cab4f0a230f18fb9ee029652261043e73272d801b4"
CONTAINER_SIZE = "~1.2 GB"
# everything `ecoli_ingest download` accepts
DOWNLOADS = (*BUNDLES, "container")

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


def _check_member(member: tarfile.TarInfo, target: Path) -> None:
    """Refuse members that would land outside ``target`` or are not plain files/directories."""
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"refusing to unpack {member.name!r}: not a regular file or directory")
    resolved = (target / member.name).resolve()
    if resolved != target.resolve() and target.resolve() not in resolved.parents:
        raise ValueError(f"refusing to unpack {member.name!r}: outside {target}")


def download_container(dest: Path, dry_run: bool = False, log=None) -> int:
    """Download the image of this environment (CONTAINER_ENV) into <dest>/container/ and check its SHA-256."""
    log = log or sys.stderr
    target = paths.container_image(dest)
    print(f"bundle   : container ({CONTAINER_SIZE})", file=log)
    print(f"url      : {CONTAINER_URL}", file=log)
    print(f"target   : {target}", file=log)
    if CONTAINER_SHA256 is None:
        print(f"\nThe image of environment {CONTAINER_ENV} has not been published yet. Build it with", file=log)
        print("    release/build_container.sh ecoli_ingest      (in a checkout of the repository)", file=log)
        print(f"and put it at {target}, or use the conda environment (--container never).", file=log)
        return 2
    if dry_run:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(".partial_" + target.name)
    h = hashlib.sha256()
    print("downloading ...", file=log, flush=True)
    try:
        # streamed into a partial file; the image only takes its name once the checksum matches
        with urllib.request.urlopen(CONTAINER_URL) as resp, open(part, "wb") as fh:
            for block in iter(lambda: resp.read(1 << 24), b""):
                h.update(block)
                fh.write(block)
    except OSError as e:
        print(f"download failed: {e}", file=log)
        part.unlink(missing_ok=True)
        return 1
    digest = h.hexdigest()
    if digest != CONTAINER_SHA256:
        print(f"checksum mismatch: got {digest}, expected {CONTAINER_SHA256}", file=log)
        part.unlink(missing_ok=True)
        return 1
    part.chmod(0o755)
    part.replace(target)
    print("checksum ok", file=log)
    return 0


def download(bundle: str, dest: Path, dry_run: bool = False, log=None) -> int:
    if bundle == "container":
        return download_container(dest, dry_run=dry_run, log=log)
    log = log or sys.stderr
    url = BUNDLE_URLS[bundle]
    print(f"bundle   : {bundle} ({BUNDLE_SIZES[bundle]})", file=log)
    print(f"url      : {url}", file=log)
    print(f"target   : {dest}", file=log)
    if dry_run:
        return 0
    target = dest / BUNDLE_SUBDIR.get(bundle, "")
    target.mkdir(parents=True, exist_ok=True)
    print("downloading and unpacking ...", file=log, flush=True)
    # streamed: no temporary copy of the (multi-GB) tarball
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            _check_member(member, target)
            print(f"  {member.name}", file=log, flush=True)
            if hasattr(tarfile, "data_filter"):
                tar.extract(member, target, filter="data")
            else:
                tar.extract(member, target)
    manifest = paths.packaged_manifest()
    if manifest.exists():
        bad = [(r, s) for r, s in verify(dest, read_manifest(manifest), {bundle}) if s != "ok"]
        if bad:
            for rel, status in bad:
                print(f"  {status:8s} {rel}", file=log)
            return 1
        print("checksums ok", file=log)
    return 0
