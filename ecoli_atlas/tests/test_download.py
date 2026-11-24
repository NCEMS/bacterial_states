import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from ecoli_atlas import download as dl

RELEASE = Path(__file__).resolve().parents[2] / "work" / "release_v2"


def _bundle(tmp_path, members, name="b.tar.gz"):
    tgz = tmp_path / name
    with tarfile.open(tgz, "w:gz") as tar:
        for arcname, data in members.items():
            info = tarfile.TarInfo(arcname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return tgz, hashlib.sha256(tgz.read_bytes()).hexdigest()


def _point_at(monkeypatch, bundle, tgz, digest):
    monkeypatch.setitem(dl.BUNDLES, bundle, dl.BUNDLES[bundle] | {"url": tgz.as_uri(), "sha256": digest})


def test_every_bundle_pins_a_checksum():
    for info in dl.BUNDLES.values():
        assert len(info["sha256"]) == 64 and int(info["sha256"], 16) >= 0
        assert info["url"].startswith("https://github.com/NCEMS/bacterial_states/releases/download/resources-v2/ecoli_atlas_")


def test_download_verifies_then_unpacks(tmp_path, monkeypatch):
    tgz, digest = _bundle(tmp_path, {"models/manifest.json": b"{}", "models/seed_1/manifest.json": b"{}"})
    _point_at(monkeypatch, "models", tgz, digest)
    log = io.StringIO()
    assert dl.download("models", tmp_path / "out", log=log) == 0
    assert "checksum ok" in log.getvalue()
    assert (tmp_path / "out" / "models" / "manifest.json").read_bytes() == b"{}"


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    tgz, _ = _bundle(tmp_path, {"atlas_v2_counts.tsv": b"x"})
    _point_at(monkeypatch, "training-data", tgz, "0" * 64)
    log = io.StringIO()
    assert dl.download("training-data", tmp_path / "out", log=log) == 1
    assert "checksum mismatch" in log.getvalue()
    assert not (tmp_path / "out" / "atlas_v2_counts.tsv").exists()      # nothing unpacked


def test_download_refuses_members_outside_target(tmp_path, monkeypatch):
    tgz, digest = _bundle(tmp_path, {"../escape.txt": b"x"})
    _point_at(monkeypatch, "training-data", tgz, digest)
    with pytest.raises(ValueError, match="outside"):
        dl.download("training-data", tmp_path / "out", log=io.StringIO())
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.skipif(not RELEASE.is_dir(), reason="release archives not built in this checkout")
def test_pinned_checksums_match_release_archives():
    for info in dl.BUNDLES.values():
        archive = RELEASE / info["url"].rsplit("/", 1)[1]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["sha256"], archive.name


def test_image_download_checks_the_pinned_sha256(tmp_path, monkeypatch):
    from ecoli_atlas import download
    from ecoli_atlas.cli import main
    payload = b"not really an image"
    src = tmp_path / "src.sif"
    src.write_bytes(payload)
    monkeypatch.setattr(download, "IMAGE_URL", src.as_uri())
    monkeypatch.setattr(download, "IMAGE_SHA256", hashlib.sha256(payload).hexdigest())
    target = download.image_path(tmp_path / "dest")
    assert target.name == f"ecoli_atlas_env{download.IMAGE_ENV}.sif"
    assert main(["download", "image", "--dest", str(tmp_path / "dest")]) == 0
    assert target.read_bytes() == payload
    target.unlink()
    monkeypatch.setattr(download, "IMAGE_SHA256", "0" * 64)                       # a mismatch leaves nothing behind
    assert main(["download", "image", "--dest", str(tmp_path / "dest")]) == 1
    assert not list((tmp_path / "dest").iterdir())
    monkeypatch.setattr(download, "IMAGE_SHA256", None)
    assert download.download_image(tmp_path / "dest", dry_run=True) == 2          # unpublished: says so


def test_image_pin_is_published():
    from ecoli_atlas import download
    assert download.IMAGE_SHA256 and len(download.IMAGE_SHA256) == 64
