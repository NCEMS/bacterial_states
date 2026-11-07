import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from ecoli_atlas import download as dl

RELEASE = Path(__file__).resolve().parents[2] / "work" / "release_v1"


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


def test_download_verifies_then_unpacks(tmp_path, monkeypatch):
    tgz, digest = _bundle(tmp_path, {"models/manifest.json": b"{}", "atlas/final_embedding.csv": b"sample\n"})
    _point_at(monkeypatch, "models", tgz, digest)
    log = io.StringIO()
    assert dl.download("models", tmp_path / "out", log=log) == 0
    assert "checksum ok" in log.getvalue()
    assert (tmp_path / "out" / "models" / "manifest.json").read_bytes() == b"{}"


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    tgz, _ = _bundle(tmp_path, {"merged_count_table.txt": b"x"})
    _point_at(monkeypatch, "training-data", tgz, "0" * 64)
    log = io.StringIO()
    assert dl.download("training-data", tmp_path / "out", log=log) == 1
    assert "checksum mismatch" in log.getvalue()
    assert not (tmp_path / "out" / "merged_count_table.txt").exists()      # nothing unpacked


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
