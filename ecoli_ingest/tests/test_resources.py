from pathlib import Path

from ecoli_ingest import paths, resources


def test_bundle_of():
    assert resources.bundle_of("vg/ecoli_graph_test.d2.gbz") == "core"
    assert resources.bundle_of("annotation/x.gff") == "core"
    assert resources.bundle_of("centrifuge/p_compressed+h+v.1.cf") == "centrifuge"
    assert resources.bundle_of("container/rnaseq_vg.sif") == "container"
    assert resources.bundle_of("linear/ecoli_linear.gbz") is None  # not distributed


def test_manifest_roundtrip_and_verify(tmp_path):
    res = tmp_path / "resources"
    (res / "vg").mkdir(parents=True)
    (res / "linear").mkdir()
    (res / "vg" / "a.d2.gbz").write_bytes(b"abc")
    (res / "linear" / "ignored.gbz").write_bytes(b"zzz")
    rows = resources.generate_manifest(res)
    assert [r[0] for r in rows] == ["vg/a.d2.gbz"]
    assert rows[0][2] == 3 and rows[0][3] == "core"
    m = tmp_path / "m.tsv"
    resources.write_manifest(rows, m)
    back = resources.read_manifest(m)
    assert back == rows
    assert resources.verify(res, back) == [("vg/a.d2.gbz", "ok")]
    (res / "vg" / "a.d2.gbz").write_bytes(b"abd")
    assert resources.verify(res, back) == [("vg/a.d2.gbz", "sha256")]
    (res / "vg" / "a.d2.gbz").write_bytes(b"abcd")
    assert resources.verify(res, back) == [("vg/a.d2.gbz", "size")]
    (res / "vg" / "a.d2.gbz").unlink()
    assert resources.verify(res, back) == [("vg/a.d2.gbz", "missing")]


def test_packaged_manifest_is_consistent():
    rows = resources.read_manifest(paths.packaged_manifest())
    rels = {r[0] for r in rows}
    for suffix in paths.VG_SUFFIXES:
        assert paths.RESOURCE_FILES["vg_index"] + suffix in rels
    for key in ("annotation_gff", "annotation_source_gff", "annotation_bed"):
        assert paths.RESOURCE_FILES[key] in rels
    assert paths.CONTAINER_IMAGE in rels
    assert all(r[3] in resources.BUNDLES for r in rows)


def test_download_stub_prints_interim_instructions(tmp_path, capsys):
    import io
    buf = io.StringIO()
    rc = resources.download("core", tmp_path / "res", dry_run=True, log=buf)
    assert rc == 2  # URL still TBD
    out = buf.getvalue()
    assert "gocmd get" in out and "TBD" in out and str(tmp_path / "res") in out


def test_centrifuge_comes_from_upstream():
    import io
    assert "TBD" not in resources.BUNDLE_URLS["centrifuge"]
    assert resources.BUNDLE_URLS["centrifuge"].startswith("https://genome-idx.s3.amazonaws.com/centrifuge/")
    buf = io.StringIO()
    assert resources.download("centrifuge", Path("/nonexistent/res"), dry_run=True, log=buf) == 0


def _bundle_tar(tmp_path, members):
    import io, tarfile
    tgz = tmp_path / "bundle.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return tgz


def test_download_unpacks_upstream_tarball_into_subdir(tmp_path, monkeypatch):
    import io
    tgz = _bundle_tar(tmp_path, {"p_compressed+h+v.1.cf": b"one", "p_compressed+h+v.2.cf": b"two"})
    monkeypatch.setitem(resources.BUNDLE_URLS, "centrifuge", tgz.as_uri())
    monkeypatch.setattr(paths, "packaged_manifest", lambda: tmp_path / "no_manifest.tsv")
    res = tmp_path / "res"
    assert resources.download("centrifuge", res, log=io.StringIO()) == 0
    assert (res / "centrifuge" / "p_compressed+h+v.1.cf").read_bytes() == b"one"
    assert (res / "centrifuge" / "p_compressed+h+v.2.cf").read_bytes() == b"two"


def test_download_refuses_members_outside_target(tmp_path, monkeypatch):
    import io
    import pytest
    tgz = _bundle_tar(tmp_path, {"../escape.txt": b"x"})
    monkeypatch.setitem(resources.BUNDLE_URLS, "centrifuge", tgz.as_uri())
    with pytest.raises(ValueError, match="outside"):
        resources.download("centrifuge", tmp_path / "res", log=io.StringIO())
    assert not (tmp_path / "res" / "escape.txt").exists()


def test_cli_download_defaults_and_with_container(tmp_path, capsys):
    from ecoli_ingest.cli import main
    res = str(tmp_path / "res")
    main(["--resources", res, "download", "--dry-run"])
    err = capsys.readouterr().err
    assert "bundle   : core" in err and "bundle   : centrifuge" in err and "bundle   : container" not in err
    assert f"target   : {res}" in err          # an explicit --resources is used even before it exists
    main(["--resources", res, "download", "--dry-run", "--with-container"])
    assert "bundle   : container" in capsys.readouterr().err
    assert main(["--resources", res, "download", "--dry-run", "nonsense"]) == 2
