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
