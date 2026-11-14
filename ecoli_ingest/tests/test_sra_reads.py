"""workflow/scripts/sra_reads.py: accession resolution (offline, with captured NCBI/ENA replies),
sample normalisation and verified downloads."""
import hashlib
import json
from pathlib import Path

import pytest

from ecoli_ingest import paths

RUNINFO_GSM = (
    "Run,ReleaseDate,spots,bases,LibraryLayout,Platform,SRAStudy,Sample,BioSample,SampleName\n"
    "SRR1945181,2015-06-16 10:51:13,2154697,435248794,PAIRED,ILLUMINA,SRP056663,SRS886978,SAMN03450852,GSM1646283\n"
)
ENA_PAIRED = (
    "run_accession\tlibrary_layout\tfastq_ftp\tfastq_md5\tfastq_bytes\n"
    "SRR1945181\tPAIRED\tftp.sra.ebi.ac.uk/vol1/fastq/SRR194/001/SRR1945181/SRR1945181_1.fastq.gz;"
    "ftp.sra.ebi.ac.uk/vol1/fastq/SRR194/001/SRR1945181/SRR1945181_2.fastq.gz\t"
    "6c134f8a188f0171343fdd9d748fba0e;890c0010140542e1c4e1bcdba6c84467\t119494889;124362910\n"
)
ENA_SINGLE = (
    "run_accession\tlibrary_layout\tfastq_ftp\tfastq_md5\tfastq_bytes\n"
    "SRR2547467\tSINGLE\tftp.sra.ebi.ac.uk/vol1/fastq/SRR254/007/SRR2547467/SRR2547467.fastq.gz\t"
    "0123456789abcdef0123456789abcdef\t3233688793\n"
)
# paired run with an extra orphan-read file: the _1/_2 pair must win
ENA_THREE = (
    "run_accession\tlibrary_layout\tfastq_ftp\tfastq_md5\tfastq_bytes\n"
    "SRR000001\tPAIRED\tx/SRR000001/SRR000001.fastq.gz;x/SRR000001/SRR000001_1.fastq.gz;x/SRR000001/SRR000001_2.fastq.gz\t"
    "a;b;c\t1;2;3\n"
)
ENA_NO_FASTQ = "run_accession\tlibrary_layout\tfastq_ftp\tfastq_md5\tfastq_bytes\nSRR000002\tSINGLE\t\t\t\n"


@pytest.fixture(scope="module")
def sra():
    return paths.load_script("sra_reads")


@pytest.fixture
def fake_net(sra, monkeypatch):
    calls = []

    def fetch(url, timeout=60, attempts=3):
        calls.append(url)
        if "runinfo?acc=GSM1646283" in url:
            return RUNINFO_GSM
        if "accession=SRR1945181" in url:
            return ENA_PAIRED
        if "accession=SRR2547467" in url:
            return ENA_SINGLE
        if "accession=SRR000001" in url:
            return ENA_THREE
        if "accession=SRR000002" in url:
            return ENA_NO_FASTQ
        if "runinfo?acc=SRX000000" in url:
            return "Run,LibraryLayout\n"
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(sra, "_fetch", fetch)
    return calls


def test_classify(sra):
    assert sra.classify("SRR1945181") == "run" and sra.classify("ERR123") == "run"
    assert sra.classify("SRX969365") == "experiment"
    assert sra.classify(" GSM1646283 ") == "geo_sample"
    assert sra.classify("SAMN03450852") == "sample" and sra.classify("SAMEA123456") == "sample"
    for study in ("GSE67218", "SRP056663", "PRJNA297600"):
        with pytest.raises(sra.AccessionError, match="study-level"):
            sra.classify(study)
    with pytest.raises(sra.AccessionError, match="not a recognised"):
        sra.classify("reads/WT_1.fastq.gz")


def test_resolve_gsm_is_cached(sra, fake_net, tmp_path):
    rec = sra.resolve_accession("GSM1646283", str(tmp_path / "manifest"))
    assert rec["layout"] == "PAIRED" and [r["run"] for r in rec["runs"]] == ["SRR1945181"]
    f = rec["runs"][0]["files"]
    assert f["1"]["url"].startswith("https://ftp.sra.ebi.ac.uk/") and f["1"]["url"].endswith("_1.fastq.gz")
    assert f["2"]["md5"] == "890c0010140542e1c4e1bcdba6c84467" and f["2"]["bytes"] == 124362910
    assert len(fake_net) == 2  # NCBI hop + one ENA report
    again = sra.resolve_accession("GSM1646283", str(tmp_path / "manifest"))
    assert again == rec and len(fake_net) == 2  # served from the cache
    assert json.loads((tmp_path / "manifest" / "GSM1646283.json").read_text())["accession"] == "GSM1646283"


def test_resolve_run_single_and_odd_listings(sra, fake_net, tmp_path):
    rec = sra.resolve_accession("SRR2547467", str(tmp_path))
    assert rec["layout"] == "SINGLE" and set(rec["runs"][0]["files"]) == {"1"}
    assert fake_net == [sra.ENA_FILEREPORT.format(acc="SRR2547467")]  # runs skip the NCBI hop
    three = sra.resolve_accession("SRR000001", str(tmp_path))
    assert three["layout"] == "PAIRED" and three["runs"][0]["files"]["1"]["url"].endswith("_1.fastq.gz")
    archive_only = sra.resolve_accession("SRR000002", str(tmp_path))          # ENA lists no FASTQ: sra-tools only
    assert archive_only["layout"] == "SINGLE" and archive_only["runs"][0]["files"] == {"1": {"url": None, "md5": None, "bytes": None}}
    with pytest.raises(sra.AccessionError, match="no SRA runs"):
        sra.resolve_accession("SRX000000", str(tmp_path))


def test_normalize_sample(sra):
    assert sra.normalize_sample("a", {"r1": "a.fq.gz"}) == {"r1": ["a.fq.gz"], "r2": None, "accessions": []}
    two = sra.normalize_sample("b", {"r1": ["l1.fq.gz", "l2.fq.gz"], "r2": ["l1_2.fq.gz", "l2_2.fq.gz"]})
    assert len(two["r1"]) == 2 and len(two["r2"]) == 2
    assert sra.normalize_sample("c", {"accession": "GSM1646283"})["accessions"] == ["GSM1646283"]
    for bad, msg in (
        ({"r1": "a.fq", "accession": "SRR1"}, "not both"),
        ({"condition": "x"}, "needs an 'r1'"),
        ({"r1": ["a.fq", "b.fq"], "r2": "c.fq"}, "r1 has 2"),
        ({"r1": ["a.fq.gz", "b.fq"]}, "mixes gzipped"),
        ({"accession": "GSE67218"}, "study-level"),
        ("a.fq", "must be a mapping"),
    ):
        with pytest.raises(sra.AccessionError, match=msg):
            sra.normalize_sample("s", bad)


def test_resolve_samples_points_at_sra_dir(sra, fake_net, tmp_path):
    cfg = {
        "wt": {"r1": "reads/wt.fq.gz", "condition": "a"},
        "ko": {"accession": "GSM1646283", "condition": "b"},
        "se": {"accession": ["SRR2547467"], "condition": "b"},
    }
    reads = sra.resolve_samples(cfg, "sra", str(tmp_path / "m"))
    assert reads["wt"] == {"r1": ["reads/wt.fq.gz"], "r2": None, "accessions": [], "runs": {}}
    assert reads["ko"]["r1"] == ["sra/SRR1945181/SRR1945181_1.fastq.gz"]
    assert reads["ko"]["r2"] == ["sra/SRR1945181/SRR1945181_2.fastq.gz"]
    assert set(reads["ko"]["runs"]["SRR1945181"]) == {"1", "2"}
    assert reads["se"]["r2"] is None and reads["se"]["r1"] == ["sra/SRR2547467/SRR2547467_1.fastq.gz"]
    summ = sra.download_summary(reads, str(tmp_path / "sra"))
    assert summ == {"runs": 2, "files": 3, "present": 0, "pending_bytes": 119494889 + 124362910 + 3233688793,
                    "no_ena": []}
    (tmp_path / "sra" / "SRR2547467").mkdir(parents=True)
    (tmp_path / "sra" / "SRR2547467" / "SRR2547467_1.fastq.gz").write_bytes(b"")
    assert sra.download_summary(reads, str(tmp_path / "sra"))["present"] == 1


def test_download_verifies_md5_and_size(sra, tmp_path):
    src = tmp_path / "src.fastq.gz"
    src.write_bytes(b"@r1\nACGT\n+\nIIII\n" * 100)
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    out = tmp_path / "sra" / "SRR1" / "SRR1_1.fastq.gz"
    sra.download(src.as_uri(), str(out), md5=md5, size=src.stat().st_size)
    assert out.read_bytes() == src.read_bytes() and not out.with_name(out.name + ".part").exists()
    bad = tmp_path / "bad.fastq.gz"
    with pytest.raises(sra.DownloadError, match="md5"):
        sra.download(src.as_uri(), str(bad), md5="0" * 32, attempts=1)
    assert not bad.exists() and not Path(str(bad) + ".part").exists()
    with pytest.raises(sra.DownloadError, match="bytes"):
        sra.download(src.as_uri(), str(bad), size=1, attempts=1)
    with pytest.raises(sra.DownloadError):
        sra.download((tmp_path / "missing").as_uri(), str(bad), attempts=1)


def test_cli_download(sra, tmp_path, capsys):
    src = tmp_path / "x.fq.gz"
    src.write_bytes(b"hello")
    out = tmp_path / "out.fq.gz"
    assert sra.main(["download", "--url", src.as_uri(), "--out", str(out), "--md5", "", "--bytes", "5"]) == 0
    assert out.read_bytes() == b"hello"
    assert sra.main(["download", "--url", src.as_uri(), "--out", str(out), "--md5", "f" * 32]) == 1
    assert "md5" in capsys.readouterr().err


def test_download_retries_transient_http_errors(sra, tmp_path, monkeypatch):
    """ENA answers 403 for a while when throttling; the same URL works again later. 404 is final."""
    import io
    import urllib.error
    calls = []

    def fake_urlopen(codes, payload=b"data"):
        def urlopen(req, timeout=None):
            calls.append(req.full_url)
            code = codes[min(len(calls) - 1, len(codes) - 1)]
            if code != 200:
                raise urllib.error.HTTPError(req.full_url, code, "nope", None, None)
            return io.BytesIO(payload)
        return urlopen

    out = tmp_path / "a.fastq.gz"
    monkeypatch.setattr(sra.urllib.request, "urlopen", fake_urlopen([403, 403, 200]))
    sra.download("https://example.org/a.fastq.gz", str(out), size=4, backoff=0)
    assert out.read_bytes() == b"data" and len(calls) == 3

    calls.clear()
    monkeypatch.setattr(sra.urllib.request, "urlopen", fake_urlopen([404]))
    with pytest.raises(sra.DownloadError, match="404"):
        sra.download("https://example.org/b.fastq.gz", str(tmp_path / "b.fastq.gz"), backoff=0)
    assert len(calls) == 1                                           # no second attempt

    calls.clear()
    monkeypatch.setattr(sra.urllib.request, "urlopen", fake_urlopen([403]))
    with pytest.raises(sra.DownloadError, match="403"):
        sra.download("https://example.org/c.fastq.gz", str(tmp_path / "c.fastq.gz"), attempts=3, backoff=0)
    assert len(calls) == 3 and not (tmp_path / "c.fastq.gz.part").exists()


# ---- fetch: sra-tools first, ENA as the backup ---------------------------------------------

FASTQ_1 = b"".join(b"@SRR9.%d r%d/1\nACGT\n+\nIIII\n" % (i, i) for i in range(1, 6))
FASTQ_2 = b"".join(b"@SRR9.%d r%d/2\nTTTT\n+\nIIII\n" % (i, i) for i in range(1, 6))


def _fake_sra_tools(bindir, mode="paired"):
    """Stand-ins for prefetch / fasterq-dump that write what the real tools would."""
    bindir.mkdir(exist_ok=True)
    (bindir / "prefetch").write_text(
        "#!/bin/sh\nrun=$1; out=$3\n" + ("exit 3\n" if mode == "prefetch_fails" else
        'mkdir -p "$out/$run" && echo archive > "$out/$run/$run.sra"\n'))
    body = {"paired": 'cp "$FAKE_SRC/r1.fastq" "$out/${run}_1.fastq"; cp "$FAKE_SRC/r2.fastq" "$out/${run}_2.fastq"',
            "single": 'cp "$FAKE_SRC/r1.fastq" "$out/${run}.fastq"',
            "unequal": 'cp "$FAKE_SRC/r1.fastq" "$out/${run}_1.fastq"; head -4 "$FAKE_SRC/r2.fastq" > "$out/${run}_2.fastq"',
            "prefetch_fails": "exit 1"}[mode]
    (bindir / "fasterq-dump").write_text(
        "#!/bin/sh\narchive=$1; run=$(basename \"$archive\" .sra)\n"
        'while [ $# -gt 0 ]; do [ "$1" = "--outdir" ] && out=$2; shift; done\n' + body + "\n")
    for tool in ("prefetch", "fasterq-dump"):
        (bindir / tool).chmod(0o755)


@pytest.fixture
def fetch_env(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "r1.fastq").write_bytes(FASTQ_1)
    (src / "r2.fastq").write_bytes(FASTQ_2)
    ena = {}
    for mate, data in (("1", FASTQ_1), ("2", FASTQ_2)):
        f = src / f"ena_{mate}.fastq.gz"
        f.write_bytes(b"ENA" + data)                                  # not real gzip: only identity matters here
        ena[mate] = {"url": f.as_uri(), "md5": hashlib.md5(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
    monkeypatch.setenv("FAKE_SRC", str(src))

    def use_tools(mode):
        bindir = tmp_path / "bin"
        if mode is None:
            monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        else:
            _fake_sra_tools(bindir, mode)
            import os
            monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    return tmp_path / "sra" / "SRR9", ena, use_tools


def _gunzip(path):
    import gzip
    return gzip.decompress(Path(path).read_bytes())


def test_fetch_prefers_sra_tools(sra, fetch_env):
    out, ena, use_tools = fetch_env
    use_tools("paired")
    assert sra.sra_tools_available()
    assert sra.fetch_run("SRR9", str(out), ("1", "2"), "auto", ena, threads=1, spots=5) == "sra-tools"
    assert _gunzip(out / "SRR9_1.fastq.gz") == FASTQ_1 and _gunzip(out / "SRR9_2.fastq.gz") == FASTQ_2
    assert sorted(p.name for p in out.iterdir()) == ["SRR9_1.fastq.gz", "SRR9_2.fastq.gz"]     # no temp files left


def test_fetch_mate_one_only_and_single_end_archive(sra, fetch_env):
    out, ena, use_tools = fetch_env
    use_tools("paired")
    assert sra.fetch_run("SRR9", str(out), ("1",), "sra-tools", None, threads=1) == "sra-tools"
    assert [p.name for p in out.iterdir()] == ["SRR9_1.fastq.gz"]
    use_tools("single")                                                 # archive holds one read per spot
    out2 = out.parent / "SRR8"
    assert sra.fetch_run("SRR8", str(out2), ("1",), "sra-tools", None, threads=1) == "sra-tools"
    assert _gunzip(out2 / "SRR8_1.fastq.gz") == FASTQ_1
    with pytest.raises(sra.DownloadError, match="no mate 2"):
        sra.fetch_run("SRR8", str(out2), ("1", "2"), "sra-tools", None, threads=1)


@pytest.mark.parametrize("mode", [None, "prefetch_fails", "unequal"])
def test_fetch_falls_back_to_ena(sra, fetch_env, mode):
    """No tools, a failing prefetch, or mates of different length: `auto` takes the ENA files."""
    out, ena, use_tools = fetch_env
    use_tools(mode)
    assert sra.fetch_run("SRR9", str(out), ("1", "2"), "auto", ena, threads=1) == "ena"
    assert (out / "SRR9_1.fastq.gz").read_bytes() == b"ENA" + FASTQ_1
    assert sorted(p.name for p in out.iterdir()) == ["SRR9_1.fastq.gz", "SRR9_2.fastq.gz"]
    if mode is not None:
        with pytest.raises(sra.DownloadError):                          # sra-tools only: no fallback
            sra.fetch_run("SRR9", str(out.parent / "x"), ("1", "2"), "sra-tools", ena, threads=1)


def test_fetch_checks_spots_methods_and_missing_backup(sra, fetch_env):
    out, ena, use_tools = fetch_env
    use_tools("paired")
    with pytest.raises(sra.DownloadError, match="ENA lists 99"):
        sra.fetch_run("SRR9", str(out), ("1", "2"), "sra-tools", ena, threads=1, spots=99)
    assert sra.fetch_run("SRR9", str(out), ("1", "2"), "ena", ena) == "ena"          # tools present but not used
    assert (out / "SRR9_2.fastq.gz").read_bytes() == b"ENA" + FASTQ_2
    use_tools(None)
    no_url = {"1": {"url": None, "md5": None, "bytes": None}}
    with pytest.raises(sra.DownloadError, match="ENA has no FASTQ file"):
        sra.fetch_run("SRR9", str(out.parent / "y"), ("1",), "auto", no_url)
    assert sra.check_method(None) == "auto" and sra.check_method("SRA_tools") == "sra-tools"
    with pytest.raises(sra.AccessionError, match="sra_method"):
        sra.check_method("aspera")


def test_cli_fetch(sra, fetch_env, capsys):
    out, ena, use_tools = fetch_env
    use_tools(None)
    assert sra.main(["fetch", "--run", "SRR9", "--out-dir", str(out), "--mates", "1,2", "--method", "auto",
                     "--ena", json.dumps(ena), "--spots", "0"]) == 0
    assert "fetched from ena" in capsys.readouterr().err
    assert sra.main(["fetch", "--run", "SRR9", "--out-dir", str(out), "--method", "sra-tools"]) == 1
