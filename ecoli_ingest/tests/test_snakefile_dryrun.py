"""Dry-run the packaged Snakefile on a run directory that mixes one-file, multi-file and
accession samples. Needs snakemake on PATH (the ecoli_ingest conda env); skipped otherwise."""
import json
import shutil
import subprocess
import textwrap

import pytest

from ecoli_ingest import paths

pytestmark = pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not on PATH")

MANIFEST = {
    "accession": "GSM1646283", "kind": "geo_sample", "layout": "PAIRED", "resolved_at": "2026-11-09T00:00:00Z",
    "runs": [{"run": "SRR1945181", "layout": "PAIRED", "ena_layout": "PAIRED", "files": {
        "1": {"url": "https://x/SRR1945181_1.fastq.gz", "md5": "a" * 32, "bytes": 1},
        "2": {"url": "https://x/SRR1945181_2.fastq.gz", "md5": "b" * 32, "bytes": 2}}}],
}


def _fake_resources(root):
    for rel in paths.RESOURCE_FILES.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
    for suffix in paths.VG_SUFFIXES:
        (root / (paths.RESOURCE_FILES["vg_index"] + suffix)).write_bytes(b"")
    for suffix in paths.CENTRIFUGE_SUFFIXES:
        (root / (paths.RESOURCE_FILES["centrifuge_index_path"] + suffix)).write_bytes(b"")
    for key in ("annotation_gff", "annotation_source_gff", "annotation_bed"):
        (root / paths.RESOURCE_FILES[key]).write_bytes(b"")


CONFIG_PAIRED = """\
experiment: "t"
samples:
  one: {r1: "reads/one.fq.gz", r2: "reads/one_2.fq.gz", condition: "a"}
  two: {r1: ["reads/two_a.fq.gz", "reads/two_b.fq.gz"], r2: ["reads/two_a_2.fq.gz", "reads/two_b_2.fq.gz"], condition: "a"}
  dl:  {accession: "GSM1646283", condition: "b"}
deseq2: {contrasts: [["condition", "a", "b"]]}
centrifuge: false
fastp_min_kept_percentage: 75.0   # without a gate the checkpoint has no inputs and the dry run shows only it
"""
CONFIG_MIXED = CONFIG_PAIRED.replace(', r2: "reads/one_2.fq.gz"', "")


@pytest.fixture
def rundir(tmp_path):
    run = tmp_path / "run"
    (run / "reads").mkdir(parents=True)
    for name in ("one", "two_a", "two_b"):
        (run / "reads" / f"{name}.fq.gz").write_bytes(b"")
        (run / "reads" / f"{name}_2.fq.gz").write_bytes(b"")
    (run / "sra" / "manifest").mkdir(parents=True)
    (run / "sra" / "manifest" / "GSM1646283.json").write_text(json.dumps(MANIFEST))
    (run / "config.yaml").write_text(CONFIG_PAIRED)
    _fake_resources(tmp_path / "res")
    return run


def _dry_run(rundir, *extra):
    cmd = ["snakemake", "--snakefile", str(paths.snakefile()), "--directory", str(rundir),
           "--configfile", str(rundir / "config.yaml"), "--config", f"resources_dir={rundir.parent / 'res'}",
           "--cores", "4", "--dry-run", "--quiet", "rules", "all", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def _job_counts(res):
    counts = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            counts[parts[0]] = int(parts[1])
    return counts


def test_dry_run_paired_downloads_and_merges(rundir):
    res = _dry_run(rundir)
    assert res.returncode == 0, res.stdout + res.stderr
    counts = _job_counts(res)
    assert counts.get("sra_download") == 2          # SRR1945181_1 and _2
    assert counts.get("merge_reads") == 2           # sample `two`, both mates
    assert counts.get("fastp_pe") == 3 and "fastp_se" not in counts


def test_dry_run_mixed_layout_fetches_mate_one_only(rundir):
    (rundir / "config.yaml").write_text(CONFIG_MIXED)
    res = _dry_run(rundir)
    assert res.returncode == 0, res.stdout + res.stderr
    counts = _job_counts(res)
    assert counts.get("sra_download") == 1          # single-end mode ignores r2, so _2 is never fetched
    assert counts.get("merge_reads") == 1
    assert counts.get("fastp_se") == 3 and "fastp_pe" not in counts
    assert "Mixed SE and PE" in res.stdout + res.stderr


def test_dry_run_fails_on_study_accession(rundir):
    (rundir / "config.yaml").write_text(CONFIG_PAIRED.replace('accession: "GSM1646283"', 'accession: "GSE67218"'))
    res = _dry_run(rundir)
    assert res.returncode != 0 and "study-level" in res.stderr + res.stdout
