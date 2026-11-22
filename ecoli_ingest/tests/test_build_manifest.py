import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "atlas_build" / "preprocessing" / "build_manifest.py"


@pytest.fixture(scope="module")
def bm():
    spec = importlib.util.spec_from_file_location("build_manifest", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def run(**kw):
    base = {"study_accession": "PRJNA1", "study_alias": "GSE271069", "experiment_alias": "GSM8368193",
            "sample_alias": "GSM8368193", "run_alias": "GSM8368193_r1", "instrument_platform": "ILLUMINA",
            "library_source": "TRANSCRIPTOMIC", "fastq_ftp": "ftp://x/a.fastq.gz", "read_count": "5000000",
            "study_title": "Nutrient stress", "experiment_title": "GSM8368193: CAS P1; RNA-Seq",
            "sample_title": "CAS P1 rep1"}
    base.update(kw)
    return base


def prepared(bm, **kw):
    r = run(**kw)
    r["GSE"], r["GSM"] = bm.geo_ids(r)
    r["library_type_match"] = bm.library_type_match(r)
    return r


def test_geo_ids(bm):
    assert bm.geo_ids(run()) == ("GSE271069", "GSM8368193")
    assert bm.geo_ids(run(experiment_alias="lib1", sample_alias="s1")) == ("GSE271069", "GSM8368193")   # from run alias
    assert bm.geo_ids(run(study_alias="my study", experiment_alias="x", sample_alias="y", run_alias="z")) == ("", "")


@pytest.mark.parametrize("title,expected", [
    ("WT ribosome profiling rep1", "ribosome_profiling"),
    ("WT Ribo-seq", "ribosome_profiling"),
    ("dRNA-seq TEX+ library", "five_prime_ends"),
    ("Term-seq of WT", "three_prime_ends"),
    ("Hfq RIL-seq", "rna_interactome"),
    ("single-cell RNA-seq of persisters", "single_cell"),
    ("WT total RNA-seq, exponential phase", ""),
    ("mRNA-seq of dms mutant", ""),                # a gene name, not DMS probing
])
def test_library_type_patterns(bm, title, expected):
    assert bm.library_type_match(run(sample_title=title)) == expected


def test_study_title_alone_does_not_exclude(bm):
    r = run(study_title="Selective ribosome profiling of chaperones", sample_title="total mRNA-seq rep1")
    assert bm.library_type_match(r) == ""
    assert bm.library_type_match(r, ("study_title",)) == "ribosome_profiling"


def test_classify_order_and_flags(bm):
    assert bm.classify(prepared(bm), 500_000) == []
    flags = bm.classify(prepared(bm, instrument_platform="OXFORD_NANOPORE", read_count="10", fastq_ftp=""),
                        500_000)
    assert flags == ["not_illumina", "no_fastq", "low_reads"]
    assert bm.classify(prepared(bm, study_alias="PRJ", experiment_alias="a", sample_alias="b", run_alias="c"),
                       0) == ["no_geo_ids"]
    assert bm.classify(prepared(bm, library_source="TRANSCRIPTOMIC SINGLE CELL"), 0) == ["not_transcriptomic"]
    assert bm.classify(prepared(bm, sample_title="Ribo-seq rep1"), 0) == ["library_type"]


def test_is_ecoli(bm):
    assert bm.is_ecoli("511145", "anything", {"562", "511145"})
    assert bm.is_ecoli("999", "Escherichia coli O157:H7", {"562"})
    assert not bm.is_ecoli("1280", "Staphylococcus aureus", {"562"})
