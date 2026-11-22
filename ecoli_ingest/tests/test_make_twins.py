import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "atlas_build" / "twins" / "make_twins.py"


@pytest.fixture(scope="module")
def mt():
    spec = importlib.util.spec_from_file_location("make_twins", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_twins"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_fastq(path, n, length=60, mate=1):
    path.write_text("".join(f"@r.{i} x/{mate}\n{'ACGT' * (length // 4)}\n+\n{'F' * length}\n" for i in range(1, n + 1)))


def read_fastq(path):
    lines = Path(path).read_text().splitlines()
    return [(lines[i], lines[i + 1], lines[i + 3]) for i in range(0, len(lines), 4)]


def test_parse_variant(mt):
    assert mt.parse_variant("t50") == {"truncate": 50, "single_end": False, "depth": None, "dna_pct": None}
    assert mt.parse_variant("se") == {"truncate": None, "single_end": True, "depth": None, "dna_pct": None}
    assert mt.parse_variant("se_t50")["truncate"] == 50 and mt.parse_variant("se_t50")["single_end"]
    assert mt.parse_variant("d250k")["depth"] == 250_000 and mt.parse_variant("d1M")["depth"] == 1_000_000
    assert mt.parse_variant("g5")["dna_pct"] == 5.0
    for bad in ("", "x3", "t", "g"):
        with pytest.raises(ValueError):
            mt.parse_variant(bad)


def test_truncate_single_end_and_subsample(mt, tmp_path):
    r1, r2 = tmp_path / "a_1.fastq", tmp_path / "a_2.fastq"
    write_fastq(r1, 200)
    write_fastq(r2, 200, mate=2)
    info = mt.make_twin(str(r1), str(r2), str(tmp_path / "s__t36"), "t36")
    a, b = read_fastq(info["r1"]), read_fastq(info["r2"])
    assert len(a) == len(b) == 200 and all(len(s) == 36 and len(q) == 36 for _, s, q in a + b)
    info = mt.make_twin(str(r1), str(r2), str(tmp_path / "s__se_t50"), "se_t50")
    assert info["r2"] is None and all(len(s) == 50 for _, s, _ in read_fastq(info["r1"]))
    info = mt.make_twin(str(r1), str(r2), str(tmp_path / "s__d50"), "d50", seed=3)
    a, b = read_fastq(info["r1"]), read_fastq(info["r2"])
    assert 25 < len(a) < 80 and [x[0][:-2] for x in a] == [x[0][:-2] for x in b]      # mates stay together
    again = mt.make_twin(str(r1), str(r2), str(tmp_path / "t__d50"), "d50", seed=3)
    assert again["reads"] == info["reads"]                                              # the seed fixes the choice


def test_genomic_spike(mt, tmp_path):
    import random
    rng = random.Random(5)
    genome = "".join(rng.choice("ACGT") for _ in range(5000))
    (tmp_path / "g.fa").write_text(">chr test\n" + "\n".join(genome[i:i + 70] for i in range(0, 5000, 70)) + "\n>other\nAAAA\n")
    assert mt.read_genome(str(tmp_path / "g.fa")) == genome
    r1, r2 = tmp_path / "a_1.fastq", tmp_path / "a_2.fastq"
    write_fastq(r1, 950)
    write_fastq(r2, 950, mate=2)
    info = mt.make_twin(str(r1), str(r2), str(tmp_path / "s__g5"), "g5", genome=genome, fragment_sizes=[[150, 1], [300, 1]])
    assert info["added_genomic_reads"] == 50 and info["genomic_fraction_of_reads"] == pytest.approx(0.05)
    a, b = read_fastq(info["r1"]), read_fastq(info["r2"])
    assert len(a) == len(b) == 1000
    sim1, sim2 = [x for x in a if x[0].startswith("@SIMDNA")], [x for x in b if x[0].startswith("@SIMDNA")]
    assert len(sim1) == 50 and all(len(s) == 60 for _, s, _ in sim1)                    # the source's read length
    doubled = genome + genome
    n_plus = 0
    for (_, s1, _), (_, s2, _) in zip(sim1, sim2):
        fwd = s1 in doubled and mt.revcomp(s2) in doubled                              # fragment on the + strand
        rev = mt.revcomp(s1) in doubled and s2 in doubled
        assert fwd or rev
        n_plus += fwd
    assert 10 < n_plus < 40                                                            # both strands
    with pytest.raises(ValueError):
        mt.make_twin(str(r1), str(r2), str(tmp_path / "x__g5"), "g5")
