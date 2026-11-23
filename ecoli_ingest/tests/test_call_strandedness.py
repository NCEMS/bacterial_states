import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "ecoli_ingest" / "workflow" / "scripts" / "call_strandedness.py"


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("call_strandedness", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["call_strandedness"] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_text(layout, forward, reverse, failed=0.002):
    if layout == "PE":
        return (f'\n\nThis is PairEnd Data\nFraction of reads failed to determine: {failed}\n'
                f'Fraction of reads explained by "1++,1--,2+-,2-+": {forward}\n'
                f'Fraction of reads explained by "1+-,1-+,2++,2--": {reverse}\n')
    return (f'\n\nThis is SingleEnd Data\nFraction of reads failed to determine: {failed}\n'
            f'Fraction of reads explained by "++,--": {forward}\n'
            f'Fraction of reads explained by "+-,-+": {reverse}\n')


@pytest.mark.parametrize("layout,forward,reverse,expected", [
    ("PE", 0.0103, 0.9879, "reverse"),      # GSE271069
    ("PE", 0.9915, 0.0060, "forward"),      # GSE67402
    ("SE", 0.5034, 0.4932, "unstranded"),   # tutorial data
    ("SE", 0.97, 0.02, "forward"),
    ("SE", 0.02, 0.97, "reverse"),
    ("PE", 0.79, 0.20, "unstranded"),       # just under the threshold
    ("PE", 0.80, 0.19, "forward"),          # at the threshold
])
def test_calls(cs, layout, forward, reverse, expected):
    row = cs.call_sample("s", infer_text(layout, forward, reverse))
    assert row["layout"] == layout and row["call"] == expected
    assert row["fraction_forward"] == forward and row["fraction_reverse"] == reverse
    assert row["counted"] == ("no" if expected == "unstranded" else "yes")
    assert row["call_source"] == "infer_experiment"


def test_a_stated_direction_is_checked_and_the_threshold_cannot_be_lowered(cs):
    text = infer_text("SE", 0.50, 0.49)                                  # an unstranded library
    row = cs.call_sample("s", text, mode="reverse")
    assert row["call"] == "unstranded" and row["counted"] == "no" and row["not_counted_reason"] == "unstranded"
    assert row["call_source"] == "infer_experiment, config reverse"
    row = cs.call_sample("s", infer_text("PE", 0.97, 0.02), mode="reverse")   # stranded the other way
    assert row["call"] == "forward" and row["counted"] == "no" and row["not_counted_reason"] == "strand_mismatch"
    row = cs.call_sample("s", infer_text("PE", 0.02, 0.97), mode="reverse")
    assert row["call"] == "reverse" and row["counted"] == "yes"
    for low in (0.7, 0.5):
        with pytest.raises(ValueError, match="at least 0.8"):
            cs.call_sample("s", infer_text("PE", 0.72, 0.27), min_fraction=low)
    assert cs.call_sample("s", infer_text("PE", 0.85, 0.14), min_fraction=0.9)["call"] == "unstranded"   # raising is allowed
    with pytest.raises(ValueError):
        cs.call_sample("s", text, mode="unstranded")
    with pytest.raises(ValueError):
        cs.parse_infer_experiment("Unknown Data type")


def test_cli_combine_and_flags(cs, tmp_path):
    for name, text in (("b", infer_text("PE", 0.01, 0.98)), ("a", infer_text("PE", 0.99, 0.005)),
                       ("c", infer_text("PE", 0.50, 0.49))):
        (tmp_path / f"{name}.txt").write_text(text)
        assert cs.main(["call", "--infer", str(tmp_path / f"{name}.txt"), "--sample", name,
                        "--out", str(tmp_path / f"{name}.tsv")]) == 0
    table, mqc, fail = tmp_path / "all.tsv", tmp_path / "mqc.tsv", tmp_path / "fail.json"
    rows = cs.combine([str(tmp_path / f"{n}.tsv") for n in "bac"], str(table), str(mqc), str(fail))
    assert [r["sample"] for r in rows] == ["a", "b", "c"]
    assert cs.counted_samples(str(table)) == ["a", "b"]
    assert cs.featurecounts_strand_flags(str(table), ["b", "a"]) == "2,1"       # follows the BAM order
    assert list(json.loads(fail.read_text())) == ["c"]
    assert json.loads(fail.read_text())["c"]["reason"] == "unstranded"
    assert mqc.read_text().startswith('# id: "strandedness"')
    with pytest.raises(KeyError):
        cs.featurecounts_strand_flags(str(table), ["c"])                        # unstranded samples have no -s value
    # samples refused by the species gate are not counted either
    rows = cs.combine([str(tmp_path / f"{n}.tsv") for n in "bac"], str(table), str(mqc), str(fail), {"b": "not_ecoli"})
    assert {r["sample"]: (r["counted"], r["not_counted_reason"]) for r in rows} == {
        "a": ("yes", ""), "b": ("no", "not_ecoli"), "c": ("no", "unstranded")}
    assert cs.counted_samples(str(table)) == ["a"]
    assert {k: v["reason"] for k, v in json.loads(fail.read_text()).items()} == {"b": "not_ecoli", "c": "unstranded"}


def test_check_refuses_a_lowered_strand_threshold():
    from ecoli_ingest.checks import FAIL, OK, check_strandedness
    assert check_strandedness({}).status == OK
    assert check_strandedness({"strandedness": "reverse", "strandedness_min_fraction": 0.9}).status == OK
    assert check_strandedness({"strandedness_min_fraction": 0.6}).status == FAIL
    assert check_strandedness({"strandedness": "unstranded"}).status == FAIL
