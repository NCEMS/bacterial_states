"""environment.yaml pins every tool exactly, and the lock files and the image check agree with it."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ecoli_ingest import container, paths

PKG = Path(__file__).resolve().parents[1]
ENV = PKG / "environment.yaml"
LOCKS = sorted((PKG / "locks").glob("*.txt"))

# executable the workflow calls -> conda package that provides it
EXECUTABLE_PACKAGES = {
    "snakemake": "snakemake-minimal", "python": "python", "vg": "vg", "fastp": "fastp", "fastqc": "fastqc",
    "centrifuge": "centrifuge", "centrifuge-kreport": "centrifuge", "featureCounts": "subread",
    "infer_experiment.py": "rseqc", "geneBody_coverage.py": "rseqc", "bedtools": "bedtools",
    "samtools": "samtools", "multiqc": "multiqc", "Rscript": "r-base",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pins():
    pins = {}
    for dep in yaml.safe_load(ENV.read_text())["dependencies"]:
        assert isinstance(dep, str), f"no pip sections in {ENV.name}: {dep}"
        name, sep, version = dep.split("#")[0].strip().partition("=")
        pins[name] = version if sep else None
    return pins


def _lock(path):
    out = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "@")):
            stem = line.split("#")[0].rsplit("/", 1)[1].removesuffix(".conda").removesuffix(".tar.bz2")
            name, version, _build = stem.rsplit("-", 2)
            out[name] = version
    return out


def test_every_dependency_is_pinned_to_one_version():
    for name, version in _pins().items():
        assert version and not version.startswith(("=", "<", ">")) and "*" not in version and "," not in version, name


def test_every_required_executable_has_a_pinned_package():
    assert set(EXECUTABLE_PACKAGES) == set(container.TOOLS_REQUIRED)
    pins = _pins()
    for exe, package in EXECUTABLE_PACKAGES.items():
        assert pins.get(package), f"{exe}: {package} not pinned"


@pytest.mark.parametrize("lock", LOCKS, ids=[p.stem for p in LOCKS])
def test_lock_matches_pins(lock):
    have = _lock(lock)
    assert lock.read_text().splitlines()[2] == f"# platform: {lock.stem}"
    bad = {n: (v, have.get(n)) for n, v in _pins().items() if have.get(n) != v}
    assert not bad, f"{lock.name} disagrees with environment.yaml: {bad}"
    assert all("#" in line for line in lock.read_text().splitlines()[4:] if line), "every URL has its MD5"


def test_linux_lock_exists():
    assert (PKG / "locks" / "linux-64.txt") in LOCKS        # the image is built from it


def test_image_check_covers_every_recorded_tool():
    sample_qc = _load("sample_qc", paths.scripts_dir() / "sample_qc.py")
    check = _load("check_versions", PKG / "build" / "check_versions.py")
    assert set(check.TOOL_PACKAGES) == set(sample_qc.VERSION_COMMANDS)
    pins = check.read_pins(str(ENV))
    # the versions the atlas recount recorded (work/atlas_full/results/*/results/tool_versions.json)
    recount = {"fastp": "1.4.0", "fastqc": "0.12.1", "centrifuge": "1.0.4", "vg": "1.63.1", "samtools": "1.25",
               "bedtools": "2.31.1", "featureCounts": "2.1.1", "rseqc": "5.0.6", "multiqc": "1.26",
               "snakemake": "9.27.1", "R": "4.5.3"}
    assert check.mismatches(recount, pins) == []
    assert check.mismatches(recount | {"fastp": "0.26.0"}, pins) == ["fastp: recorded 0.26.0, fastp pinned 1.4.0"]
    assert check.matches("1.0.4", "1.0.4.2")                       # prefix at a dot only
    assert not check.matches("1.0.4.2", "1.0.4") and not check.matches("1.0.", "1.0.4")
    assert not check.matches("1.0.4", "1.0.42")


def test_check_versions_cli(tmp_path):
    check = _load("check_versions", PKG / "build" / "check_versions.py")
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps({"versions": {t: None for t in check.TOOL_PACKAGES}}))
    assert check.main(["--env", str(ENV), "--versions", str(vfile)]) == 1
