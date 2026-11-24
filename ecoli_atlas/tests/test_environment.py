"""environment.yaml pins every dependency exactly; the lock files and the model bundle agree with it."""
import json
from pathlib import Path

import pytest
import yaml

from ecoli_atlas import download

PKG = Path(__file__).resolve().parents[1]
ENV = PKG / "environment.yaml"
LOCKS = sorted((PKG / "locks").glob("*.txt"))
# what the bundle manifest records (artifacts.environment) -> conda package of environment.yaml
MANIFEST_PACKAGES = {"python": "python", "numpy": "numpy", "torch": "pytorch-cpu"}


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


def test_pyproject_dependencies_are_pinned():
    import tomllib
    proj = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]
    names = [d.split(">")[0].split("<")[0].split("=")[0].strip() for d in proj["dependencies"]]
    for extra in proj["optional-dependencies"].values():
        names += [d.split(">")[0].split("<")[0].split("=")[0].strip() for d in extra]
    conda = {"torch": "pytorch-cpu", "matplotlib": "matplotlib-base"}      # pip name -> conda package
    pins = _pins()
    assert not [n for n in names if conda.get(n, n) not in pins], "every pyproject dependency is in environment.yaml"


@pytest.mark.parametrize("lock", LOCKS, ids=[p.stem for p in LOCKS])
def test_lock_matches_pins(lock):
    have = _lock(lock)
    assert lock.read_text().splitlines()[2] == f"# platform: {lock.stem}"
    bad = {n: (v, have.get(n)) for n, v in _pins().items() if have.get(n) != v}
    assert not bad, f"{lock.name} disagrees with environment.yaml: {bad}"
    assert not [n for n in have if n.startswith(("nvidia-", "cuda-"))], "the locked torch is the CPU build"


def test_linux_lock_exists():
    assert (PKG / "locks" / "linux-64.txt") in LOCKS        # the image is built from it


def test_cuda_requirements_pin_the_same_torch():
    lines = [l.split("\\")[0].strip() for l in (PKG / "requirements-cuda.txt").read_text().splitlines()]
    torch = [l for l in lines if l.startswith("torch==")]
    assert len(torch) == 1 and torch[0].split("==")[1].split("+")[0] == _pins()["pytorch-cpu"]
    text = (PKG / "requirements-cuda.txt").read_text()
    assert text.count("--hash=sha256:") >= sum(1 for l in lines if "==" in l)


def test_pins_match_the_model_bundle():
    manifest = download.default_models_dir() / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"no model bundle at {manifest.parent}")
    env = json.loads(manifest.read_text()).get("environment", {})
    pins = _pins()
    for key, package in MANIFEST_PACKAGES.items():
        if key in env:
            assert str(env[key]).split("+")[0] == pins[package], f"bundle {key} {env[key]}, pinned {pins[package]}"
