"""environment.yaml (using the model; the image) and environment-train.yaml (training: the same pins plus optuna and
pytest) pin every dependency exactly; the lock files and the model bundle agree with them."""
import json
from pathlib import Path

import pytest
import yaml

from ecoli_atlas import download

PKG = Path(__file__).resolve().parents[1]
ENV = PKG / "environment.yaml"
TRAIN_ENV = PKG / "environment-train.yaml"
LOCKS = sorted(p for p in (PKG / "locks").glob("*.txt") if not p.name.startswith("train-"))
TRAIN_LOCKS = sorted((PKG / "locks").glob("train-*.txt"))
TRAIN_ONLY = {"optuna", "pytest"}           # what the training environment adds to the use environment
TRAIN_EXTRAS = {"tune", "dev"}               # pyproject extras that belong to the training environment
# what the bundle manifest records (artifacts.environment) -> conda package of environment.yaml
MANIFEST_PACKAGES = {"python": "python", "numpy": "numpy", "torch": "pytorch-cpu"}


def _pins(path=ENV):
    pins = {}
    for dep in yaml.safe_load(path.read_text())["dependencies"]:
        assert isinstance(dep, str), f"no pip sections in {path.name}: {dep}"
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
    for path in (ENV, TRAIN_ENV):
        for name, version in _pins(path).items():
            assert version and not version.startswith(("=", "<", ">")) and "*" not in version and "," not in version, name


def test_train_environment_is_the_use_environment_plus_training():
    use, train = _pins(), _pins(TRAIN_ENV)
    assert {n: v for n, v in train.items() if n in use} == use, "environment-train.yaml repeats every use pin, same version"
    assert set(train) - set(use) == TRAIN_ONLY


def test_pyproject_dependencies_are_pinned():
    import tomllib
    proj = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]
    def names(deps):
        return [d.split(">")[0].split("<")[0].split("=")[0].strip() for d in deps]
    use_names = names(proj["dependencies"])
    train_names = []
    for extra, deps in proj["optional-dependencies"].items():
        (train_names if extra in TRAIN_EXTRAS else use_names).extend(names(deps))
    conda = {"torch": "pytorch-cpu", "matplotlib": "matplotlib-base"}      # pip name -> conda package
    assert not [n for n in use_names if conda.get(n, n) not in _pins()], "every use dependency is in environment.yaml"
    assert not [n for n in train_names if conda.get(n, n) not in _pins(TRAIN_ENV)], "tune/dev extras are in environment-train.yaml"


@pytest.mark.parametrize("lock", LOCKS + TRAIN_LOCKS, ids=[p.stem for p in LOCKS + TRAIN_LOCKS])
def test_lock_matches_pins(lock):
    have = _lock(lock)
    train = lock.name.startswith("train-")
    assert lock.read_text().splitlines()[2] == f"# platform: {lock.stem.removeprefix('train-')}"
    pins = _pins(TRAIN_ENV if train else ENV)
    bad = {n: (v, have.get(n)) for n, v in pins.items() if have.get(n) != v}
    assert not bad, f"{lock.name} disagrees with {(TRAIN_ENV if train else ENV).name}: {bad}"
    assert not [n for n in have if n.startswith(("nvidia-", "cuda-"))], "the locked torch is the CPU build"
    if not train:
        assert not TRAIN_ONLY & set(have), "the use environment (and the image) has no training-only packages"


def test_train_lock_exists():
    assert (PKG / "locks" / "train-linux-64.txt") in TRAIN_LOCKS


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


def test_image_is_the_environment_of_this_major_version():
    import hashlib
    from ecoli_atlas import __version__
    major = download.IMAGE_ENV.split(".")[0]
    assert major == __version__.split(".")[0]
    assert download.IMAGE_URL == ("https://github.com/NCEMS/bacterial_states/releases/download/"
                                  f"containers-v{major}/ecoli_atlas_env{download.IMAGE_ENV}.sif")
    # a changed lock needs a new environment id and image (release/build_container.sh refuses otherwise)
    assert hashlib.sha256((PKG / "locks" / "linux-64.txt").read_bytes()).hexdigest() == download.IMAGE_LOCK_SHA256, (
        "locks/linux-64.txt changed: set a new IMAGE_ENV and IMAGE_LOCK_SHA256 in download.py and build its image")
