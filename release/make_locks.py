"""Write the conda lock files (explicit package lists) of ecoli_ingest and ecoli_atlas.

    python release/make_locks.py ecoli_ingest --platform linux-64 --from-env ecoli_ingest
    python release/make_locks.py ecoli_ingest --platform osx-64 --platform osx-arm64
    python release/make_locks.py ecoli_atlas  --platform linux-64 --platform osx-64 --platform osx-arm64

A lock is `<package>/locks/<platform>.txt` in conda's explicit format (one package URL with its MD5 per
line), installable with `conda create -n NAME --file <lock>` or `micromamba create -n NAME -f <lock>`
without a solve. Each lock is either

  * solved from the pinned `<package>/environment.yaml` for the platform (default; a dry-run solve with an
    empty package cache, strict channel priority, the yaml's channels only), or
  * exported from an existing environment (`--from-env NAME`, linux-64 only), so that the lock is the exact
    environment that produced a result (the inGEST linux-64 lock is the environment of the atlas recount).

Either way every direct dependency of the yaml must appear in the lock at its pinned version, or nothing is
written. Packages installed with pip are never part of a lock.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("ecoli_ingest", "ecoli_atlas")
PLATFORMS = ("linux-64", "osx-64", "osx-arm64")
# virtual packages the solve assumes: glibc of RHEL 8 / Debian 11 class systems; macOS 12, the oldest that the
# pinned conda-forge builds support (pytorch 2.14.1 needs 12, scipy / scikit-learn 11), on Intel and Apple Silicon
OVERRIDES = {
    "linux-64": {"CONDA_OVERRIDE_GLIBC": "2.28"},
    "osx-64": {"CONDA_OVERRIDE_OSX": "12.0"},
    "osx-arm64": {"CONDA_OVERRIDE_OSX": "12.0"},
}


def read_env(path: Path) -> tuple[list[str], dict[str, str]]:
    """Channels and {name: pinned version} of an environment.yaml; refuses unpinned or pip entries."""
    env = yaml.safe_load(path.read_text())
    pins: dict[str, str] = {}
    for dep in env["dependencies"]:
        if not isinstance(dep, str):
            raise SystemExit(f"{path}: pip sections are not allowed in a locked environment ({dep})")
        name, sep, version = dep.split("#")[0].strip().partition("=")
        if not sep or not version or version.startswith(("=", "<", ">")) or "*" in version:
            raise SystemExit(f"{path}: {dep!r} is not pinned to one version (name=version)")
        pins[name] = version
    return list(env["channels"]), pins


def lock_entries(lines: list[str]) -> dict[str, str]:
    """{name: version} from the URL lines of an explicit lock."""
    out = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        fn = line.split("#")[0].rsplit("/", 1)[1]
        stem = fn.removesuffix(".conda").removesuffix(".tar.bz2")
        name, version, _build = stem.rsplit("-", 2)
        out[name] = version
    return out


def check_pins(pins: dict[str, str], lines: list[str]) -> list[str]:
    have = lock_entries(lines)
    return [f"{n}: pinned {v}, lock has {have.get(n, 'nothing')}" for n, v in pins.items() if have.get(n) != v]


def conda_exe() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not exe:
        raise SystemExit("conda not found (set CONDA_EXE)")
    return exe


def solve(channels: list[str], pins: dict[str, str], platform: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="make_locks_") as tmp:
        env = dict(os.environ, CONDA_PKGS_DIRS=str(Path(tmp) / "pkgs"), **OVERRIDES[platform])
        cmd = [conda_exe(), "create", "--dry-run", "--json", "--quiet", "--override-channels",
               "--strict-channel-priority", "--platform", platform, "--prefix", str(Path(tmp) / "env")]
        for ch in channels:
            cmd += ["-c", ch]
        cmd += [f"{n}={v}" for n, v in pins.items()]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        try:
            out = json.loads(res.stdout)
        except json.JSONDecodeError:
            raise SystemExit(f"solve for {platform} failed:\n{res.stderr or res.stdout}")
        if res.returncode or "actions" not in out:
            raise SystemExit(f"solve for {platform} failed:\n{out.get('message') or res.stderr}")
    fetch = {p["name"]: p for p in out["actions"]["FETCH"]}
    order = [p["name"] for p in out["actions"]["LINK"]]          # dependency order
    missing = [n for n in order if n not in fetch]
    if missing:
        raise SystemExit(f"{platform}: no URL for {missing} (package cache not empty?)")
    return [f"{fetch[n]['url']}#{fetch[n]['md5']}" for n in order]


def export(env_name: str) -> list[str]:
    res = subprocess.run([conda_exe(), "list", "-n", env_name, "--explicit", "--md5"],
                         capture_output=True, text=True, check=True)
    return [line for line in res.stdout.splitlines() if line and not line.startswith(("#", "@"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", choices=PACKAGES)
    ap.add_argument("--platform", action="append", choices=PLATFORMS, required=True)
    ap.add_argument("--from-env", metavar="NAME", help="export linux-64 from this environment instead of solving")
    args = ap.parse_args(argv)
    yml = ROOT / args.package / "environment.yaml"
    channels, pins = read_env(yml)
    outdir = ROOT / args.package / "locks"
    outdir.mkdir(exist_ok=True)
    for platform in args.platform:
        if args.from_env:
            if platform != "linux-64":
                raise SystemExit("--from-env exports the host environment: linux-64 only")
            urls, how = export(args.from_env), f"exported from the environment `{args.from_env}`"
        else:
            urls, how = solve(channels, pins, platform), "solved from environment.yaml (" + ", ".join(
                f"{k}={v}" for k, v in OVERRIDES[platform].items()) + ")"
        bad = check_pins(pins, urls)
        if bad:
            raise SystemExit(f"{platform}: lock does not match {yml}:\n  " + "\n  ".join(bad))
        text = "\n".join([
            f"# {args.package} {platform} lock, {how}, {dt.date.today().isoformat()}",
            f"# made by release/make_locks.py; install with: conda create -n {args.package} --file <this file>",
            f"# platform: {platform}",
            "@EXPLICIT",
            *urls, ""])
        path = outdir / f"{platform}.txt"
        path.write_text(text)
        print(f"{path.relative_to(ROOT)}: {len(urls)} packages", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
