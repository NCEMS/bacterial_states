"""Check the tool versions the workflow records against the pins of environment.yaml.

    python sample_qc.py versions --out versions.json
    python check_versions.py --env environment.yaml --versions versions.json

The container's %test runs this, so an image whose tools differ from the pins fails its build. A recorded
version matches a pin when it is the pin or a prefix of it at a dot (centrifuge reports 1.0.4 for the package
1.0.4.2). Exit status 1 lists the mismatches.
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

# tool name in sample_qc.VERSION_COMMANDS -> conda package
TOOL_PACKAGES = {
    "fastp": "fastp", "fastqc": "fastqc", "centrifuge": "centrifuge", "vg": "vg", "samtools": "samtools",
    "bedtools": "bedtools", "featureCounts": "subread", "rseqc": "rseqc", "multiqc": "multiqc",
    "snakemake": "snakemake-minimal", "R": "r-base",
}


def read_pins(path: str) -> dict[str, str]:
    deps = yaml.safe_load(open(path, encoding="utf-8"))["dependencies"]
    pins = {}
    for d in deps:
        if isinstance(d, str):
            name, _, version = d.split("#")[0].strip().partition("=")
            pins[name] = version
    return pins


def matches(recorded: str | None, pinned: str | None) -> bool:
    if not recorded or not pinned:
        return False
    return recorded == pinned or pinned.startswith(recorded + ".")


def mismatches(versions: dict[str, str | None], pins: dict[str, str]) -> list[str]:
    out = []
    for tool, package in TOOL_PACKAGES.items():
        rec, pin = versions.get(tool), pins.get(package)
        if not matches(rec, pin):
            out.append(f"{tool}: recorded {rec}, {package} pinned {pin}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--versions", required=True, help="output of `sample_qc.py versions` (or a tool_versions.json)")
    args = ap.parse_args(argv)
    data = json.load(open(args.versions, encoding="utf-8"))
    bad = mismatches(data.get("versions", data), read_pins(args.env))
    for line in bad:
        print(line, file=sys.stderr)
    print(f"{len(TOOL_PACKAGES) - len(bad)}/{len(TOOL_PACKAGES)} tool versions match the pins", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
