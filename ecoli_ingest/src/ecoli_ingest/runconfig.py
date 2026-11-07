"""Reading the per-run config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Sample:
    name: str
    r1: str
    r2: str | None
    condition: str | None


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return cfg


def samples(cfg: dict) -> list[Sample]:
    out = []
    for name, info in (cfg.get("samples") or {}).items():
        if not isinstance(info, dict) or "r1" not in info:
            raise ValueError(f"sample {name!r}: needs at least an 'r1' entry")
        out.append(Sample(str(name), str(info["r1"]), info.get("r2"), info.get("condition")))
    return out


def resolve_path(value: str | Path, base: Path) -> Path:
    """Resolve a config path relative to the run directory, like Snakemake does."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def is_paired(cfg: dict) -> tuple[bool, bool]:
    """Return (all_paired, mixed). The workflow treats any run with a missing r2 as single-end."""
    ss = samples(cfg)
    n_pe = sum(s.r2 is not None for s in ss)
    return (n_pe == len(ss) and n_pe > 0), (0 < n_pe < len(ss))


def centrifuge_enabled(cfg: dict) -> bool:
    return cfg.get("centrifuge_min_percentage") is not None
