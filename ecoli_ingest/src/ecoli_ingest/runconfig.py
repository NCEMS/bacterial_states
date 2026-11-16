"""Reading the per-run config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Sample:
    name: str
    r1: list[str]                 # local files for mate 1 (empty for accession samples)
    r2: list[str] | None          # local files for mate 2; None = single-end (or unknown for accessions)
    accessions: list[str]         # SRR/SRX/SRS/GSM ids fetched from ENA
    condition: str | None


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return cfg


def samples(cfg: dict) -> list[Sample]:
    """Validate the ``samples:`` block. Each sample gives ``r1`` (and ``r2`` for paired-end), each a
    file or a list of files that are concatenated in order, *or* an ``accession`` (string or list).
    Accession layout (single/paired) is only known after resolution (see workflow/scripts/sra_reads.py)."""
    out = []
    for name, info in (cfg.get("samples") or {}).items():
        if not isinstance(info, dict):
            raise ValueError(f"sample {name!r}: must be a mapping with r1/r2 or accession")
        r1, r2, acc = _as_list(info.get("r1")), _as_list(info.get("r2")), _as_list(info.get("accession"))
        if acc and (r1 or r2):
            raise ValueError(f"sample {name!r}: give either r1/r2 or accession, not both")
        if not acc and not r1:
            raise ValueError(f"sample {name!r}: needs an 'r1' entry (file or list of files) or an 'accession'")
        if r2 and len(r2) != len(r1):
            raise ValueError(f"sample {name!r}: r1 has {len(r1)} file(s) but r2 has {len(r2)}")
        out.append(Sample(str(name), r1, r2 or None, acc, info.get("condition")))
    return out


def uses_accessions(cfg: dict) -> bool:
    return any(s.accessions for s in samples(cfg))


def sra_dir(cfg: dict, rundir: Path) -> Path:
    """Where accession downloads and the resolution cache live (config ``sra_dir``, default <run dir>/sra)."""
    return resolve_path(cfg.get("sra_dir") or "sra", rundir)


def resolve_path(value: str | Path, base: Path) -> Path:
    """Resolve a config path relative to the run directory, like Snakemake does."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def is_paired(cfg: dict, layouts: dict[str, bool] | None = None) -> tuple[bool, bool]:
    """Return (all_paired, mixed). The workflow treats any run with a missing r2 as single-end.
    ``layouts`` maps accession samples to True (paired) / False once they have been resolved; accession
    samples without an entry are left out (their layout is unknown until resolved)."""
    known = []
    for s in samples(cfg):
        if s.accessions:
            if layouts and s.name in layouts:
                known.append(layouts[s.name])
        else:
            known.append(s.r2 is not None)
    n_pe = sum(known)
    return (n_pe == len(known) and n_pe > 0), (0 < n_pe < len(known))


def centrifuge_enabled(cfg: dict) -> bool:
    """Centrifuge runs unless the config says ``centrifuge: false`` (mirrors the Snakefile)."""
    return bool(cfg.get("centrifuge", True))


def centrifuge_gate_enabled(cfg: dict) -> bool:
    """The species gate: on whenever Centrifuge runs, unless ``species_gate: false``."""
    return centrifuge_enabled(cfg) and bool(cfg.get("species_gate", True))
