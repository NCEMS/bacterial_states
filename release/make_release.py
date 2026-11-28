"""Build the data release archives (GitHub release `resources-<version>` of NCEMS/bacterial_states).

    python release/make_release.py --root /path/to/checkout-with-data --out work/release_v2 [--version v2]

The archives are reproducible: members sorted by name, owner 0/0, modes 0644 / 0755, mtime fixed, gzip header without
a time stamp, so the same files always give the same bytes and SHA-256. Writes the archives and `SHA256SUMS` into
--out and prints the checksums; the URLs and pinned checksums live in ecoli_ingest/src/ecoli_ingest/resources.py
(+ resources.manifest.tsv, per file) and ecoli_atlas/src/ecoli_atlas/download.py (per archive).

Bundles (paths relative to --root; the name in the archive is what the download unpacks into its target):
  ecoli_ingest_core_<v>.tar.gz       resources/ -> vg/*, annotation/ (counting annotation, reference genome FASTA and
                                     its NCBI GenBank flat file, for breseq)
                                     unpacked into the inGEST resources directory
  ecoli_atlas_training_data_<v>.tar.gz   atlas_data/ -> the training tables, the metadata with its GEO text and
                                     evidence, and the genes called missing per sample; unpacked into atlas_data/
  ecoli_atlas_models_<v>.tar.gz      outputs/models/ -> models/ (the selected bundle and all seeds); unpacked into
                                     outputs/
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

MTIME = 1_700_000_000          # fixed member time (2023-11-14), so archives do not change with the file dates

CORE_FILES = (
    "vg/ecoli_graph_test.d2.dist",
    "vg/ecoli_graph_test.d2.gbz",
    "vg/ecoli_graph_test.d2.k15.min",
    "vg/ecoli_graph_test.d2.min",
    "vg/ecoli_graph_test.d2.shortread.zipcodes",
    "annotation/GCF_000005845.2_ASM584v2_counting_v2.bed",
    "annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv",
    "annotation/GCF_000005845.2_ASM584v2_counting_v2.gff",
    "annotation/GCF_000005845.2_ASM584v2_counting_v2.rfam_merge_report.tsv",
    "annotation/GCF_000005845.2_ASM584v2_genomic.fna",
    "annotation/GCF_000005845.2_ASM584v2_genomic.gbff",
)
TRAINING_FILES = (                 # (source under --root, name in the archive)
    ("atlas_data/atlas_v2_counts.tsv", "atlas_v2_counts.tsv"),
    ("atlas_data/atlas_v2_antisense_counts.tsv", "atlas_v2_antisense_counts.tsv"),
    ("atlas_data/metadata_v2.tsv", "metadata_v2.tsv"),
    ("atlas_data/sample_qc.tsv", "sample_qc.tsv"),
    ("atlas_data/time_courses.tsv", "time_courses.tsv"),
    ("atlas_data/metadata_v2_studies.tsv", "metadata_v2_studies.tsv"),
    ("atlas_data/metadata_v2_sample_text.tsv", "metadata_v2_sample_text.tsv"),
    ("atlas_data/metadata_v2_evidence.tsv", "metadata_v2_evidence.tsv"),
    ("atlas_data/metadata_v2_label_scheme.yaml", "metadata_v2_label_scheme.yaml"),
    ("outputs/models/missing_genes.tsv", "atlas_v2_missing_genes.tsv"),
)


def _members(root: Path, version: str) -> dict:
    """archive name -> list of (source path, name in the archive)."""
    models = root / "outputs" / "models"
    model_files = sorted(p for p in models.rglob("*") if p.is_file())
    return {
        f"ecoli_ingest_core_{version}.tar.gz": [(root / "resources" / f, f) for f in CORE_FILES],
        f"ecoli_atlas_training_data_{version}.tar.gz": [(root / src, name) for src, name in TRAINING_FILES],
        f"ecoli_atlas_models_{version}.tar.gz": [(p, "models/" + p.relative_to(models).as_posix()) for p in model_files],
    }


def _info(name: str, size: int | None = None) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = MTIME
    if size is None:
        info.type, info.mode = tarfile.DIRTYPE, 0o755
    else:
        info.size, info.mode = size, 0o644
    return info


def build(archive: Path, members: list) -> str:
    """Write one reproducible .tar.gz; returns its SHA-256."""
    missing = [str(src) for src, _ in members if not src.is_file()]
    if missing:
        raise FileNotFoundError(f"{archive.name}: missing {missing}")
    dirs = sorted({str(Path(name).parent.as_posix()) for _, name in members} - {"."})
    with open(archive, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for d in dirs:
            for i in range(1, len(Path(d).parts) + 1):                # parents first, each once
                part = "/".join(Path(d).parts[:i])
                if part not in tar.getnames():
                    tar.addfile(_info(part))
        for src, name in sorted(members, key=lambda m: m[1]):
            with open(src, "rb") as fh:
                tar.addfile(_info(name, src.stat().st_size), fh)
    h = hashlib.sha256()
    with open(archive, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="checkout with resources/, atlas_data/ and outputs/models/ (default: here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v2")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for name, members in _members(root, a.version).items():
        digest = build(out / name, members)
        size = (out / name).stat().st_size
        unpacked = sum(src.stat().st_size for src, _ in members)
        sums.append(f"{digest}  {name}")
        print(f"{name}: {len(members)} files, {size / 1e6:.0f} MB ({unpacked / 1e6:.0f} MB unpacked), sha256 {digest}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {out / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
