#!/usr/bin/env python3
"""Where a sample's reads come from: local FASTQ files (one or several per mate) or SRA/GEO
accessions whose FASTQ files are fetched from the ENA mirror.

Standard library only. The Snakefile imports this module at parse time (``sys.path`` points at
the scripts directory) and the ``ecoli_ingest`` launcher loads it for ``check``; the package is
not installed inside the container, so this file must not import ``ecoli_ingest``.

Accession resolution (network, cached as JSON under ``<sra_dir>/manifest/<ACC>.json``):

* SRR/ERR/DRR ids are runs already.
* SRX/SRS/SAMN/GSM ids go through the NCBI runinfo endpoint, which lists their runs.
* Each run's FASTQ URLs, md5s and sizes come from the ENA file report. ENA does not know GEO ids,
  hence the NCBI hop for GSM accessions.

Study-level ids (GSE, SRP, PRJ*) are rejected: one config sample cannot be a whole study.

Command line (used by the ``sra_download`` rule)::

    sra_reads.py download --url URL --md5 MD5 --bytes N --out FILE
    sra_reads.py resolve ACC [ACC ...] --cache-dir DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "ecoli_ingest (https://github.com/NCEMS/bacterial_states)"
NCBI_RUNINFO = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={acc}"
ENA_FILEREPORT = ("https://www.ebi.ac.uk/ena/portal/api/filereport?accession={acc}&result=read_run"
                  "&fields=run_accession,library_layout,fastq_ftp,fastq_md5,fastq_bytes&format=tsv")

RUN_RE = re.compile(r"^[SED]RR\d+$")
_KINDS = (
    ("run", RUN_RE),
    ("experiment", re.compile(r"^[SED]RX\d+$")),
    ("sample", re.compile(r"^[SED]RS\d+$|^SAMN\d+$|^SAMD\d+$|^SAMEA?\d+$")),
    ("geo_sample", re.compile(r"^GSM\d+$")),
)
_STUDY_RE = re.compile(r"^GSE\d+$|^[SED]RP\d+$|^PRJ[A-Z]{2}\d+$|^[SED]RA\d+$")
GZ_SUFFIXES = (".gz", ".gzip")


class AccessionError(ValueError):
    """Bad, unknown or unresolvable accession; or inconsistent sample read specification."""


class DownloadError(RuntimeError):
    pass


def _log(log, msg: str) -> None:
    if log is not None:
        print(msg, file=log, flush=True)


# --------------------------------------------------------------------------- accessions

def classify(acc: str) -> str:
    """'run' | 'experiment' | 'sample' | 'geo_sample'; AccessionError for anything else."""
    acc = acc.strip()
    for kind, rx in _KINDS:
        if rx.match(acc):
            return kind
    if _STUDY_RE.match(acc):
        raise AccessionError(
            f"{acc} is a study-level accession (whole series/project). One config sample cannot be a "
            f"whole study: list its GSM/SRX/SRR ids as separate samples instead.")
    raise AccessionError(f"{acc!r} is not a recognised accession (expected SRR/ERR/DRR, SRX, SRS, SAMN/SAMEA or GSM)")


def _fetch(url: str, timeout: float = 60, attempts: int = 3) -> str:
    """GET a small text document, retrying transient failures."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            last = e
            if 400 <= e.code < 500 and e.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(2 ** i)
    raise AccessionError(f"could not fetch {url}: {last}")


def _runs_from_ncbi(acc: str) -> list[str]:
    text = _fetch(NCBI_RUNINFO.format(acc=acc))
    rows = list(csv.DictReader(io.StringIO(text)))
    runs = [r["Run"] for r in rows if r.get("Run") and RUN_RE.match(r["Run"])]
    if not runs:
        raise AccessionError(f"NCBI lists no SRA runs for {acc}")
    return runs


def _parse_ena_report(text: str, run: str) -> dict:
    rows = [r for r in csv.DictReader(io.StringIO(text), delimiter="\t") if r.get("run_accession") == run]
    if not rows:
        raise AccessionError(f"ENA has no read_run record for {run}")
    row = rows[0]
    urls = [u for u in (row.get("fastq_ftp") or "").split(";") if u]
    if not urls:
        raise AccessionError(f"ENA has no FASTQ files for {run} (only the SRA archive). Download it yourself "
                             f"(e.g. fasterq-dump) and list the files under r1/r2.")
    md5s = (row.get("fastq_md5") or "").split(";")
    sizes = (row.get("fastq_bytes") or "").split(";")
    by_name = {}
    for i, u in enumerate(urls):
        name = u.rsplit("/", 1)[-1]
        by_name[name] = {
            "url": u if "://" in u else "https://" + u,
            "md5": md5s[i] if i < len(md5s) and md5s[i] else None,
            "bytes": int(sizes[i]) if i < len(sizes) and sizes[i].isdigit() else None,
        }
    files: dict[str, dict] = {}
    if f"{run}_1.fastq.gz" in by_name and f"{run}_2.fastq.gz" in by_name:
        files = {"1": by_name[f"{run}_1.fastq.gz"], "2": by_name[f"{run}_2.fastq.gz"]}
    elif f"{run}.fastq.gz" in by_name:
        files = {"1": by_name[f"{run}.fastq.gz"]}
    elif len(by_name) == 1:
        files = {"1": next(iter(by_name.values()))}
    else:
        raise AccessionError(f"cannot pick read files for {run} from ENA listing: {sorted(by_name)}")
    return {"run": run, "layout": "PAIRED" if "2" in files else "SINGLE",
            "ena_layout": row.get("library_layout"), "files": files}


def _run_files(run: str) -> dict:
    return _parse_ena_report(_fetch(ENA_FILEREPORT.format(acc=run)), run)


def resolve_accession(acc: str, cache_dir: str, log=None) -> dict:
    """Runs and FASTQ files for one accession; cached in ``<cache_dir>/<ACC>.json``."""
    acc = acc.strip()
    kind = classify(acc)
    cache = os.path.join(cache_dir, acc + ".json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)
    _log(log, f"resolving {acc} ({kind}) via NCBI/ENA ...")
    runs = [acc] if kind == "run" else _runs_from_ncbi(acc)
    entries = [_run_files(r) for r in runs]
    layouts = {e["layout"] for e in entries}
    if len(layouts) > 1:
        raise AccessionError(f"{acc}: its runs mix single- and paired-end layouts "
                             f"({', '.join(e['run'] + '=' + e['layout'] for e in entries)}); list the runs you want")
    rec = {"accession": acc, "kind": kind, "layout": entries[0]["layout"], "runs": entries,
           "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, cache)
    _log(log, f"  {acc}: {len(entries)} run(s), {rec['layout'].lower()}-end: " + ", ".join(e["run"] for e in entries))
    return rec


# --------------------------------------------------------------------------- samples

def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def is_gz(path: str) -> bool:
    return path.lower().endswith(GZ_SUFFIXES)


def run_file(sra_dir: str, run: str, mate: str) -> str:
    """Local path of a downloaded mate file (single-end runs are stored as ``_1``)."""
    return os.path.join(sra_dir, run, f"{run}_{mate}.fastq.gz")


def normalize_sample(name: str, info) -> dict:
    """Validate one ``samples:`` entry and return {'r1': [...], 'r2': [...]|None, 'accessions': [...]}."""
    if not isinstance(info, dict):
        raise AccessionError(f"sample {name!r}: must be a mapping with r1/r2 or accession")
    r1, r2, acc = as_list(info.get("r1")), as_list(info.get("r2")), as_list(info.get("accession"))
    if acc and (r1 or r2):
        raise AccessionError(f"sample {name!r}: give either r1/r2 or accession, not both")
    if not acc and not r1:
        raise AccessionError(f"sample {name!r}: needs an 'r1' entry (file or list of files) or an 'accession'")
    if r2 and len(r2) != len(r1):
        raise AccessionError(f"sample {name!r}: r1 has {len(r1)} file(s) but r2 has {len(r2)}")
    for mate, files in (("r1", r1), ("r2", r2)):
        if len({is_gz(f) for f in files}) > 1:
            raise AccessionError(f"sample {name!r}: {mate} mixes gzipped and plain FASTQ files; "
                                 f"files of one mate are concatenated as-is and must share compression")
    for a in acc:
        classify(a)
    return {"r1": r1, "r2": r2 or None, "accessions": acc}


def resolve_samples(samples_cfg: dict, sra_dir: str, cache_dir: str, log=None) -> dict:
    """For every sample: the files fastp reads (``r1``/``r2`` lists, ``r2`` None for single-end),
    its accessions and, for downloaded runs, ``runs = {run: {mate: {url, md5, bytes}}}``."""
    out: dict[str, dict] = {}
    for name, info in (samples_cfg or {}).items():
        spec = normalize_sample(str(name), info)
        spec["runs"] = {}
        if spec["accessions"]:
            entries = []
            for a in spec["accessions"]:
                entries.extend(resolve_accession(a, cache_dir, log)["runs"])
            layouts = {e["layout"] for e in entries}
            if len(layouts) > 1:
                raise AccessionError(f"sample {name!r}: accessions mix single- and paired-end runs")
            paired = entries[0]["layout"] == "PAIRED"
            spec["r1"] = [run_file(sra_dir, e["run"], "1") for e in entries]
            spec["r2"] = [run_file(sra_dir, e["run"], "2") for e in entries] if paired else None
            spec["runs"] = {e["run"]: e["files"] for e in entries}
        out[str(name)] = spec
    return out


def download_summary(reads: dict, sra_dir: str) -> dict:
    """Totals for `ecoli_ingest check`: runs, files, bytes still to fetch."""
    n_files = n_present = pending_bytes = 0
    runs = set()
    for spec in reads.values():
        for run, files in spec["runs"].items():
            runs.add(run)
            for mate, f in files.items():
                n_files += 1
                if os.path.exists(run_file(sra_dir, run, mate)):
                    n_present += 1
                else:
                    pending_bytes += f.get("bytes") or 0
    return {"runs": len(runs), "files": n_files, "present": n_present, "pending_bytes": pending_bytes}


# --------------------------------------------------------------------------- download

def download(url: str, out: str, md5: str | None = None, size: int | None = None, log=None,
             attempts: int = 3, timeout: float = 120) -> None:
    """Stream ``url`` to ``out`` via ``out.part``, verifying md5/size before the final rename."""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    part = out + ".part"
    last: Exception | None = None
    for i in range(attempts):
        h = hashlib.md5()
        n = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            _log(log, f"downloading {url} -> {out}" + (f" (attempt {i + 1})" if i else ""))
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            if size is not None and n != size:
                raise DownloadError(f"{out}: got {n} bytes, expected {size}")
            if md5 and h.hexdigest() != md5.lower():
                raise DownloadError(f"{out}: md5 {h.hexdigest()} does not match expected {md5}")
            os.replace(part, out)
            _log(log, f"  ok: {n} bytes" + (", md5 verified" if md5 else ""))
            return
        except (DownloadError, urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if os.path.exists(part):
                os.remove(part)
            _log(log, f"  failed: {e}")
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                break
            if i + 1 < attempts:
                time.sleep(5 * 2 ** i)
    raise DownloadError(f"download of {url} failed: {last}")


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch one file with md5/size verification")
    d.add_argument("--url", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--md5", default=None)
    d.add_argument("--bytes", type=int, default=None)
    r = sub.add_parser("resolve", help="print the runs and files for accessions (JSON)")
    r.add_argument("accessions", nargs="+")
    r.add_argument("--cache-dir", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "download":
            md5 = None if args.md5 in (None, "", "None") else args.md5
            download(args.url, args.out, md5=md5, size=args.bytes, log=sys.stderr)
        else:
            recs = [resolve_accession(a, args.cache_dir, log=sys.stderr) for a in args.accessions]
            json.dump(recs, sys.stdout, indent=1)
            print()
    except (AccessionError, DownloadError) as e:
        print(f"sra_reads: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
