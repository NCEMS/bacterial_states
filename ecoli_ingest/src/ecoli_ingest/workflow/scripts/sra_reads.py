#!/usr/bin/env python3
"""Where a sample's reads come from: local FASTQ files (one or several per mate) or SRA/GEO
accessions whose reads are fetched with sra-tools (``prefetch`` + ``fasterq-dump``, from NCBI) or,
as the backup, as ready-made FASTQ files from the ENA mirror.

Standard library only. The Snakefile imports this module at parse time (``sys.path`` points at
the scripts directory) and the ``ecoli_ingest`` launcher loads it for ``check``; the package is
not installed inside the container, so this file must not import ``ecoli_ingest``.

Accession resolution (network, cached as JSON under ``<sra_dir>/manifest/<ACC>.json``):

* SRR/ERR/DRR ids are runs already.
* SRX/SRS/SAMN/GSM ids go through the NCBI runinfo endpoint, which lists their runs.
* Each run's FASTQ URLs, md5s and sizes come from the ENA file report. ENA does not know GEO ids,
  hence the NCBI hop for GSM accessions.

Study-level ids (GSE, SRP, PRJ*) are rejected: one config sample cannot be a whole study.

Command line (``fetch`` is used by the ``sra_download`` rules)::

    sra_reads.py fetch --run SRR --out-dir DIR --mates 1,2 --method auto|sra-tools|ena --ena JSON
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
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "ecoli_ingest (https://github.com/NCEMS/bacterial_states)"
NCBI_RUNINFO = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={acc}"
ENA_FILEREPORT = ("https://www.ebi.ac.uk/ena/portal/api/filereport?accession={acc}&result=read_run"
                  "&fields=run_accession,library_layout,fastq_ftp,fastq_md5,fastq_bytes,read_count&format=tsv")
METHODS = ("auto", "sra-tools", "ena")
SRA_TOOLS = ("prefetch", "fasterq-dump")

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
    return ena_files_from_row(rows[0], run)


def ena_files_from_row(row: dict, run: str) -> dict:
    """Layout and FASTQ files of one run from the fields of an ENA read_run record
    (``fastq_ftp``, ``fastq_md5``, ``fastq_bytes``, ``library_layout``, ``read_count``)."""
    urls = [u for u in (row.get("fastq_ftp") or "").split(";") if u]
    spots = (row.get("read_count") or "").strip()
    spots = int(spots) if spots.isdigit() else None
    if not urls:
        # Only the SRA archive exists: sra-tools can still fetch it. No ENA backup for this run.
        layout = "PAIRED" if (row.get("library_layout") or "").upper() == "PAIRED" else "SINGLE"
        empty = {"url": None, "md5": None, "bytes": None}
        files = {"1": dict(empty), "2": dict(empty)} if layout == "PAIRED" else {"1": dict(empty)}
        return {"run": run, "layout": layout, "ena_layout": row.get("library_layout"), "files": files, "spots": spots}
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
            "ena_layout": row.get("library_layout"), "files": files, "spots": spots}


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
            spec["spots"] = {e["run"]: e.get("spots") for e in entries}
        out[str(name)] = spec
    return out


def download_summary(reads: dict, sra_dir: str) -> dict:
    """Totals for `ecoli_ingest check`: runs, files, bytes still to fetch (ENA FASTQ sizes), and
    ``no_ena``: runs that exist only as an SRA archive (sra-tools only, no backup)."""
    n_files = n_present = pending_bytes = 0
    no_ena = sorted({run for spec in reads.values() for run, files in spec["runs"].items()
                     if any(not f.get("url") for f in files.values())})
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
    return {"runs": len(runs), "files": n_files, "present": n_present, "pending_bytes": pending_bytes, "no_ena": no_ena}


# --------------------------------------------------------------------------- download

RETRY_HTTP = (403, 408, 429)      # transient on the ENA file servers (throttling), worth another attempt

def download(url: str, out: str, md5: str | None = None, size: int | None = None, log=None,
             attempts: int = 5, timeout: float = 120, backoff: float = 5.0) -> None:
    """Stream ``url`` to ``out`` via ``out.part``, verifying md5/size before the final rename.

    Failed attempts are repeated after ``backoff * 2**i`` seconds. That includes HTTP 403, 408 and
    429: the ENA file servers answer 403 for a while when one address opens many connections, and
    the same URL works again shortly after. Other 4xx answers (e.g. 404) end the attempts at once."""
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
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code not in RETRY_HTTP:
                break
            if i + 1 < attempts:
                time.sleep(backoff * 2 ** i)
    raise DownloadError(f"download of {url} failed: {last}")


# --------------------------------------------------------------------------- fetch a run

def sra_tools_available() -> bool:
    return all(shutil.which(tool) for tool in SRA_TOOLS)


def check_method(method: str | None) -> str:
    method = (method or "auto").strip().lower().replace("_", "-")
    if method in ("sratools", "sra"):
        method = "sra-tools"
    if method not in METHODS:
        raise AccessionError(f"sra_method must be one of {', '.join(METHODS)} (got {method!r})")
    return method


def _run_tool(cmd: list, log=None) -> None:
    _log(log, "  $ " + " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(res.stdout.strip().splitlines()[-6:])
    if tail:
        _log(log, "    " + tail.replace("\n", "\n    "))
    if res.returncode != 0:
        raise DownloadError(f"{cmd[0]} exited with status {res.returncode}")


def _compress(src: str, dst: str, threads: int = 1) -> int:
    """gzip ``src`` to ``dst`` (pigz when available) and return the number of FASTQ records."""
    lines = 0
    with open(src, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            lines += chunk.count(b"\n")
    if lines % 4:
        raise DownloadError(f"{os.path.basename(src)}: {lines} lines is not a whole number of FASTQ records")
    part = dst + ".part"
    if shutil.which("pigz"):
        with open(part, "wb") as out:
            res = subprocess.run(["pigz", "-c", "-p", str(max(1, threads)), src], stdout=out, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise DownloadError(f"pigz failed on {src}: {res.stderr.decode(errors='replace')[-300:]}")
    else:
        import gzip
        with open(src, "rb") as fh, gzip.open(part, "wb", compresslevel=6) as out:
            shutil.copyfileobj(fh, out, 1 << 24)
    os.replace(part, dst)
    return lines // 4


def fetch_sra_tools(run: str, out_dir: str, mates=("1", "2"), threads: int = 4, spots: int | None = None,
                    log=None) -> dict:
    """``prefetch`` the run from the SRA and convert it with ``fasterq-dump --split-3``; write
    ``<out_dir>/<run>_<mate>.fastq.gz`` for the wanted mates and return ``{mate: n_reads}``.

    Read names follow the ENA FASTQ files (``@RUN.N NAME/MATE``, bare ``+`` line), so both sources
    give the same records. Technical reads are skipped and reads without a partner (a third file
    of ``--split-3``) are left out, as in the ENA files. Checks: both mates hold the same number of
    reads, and that number equals ``spots`` (the ENA read count) when it is known."""
    mates = tuple(str(m) for m in mates)
    missing = [tool for tool in SRA_TOOLS if not shutil.which(tool)]
    if missing:
        raise DownloadError(f"sra-tools not on PATH (missing: {', '.join(missing)})")
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f".fetch_{run}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        _log(log, f"fetching {run} with sra-tools -> {out_dir}")
        _run_tool(["prefetch", run, "--output-directory", tmp, "--max-size", "u"], log)
        archive = next((os.path.join(tmp, run, f) for f in (f"{run}.sra", f"{run}.sralite")
                        if os.path.exists(os.path.join(tmp, run, f))), None)
        if archive is None:
            raise DownloadError(f"prefetch left no {run}.sra under {tmp}")
        _run_tool(["fasterq-dump", archive, "--split-3", "--skip-technical", "--threads", str(max(1, threads)),
                   "--temp", tmp, "--outdir", tmp, "--seq-defline", "@$ac.$si $sn/$ri", "--qual-defline", "+"], log)
        produced = {m: os.path.join(tmp, f"{run}_{m}.fastq") for m in ("1", "2")}
        single = os.path.join(tmp, f"{run}.fastq")
        if all(os.path.exists(p) for p in produced.values()):
            sources = produced
            if os.path.exists(single) and os.path.getsize(single):
                _log(log, f"  note: reads without a partner in {run}.fastq are left out")
        elif os.path.exists(single):
            sources = {"1": single}
        else:
            raise DownloadError(f"fasterq-dump wrote no FASTQ for {run}")
        absent = [m for m in mates if m not in sources]
        if absent:
            raise DownloadError(f"{run}: the SRA archive has no mate {', '.join(absent)} (single-end archive?)")
        counts = {m: _compress(sources[m], run_file_in(out_dir, run, m) + ".tmp", threads) for m in mates}
        if len(set(counts.values())) > 1:
            raise DownloadError(f"{run}: mates differ in read number ({counts})")
        if spots and counts[mates[0]] != spots and set(sources) == {"1", "2"} == set(mates):
            raise DownloadError(f"{run}: {counts[mates[0]]} reads per mate, ENA lists {spots}")
        for m in mates:
            os.replace(run_file_in(out_dir, run, m) + ".tmp", run_file_in(out_dir, run, m))
        _log(log, f"  ok: {run} " + ", ".join(f"mate {m}: {n} reads" for m, n in counts.items()))
        return counts
    except Exception:
        for m in mates:
            for suffix in (".tmp", ".tmp.part"):
                if os.path.exists(run_file_in(out_dir, run, m) + suffix):
                    os.remove(run_file_in(out_dir, run, m) + suffix)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_file_in(out_dir: str, run: str, mate: str) -> str:
    return os.path.join(out_dir, f"{run}_{mate}.fastq.gz")


def fetch_run(run: str, out_dir: str, mates=("1", "2"), method: str = "auto", ena_files: dict | None = None,
              threads: int = 4, spots: int | None = None, log=None) -> str:
    """Get the wanted mate files of one run into ``out_dir`` and return the source that worked.

    ``auto``: sra-tools first; if the tools are not installed or the fetch fails, the ENA FASTQ
    files (``ena_files = {mate: {url, md5, bytes}}``). ``sra-tools`` / ``ena``: that source only."""
    method = check_method(method)
    mates = tuple(str(m) for m in mates)
    errors = []
    if method in ("auto", "sra-tools"):
        if method == "auto" and not sra_tools_available():
            _log(log, f"{run}: sra-tools not on PATH, using ENA")
            errors.append("sra-tools not installed")
        else:
            try:
                fetch_sra_tools(run, out_dir, mates, threads=threads, spots=spots, log=log)
                return "sra-tools"
            except (DownloadError, OSError) as e:
                errors.append(f"sra-tools: {e}")
                _log(log, f"  sra-tools failed for {run}: {e}")
                if method == "sra-tools":
                    raise DownloadError(f"{run}: {e}")
                _log(log, f"  falling back to ENA for {run}")
    no_ena = [m for m in mates if not (ena_files or {}).get(m, {}).get("url")]
    if no_ena:
        raise DownloadError(f"{run}: ENA has no FASTQ file for mate(s) {', '.join(no_ena)}"
                            + (f" ({'; '.join(errors)})" if errors else ""))
    try:
        for m in mates:
            f = ena_files[m]
            download(f["url"], run_file_in(out_dir, run, m), md5=f.get("md5"), size=f.get("bytes"), log=log)
    except DownloadError as e:
        raise DownloadError(f"{run}: {'; '.join(errors + ['ENA: ' + str(e)])}")
    return "ena"


# --------------------------------------------------------------------------- read subsample

def stream_head(url: str, out: str, max_reads: int, log=None, attempts: int = 4, timeout: float = 120,
                backoff: float = 5.0) -> int:
    """Write the first ``max_reads`` records of a remote ``.fastq.gz`` to ``out`` (uncompressed)
    and return the number written. The connection is closed as soon as enough has arrived, so
    only the start of the file is transferred."""
    import zlib
    want = 4 * max_reads
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            _log(log, f"reading first {max_reads} reads of {url}" + (f" (attempt {i + 1})" if i else ""))
            lines = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(out + ".part", "wb") as fh:
                z = zlib.decompressobj(16 + zlib.MAX_WBITS)
                buf = b""
                while lines < want:
                    chunk = resp.read(1 << 18)
                    if not chunk:
                        break
                    while chunk and lines < want:
                        data = z.decompress(chunk)
                        chunk = b""
                        if z.eof:                      # a file made of several gzip members
                            chunk = z.unused_data
                            z = zlib.decompressobj(16 + zlib.MAX_WBITS)
                        buf += data
                        cut = buf.rfind(b"\n") + 1
                        block, buf = buf[:cut], buf[cut:]
                        n = block.count(b"\n")
                        if lines + n > want:
                            keep = want - lines
                            pos = -1
                            for _ in range(keep):
                                pos = block.index(b"\n", pos + 1)
                            block, n = block[:pos + 1], keep
                        fh.write(block)
                        lines += n
            lines -= lines % 4 if lines < want else 0
            if lines % 4:
                raise DownloadError(f"{url}: stream ended inside a FASTQ record")
            if lines == 0:
                raise DownloadError(f"{url}: no reads received")
            os.replace(out + ".part", out)
            return lines // 4
        except (DownloadError, urllib.error.URLError, TimeoutError, OSError, zlib.error) as e:
            last = e
            if os.path.exists(out + ".part"):
                os.remove(out + ".part")
            _log(log, f"  failed: {e}")
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code not in RETRY_HTTP:
                break
            if i + 1 < attempts:
                time.sleep(backoff * 2 ** i)
    raise DownloadError(f"reading {url} failed: {last}")


def _truncate_fastq(path: str, n_reads: int) -> None:
    """Cut a FASTQ file down to its first ``n_reads`` records."""
    with open(path, "rb") as fh, open(path + ".cut", "wb") as out:
        for i, line in enumerate(fh):
            if i >= 4 * n_reads:
                break
            out.write(line)
    os.replace(path + ".cut", path)


def fetch_subsample(run: str, out_dir: str, mates=("1", "2"), max_spots: int = 200_000, skip_spots: int = 0,
                    method: str = "auto", ena_files: dict | None = None, log=None) -> tuple[str, dict]:
    """Write up to ``max_spots`` reads of a run to ``<out_dir>/<run>_<mate>.fastq`` (uncompressed)
    without fetching the whole run. Returns ``(source, {mate: n_reads})``.

    ``sra-tools``: ``fastq-dump -N/-X`` streams spots ``skip_spots+1 .. skip_spots+max_spots``
    from the SRA. ``ena``: the first ``max_spots`` records of each ENA FASTQ file (``skip_spots``
    does not apply). ``auto`` tries sra-tools when ``fastq-dump`` is on PATH, then ENA. Both mates
    are cut to the same number of reads."""
    method = check_method(method)
    mates = tuple(str(m) for m in mates)
    os.makedirs(out_dir, exist_ok=True)
    out = {m: os.path.join(out_dir, f"{run}_{m}.fastq") for m in mates}
    errors = []
    if method in ("auto", "sra-tools"):
        if not shutil.which("fastq-dump"):
            errors.append("fastq-dump not on PATH")
            if method == "sra-tools":
                raise DownloadError(f"{run}: fastq-dump not on PATH")
        else:
            tmp = os.path.join(out_dir, f".sub_{run}")
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            try:
                _run_tool(["fastq-dump", "-N", str(skip_spots + 1), "-X", str(skip_spots + max_spots), "--split-3",
                           "--skip-technical", "--defline-seq", "@$ac.$si $sn/$ri", "--defline-qual", "+",
                           "-O", tmp, run], log)
                paired = {m: os.path.join(tmp, f"{run}_{m}.fastq") for m in ("1", "2")}
                single = os.path.join(tmp, f"{run}.fastq")
                sources = paired if all(os.path.exists(p) for p in paired.values()) else (
                    {"1": single} if os.path.exists(single) else {})
                absent = [m for m in mates if m not in sources]
                if absent:
                    raise DownloadError(f"{run}: fastq-dump wrote no file for mate {', '.join(absent)}")
                for m in mates:
                    os.replace(sources[m], out[m])
                counts = _equalize(out)
                if not min(counts.values()):
                    raise DownloadError(f"{run}: fastq-dump returned no reads")
                return "sra-tools", counts
            except (DownloadError, OSError) as e:
                errors.append(f"sra-tools: {e}")
                _log(log, f"  sra-tools failed for {run}: {e}")
                if method == "sra-tools":
                    raise DownloadError(f"{run}: {e}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    no_ena = [m for m in mates if not (ena_files or {}).get(m, {}).get("url")]
    if no_ena:
        raise DownloadError(f"{run}: ENA has no FASTQ file for mate(s) {', '.join(no_ena)}"
                            + (f" ({'; '.join(errors)})" if errors else ""))
    try:
        for m in mates:
            stream_head(ena_files[m]["url"], out[m], max_spots, log=log)
    except DownloadError as e:
        raise DownloadError(f"{run}: {'; '.join(errors + ['ENA: ' + str(e)])}")
    return "ena", _equalize(out)


def _equalize(files: dict) -> dict:
    """Count the reads of each mate file and cut all of them to the smallest count."""
    counts = {}
    for m, path in files.items():
        with open(path, "rb") as fh:
            counts[m] = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 22), b"")) // 4
    n = min(counts.values())
    for m, path in files.items():
        if counts[m] != n:
            _truncate_fastq(path, n)
    return {m: n for m in files}


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch one file with md5/size verification")
    d.add_argument("--url", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--md5", default=None)
    d.add_argument("--bytes", type=int, default=None)
    f = sub.add_parser("fetch", help="get one run's FASTQ files: sra-tools first, ENA as the backup")
    f.add_argument("--run", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--mates", default="1,2", help="1 or 1,2")
    f.add_argument("--method", default="auto", help="|".join(METHODS))
    f.add_argument("--ena", default=None, help='JSON {"1": {"url":..., "md5":..., "bytes":...}, "2": {...}}')
    f.add_argument("--spots", default=None, help="expected reads per mate (ENA read_count), if known")
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--max-spots", type=int, default=None,
                   help="fetch only this many reads (uncompressed <run>_<mate>.fastq), without the whole run")
    f.add_argument("--skip-spots", type=int, default=0, help="with --max-spots and sra-tools: spots to skip first")
    r = sub.add_parser("resolve", help="print the runs and files for accessions (JSON)")
    r.add_argument("accessions", nargs="+")
    r.add_argument("--cache-dir", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "download":
            md5 = None if args.md5 in (None, "", "None") else args.md5
            download(args.url, args.out, md5=md5, size=args.bytes, log=sys.stderr)
        elif args.cmd == "fetch":
            spots = int(args.spots) if args.spots and str(args.spots).isdigit() and int(args.spots) > 0 else None
            if args.max_spots:
                source, counts = fetch_subsample(args.run, args.out_dir, tuple(m for m in args.mates.split(",") if m),
                                                 args.max_spots, args.skip_spots, args.method,
                                                 json.loads(args.ena) if args.ena else None, log=sys.stderr)
                print(f"{args.run}: {counts} reads from {source}", file=sys.stderr)
                return 0
            source = fetch_run(args.run, args.out_dir, tuple(m for m in args.mates.split(",") if m), args.method,
                               json.loads(args.ena) if args.ena else None, args.threads, spots, log=sys.stderr)
            print(f"{args.run}: fetched from {source}", file=sys.stderr)
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
