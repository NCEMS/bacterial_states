#!/usr/bin/env python
"""Prescreen atlas candidate samples on a small read subsample.

For each sample of the manifest (build_manifest.py `samples.tsv` + `runs.tsv`) this takes up to
--spots reads of its first run without downloading the run, runs the inGEST steps on them with
the workflow's own options, and writes `<out>/samples/<GSM>.json` with the QC metrics:

  fastp                         reads kept, read length after trimming, adapters, insert size
  vg giraffe + surject          alignment rate
  RSeQC infer_experiment        strandedness (call_strandedness.py)
  gene_type.sh                  share of reads on CDS / rRNA / tRNA / ncRNA features
  Centrifuge (--centrifuge-spots reads, one pass)         is it E. coli: share of genus-resolved reads in
                                                          Escherichia or Shigella and the largest other genus,
                                                          from all reads and again without reads on rRNA / tRNA
                                                          genes; also % unclassified, % Enterobacteriaceae,
                                                          % synthetic constructs, the workflow gate's % E. coli

The Centrifuge index (8.6 GB) is copied once to --index-cache (default /dev/shm, memory-backed)
and memory-mapped from there, so every sample opens it in about a second. Memory-mapping it
from a network file system instead takes minutes per sample; with `--index-cache none` the index
is read from the resources directory without memory-mapping (a few seconds per sample once the
file system has cached it).

Samples that already have a JSON are skipped (--redo, --redo-errors to repeat). Merge the JSON
files into the sample QC table with `qc_table.py merge-prescreen`.

    prescreen.py --manifest work/atlas_manifest --out work/prescreen --resources resources --jobs 16
    prescreen.py ... --shard 3/20          # one part of twenty, e.g. one SLURM array task
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qc_metrics  # noqa: E402  (also puts the workflow scripts on sys.path)
import sra_reads   # noqa: E402

# Relative paths inside the resources directory; mirror ecoli_ingest/paths.py RESOURCE_FILES.
RESOURCES = {
    "vg_index": "vg/ecoli_graph_test",
    "annotation_bed": "annotation/GCF_000005845.2_ASM584v2_counting_v2.bed",
    "centrifuge_index": "centrifuge/p_compressed+h+v",
}
GREF = "GCF_000005845_2_ASM584v2_genomic#0#NC_000913.3"
TOOLS = ("fastp", "vg", "samtools", "bedtools", "infer_experiment.py", "centrifuge", "centrifuge-kreport")


class StepError(RuntimeError):
    pass


STEP_TIMEOUT = 1500     # seconds; vg giraffe can take very long on reads that are not E. coli


def sh(cmd: str, step: str, log_path: str, timeout: float | None = None) -> None:
    import signal
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {cmd}\n")
        log.flush()
        proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout or STEP_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise StepError(f"{step}: timed out after {timeout or STEP_TIMEOUT:.0f} s")
        res = proc
    if res.returncode != 0:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            tail = " | ".join(fh.read().strip().splitlines()[-3:])
        raise StepError(f"{step}: exit {res.returncode}: {tail[-400:]}")


def wait_for_fetch_slot(lock_path: str, interval: float) -> None:
    """Space out the starts of read downloads across all worker processes: at most one new
    request to NCBI / ENA every `interval` seconds, however many samples run at once."""
    if interval <= 0:
        return
    import fcntl
    with open(lock_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        text = fh.read().strip()
        last = float(text) if text else 0.0
        wait = last + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        fh.seek(0)
        fh.truncate()
        fh.write(repr(time.time()))
        fh.flush()


def head_fastq(src: str, dst: str, n_reads: int, skip_ids: set | None = None) -> int:
    """Write the first `n_reads` records of `src` whose read id is not in `skip_ids`; return how many."""
    n = 0
    keep = True
    with open(src, "rb") as fh, open(dst, "wb") as out:
        for i, line in enumerate(fh):
            if i % 4 == 0:
                if n >= n_reads:
                    break
                keep = not skip_ids or line[1:].split(None, 1)[0].split(b"/")[0] not in skip_ids
                n += keep
            if keep:
                out.write(line)
    return n


def screen_sample(sample: dict, run_row: dict, opts: dict) -> dict:
    """Screen one sample; returns the record that is written to <GSM>.json."""
    gsm, run = sample["GSM"], run_row["run_accession"]
    rec = {"GSM": gsm, "GSE": sample["GSE"], "run": run, "spots_requested": opts["spots"],
           "date": datetime.date.today().isoformat(), "pipeline": opts["pipeline"], "seconds": {}}
    work = os.path.join(opts["scratch"], gsm)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    log = os.path.join(work, "log.txt")
    q = lambda p: "'" + p.replace("'", "'\\''") + "'"  # noqa: E731
    res, t = opts["resources"], opts["threads"]
    step = "start"

    def timed(name, fn):
        nonlocal step
        step = name
        t0 = time.time()
        out = fn()
        rec["seconds"][name] = round(time.time() - t0, 1)
        return out

    try:
        ena = sra_reads.ena_files_from_row(run_row, run)
        paired = ena["layout"] == "PAIRED"
        mates = ("1", "2") if paired else ("1",)
        wait_for_fetch_slot(os.path.join(opts["scratch"], ".fetch_slot"), opts["fetch_interval"])
        with open(log, "a", encoding="utf-8") as lf:
            source, counts = timed("fetch", lambda: sra_reads.fetch_subsample(
                run, work, mates, opts["spots"], opts["skip_spots"], opts["method"], ena["files"], log=lf))
        rec.update(source=source, spots=counts["1"], paired=paired)
        raw = {m: os.path.join(work, f"{run}_{m}.fastq") for m in mates}
        clean = {m: os.path.join(work, f"clean_{m}.fastq") for m in mates}
        fastp_json = os.path.join(work, "fastp.json")
        if paired:
            cmd = (f"fastp -i {q(raw['1'])} -I {q(raw['2'])} -o {q(clean['1'])} -O {q(clean['2'])} "
                   f"-h {q(work)}/fastp.html -j {q(fastp_json)} -w {t} --detect_adapter_for_pe")
        else:
            cmd = f"fastp -i {q(raw['1'])} -o {q(clean['1'])} -h {q(work)}/fastp.html -j {q(fastp_json)} -w {t}"
        timed("fastp", lambda: sh(cmd, "fastp", log))
        metrics = qc_metrics.fastp_metrics(fastp_json)
        if not metrics["reads_kept"]:
            raise StepError("fastp: no reads left after trimming")

        # same rule as the workflow: reads shorter than 50 bp use the k=15 minimizer index
        graph = os.path.join(res, RESOURCES["vg_index"])
        short = (metrics["mean_length_r1"] or 0) < 50
        minimizer = graph + (".d2.k15.min" if short and os.path.exists(graph + ".d2.k15.min") else ".d2.min")
        gam, bam = os.path.join(work, "s.gam"), os.path.join(work, "s_sort.bam")
        reads = " ".join(f"-f {q(clean[m])}" for m in mates)
        timed("giraffe", lambda: sh(
            f"vg giraffe -Z {q(graph)}.d2.gbz {reads} -d {q(graph)}.d2.dist -m {q(minimizer)} -t {t} -o GAM > {q(gam)}",
            "giraffe", log))
        vg_stats = os.path.join(work, "vg_stats.txt")
        prefix = "#".join(GREF.split("#")[:-1]) + "#"
        timed("surject", lambda: sh(
            f"vg stats -a {q(gam)} > {q(vg_stats)} && "
            f"vg surject -x {q(graph)}.d2.gbz -b {'-i ' if paired else ''}-p '{GREF}' -t {t} {q(gam)} | "
            f"samtools view -h - | sed 's/{prefix}//g' | samtools sort -O bam -o {q(bam)} - && samtools index {q(bam)}",
            "surject", log))

        bed = os.path.join(res, RESOURCES["annotation_bed"])
        infer = os.path.join(work, "infer_experiment.txt")
        timed("infer_experiment", lambda: sh(f"infer_experiment.py -i {q(bam)} -r {q(bed)} > {q(infer)}",
                                             "infer_experiment", log))
        strand = qc_metrics.strand_metrics(infer, opts["min_strand_fraction"])
        overlap = os.path.join(work, "feature_overlap_mqc.tsv")
        timed("feature_overlap", lambda: sh(
            f"bash {q(os.path.join(qc_metrics.SCRIPTS_DIR, 'gene_type.sh'))} {q(bed)} {q(bam)} {q(overlap)} "
            f"{q(qc_metrics.SCRIPTS_DIR)} {strand['strand_call']} {'PE' if paired else 'SE'}", "feature_overlap", log))

        kreport = os.path.join(work, "kreport.txt")
        if opts["centrifuge_spots"] > 0:
            # reads on rRNA / tRNA genes, to report the species measure with and without them
            stable_bed, stable_ids = os.path.join(work, "stable.bed"), os.path.join(work, "stable_ids.txt")
            timed("stable_reads", lambda: sh(
                f"awk -F'\\t' '$4 ~ /^(rRNA|tRNA):/' {q(bed)} > {q(stable_bed)} && "
                f"bedtools intersect -u -a {q(bam)} -b {q(stable_bed)} | samtools view - | cut -f1 | sort -u > {q(stable_ids)}",
                "stable_reads", log))
            sub = {m: os.path.join(work, f"cf_{m}.fastq") for m in mates}
            for m in mates:
                rec["centrifuge_reads"] = head_fastq(clean[m], sub[m], opts["centrifuge_spots"])
            db, mm = opts["centrifuge_db"], "--mm " if opts["centrifuge_mm"] else ""
            inputs = f"-1 {q(sub['1'])} -2 {q(sub['2'])}" if paired else f"-U {q(sub['1'])}"
            # one pass, then the workflow's filter (classified, single best match) before the report
            timed("centrifuge", lambda: sh(
                f"centrifuge {mm}-x {q(db)} {inputs} -S {q(work)}/cf_all.tsv -p {t} --report-file /dev/null "
                f"&& awk -F'\\t' 'NR==1 || ($3!=0 && $8==1)' {q(work)}/cf_all.tsv > {q(work)}/cf.tsv "
                f"&& centrifuge-kreport -x {q(db)} {q(work)}/cf.tsv > {q(kreport)} "
                f"&& centrifuge-kreport -x {q(db)} {q(work)}/cf_all.tsv > {q(work)}/kreport_all.txt", "centrifuge", log))
        rec["metrics"] = qc_metrics.collect({"fastp": fastp_json, "vg_stats": vg_stats, "infer_experiment": infer,
                                             "feature_overlap": overlap, "kreport": kreport,
                                             "kreport_all": os.path.join(work, "kreport_all.txt"),
                                             "classifications": os.path.join(work, "cf_all.tsv"),
                                             "stable_ids": os.path.join(work, "stable_ids.txt")},
                                            opts["min_strand_fraction"])
        if not opts["keep_scratch"]:
            shutil.rmtree(work, ignore_errors=True)
    except (StepError, sra_reads.DownloadError, sra_reads.AccessionError, OSError, ValueError, KeyError) as e:
        rec["error"] = f"{step}: {e}"[:600]
        if not opts["keep_scratch"]:
            for name in os.listdir(work):       # keep the log of a failed sample, drop the reads
                if name != "log.txt":
                    path = os.path.join(work, name)
                    shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
    return rec


CENTRIFUGE_SUFFIXES = (".1.cf", ".2.cf", ".3.cf")


def cache_centrifuge_index(resources: str, cache_dir: str) -> str:
    """Copy the Centrifuge index to `cache_dir` unless it is already there; return its prefix.
    Each file is copied under a temporary name and renamed, so several prescreen processes can
    start at once."""
    src = os.path.join(resources, RESOURCES["centrifuge_index"])
    dst = os.path.join(cache_dir, os.path.basename(src))
    os.makedirs(cache_dir, exist_ok=True)
    for suffix in CENTRIFUGE_SUFFIXES:
        if os.path.exists(dst + suffix) and os.path.getsize(dst + suffix) == os.path.getsize(src + suffix):
            continue
        tmp = f"{dst}{suffix}.{os.getpid()}.part"
        shutil.copyfile(src + suffix, tmp)
        os.replace(tmp, dst + suffix)
    return dst


def write_record(rec: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "samples", rec["GSM"] + ".json")
    with open(path + ".part", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(path + ".part", path)


def read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def in_shard(gsm: str, shard: tuple[int, int]) -> bool:
    return zlib.crc32(gsm.encode()) % shard[1] == shard[0] - 1


def select_samples(samples: list[dict], out_dir: str, status: str = "candidate", gsm=(), gse=(),
                   shard: tuple[int, int] = (1, 1), redo: bool = False, redo_errors: bool = False,
                   limit: int | None = None) -> list[dict]:
    chosen = []
    for s in samples:
        if status != "any" and s["status"] != status:
            continue
        if gsm and s["GSM"] not in gsm:
            continue
        if gse and s["GSE"] not in gse:
            continue
        if not in_shard(s["GSM"], shard):
            continue
        done = os.path.join(out_dir, "samples", s["GSM"] + ".json")
        if os.path.exists(done) and not redo:
            if not redo_errors:
                continue
            with open(done, encoding="utf-8") as fh:
                if "error" not in json.load(fh):
                    continue
        chosen.append(s)
    return chosen[:limit] if limit else chosen


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="directory with samples.tsv and runs.tsv (build_manifest.py)")
    ap.add_argument("--out", required=True, help="output directory; one JSON per sample under <out>/samples")
    ap.add_argument("--resources", required=True, help="inGEST resources directory")
    ap.add_argument("--scratch", default=None, help="working directory for reads (default <out>/scratch)")
    ap.add_argument("--jobs", type=int, default=4, help="samples screened at the same time")
    ap.add_argument("--threads", type=int, default=2, help="threads per sample")
    ap.add_argument("--spots", type=int, default=200_000, help="reads (pairs) taken per sample")
    ap.add_argument("--skip-spots", type=int, default=50_000, help="spots skipped first (sra-tools only)")
    ap.add_argument("--centrifuge-spots", type=int, default=50_000, help="trimmed reads classified; 0 = skip Centrifuge")
    ap.add_argument("--index-cache", default="/dev/shm/ecoli_ingest_centrifuge",
                    help="memory-backed or local directory the Centrifuge index is copied to; 'none' to read it "
                         "from the resources directory")
    ap.add_argument("--remove-index-cache", action="store_true", help="delete the cached index when done")
    ap.add_argument("--method", default="auto", help="auto | sra-tools | ena")
    ap.add_argument("--fetch-interval", type=float, default=2.0,
                    help="minimum seconds between the starts of two read downloads, over all jobs (rate limit "
                         "for NCBI / ENA); 0 = no limit")
    ap.add_argument("--min-strand-fraction", type=float, default=0.8, help="at least 0.8: unstranded libraries are never accepted")
    ap.add_argument("--status", default="candidate", help="manifest status to screen, or 'any'")
    ap.add_argument("--gsm", nargs="*", default=[], help="only these samples")
    ap.add_argument("--gse", nargs="*", default=[], help="only these studies")
    ap.add_argument("--shard", default="1/1", help="i/n: screen the i-th of n parts of the sample list")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true", help="screen again samples that already have a JSON")
    ap.add_argument("--redo-errors", action="store_true", help="screen again samples whose JSON records an error")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the samples that would be screened and exit")
    args = ap.parse_args(argv)
    if not args.min_strand_fraction >= 0.8:
        ap.error("--min-strand-fraction is at least 0.8: unstranded libraries are never accepted")

    i, _, n = args.shard.partition("/")
    shard = (int(i), int(n or 1))
    if not 1 <= shard[0] <= shard[1]:
        ap.error("--shard must be i/n with 1 <= i <= n")
    samples = read_tsv(os.path.join(args.manifest, "samples.tsv"))
    runs = {r["run_accession"]: r for r in read_tsv(os.path.join(args.manifest, "runs.tsv"))}
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    todo = select_samples(samples, args.out, args.status, set(args.gsm), set(args.gse), shard, args.redo,
                          args.redo_errors, args.limit)
    print(f"{len(todo)} samples to screen (shard {shard[0]}/{shard[1]})", file=sys.stderr)
    if args.list:
        print("\n".join(s["GSM"] for s in todo))
        return 0
    missing = [tool for tool in TOOLS if not shutil.which(tool)]
    if args.centrifuge_spots <= 0:
        missing = [m for m in missing if not m.startswith("centrifuge")]
    if missing:
        print(f"prescreen: tools not on PATH: {', '.join(missing)}", file=sys.stderr)
        return 1
    absent = [p for p in (os.path.join(args.resources, RESOURCES["vg_index"]) + ".d2.gbz",
                          os.path.join(args.resources, RESOURCES["annotation_bed"])) if not os.path.exists(p)]
    if absent:
        print(f"prescreen: missing resources: {', '.join(absent)}", file=sys.stderr)
        return 1
    opts = {"scratch": args.scratch or os.path.join(args.out, "scratch"), "resources": os.path.abspath(args.resources),
            "threads": args.threads, "spots": args.spots, "skip_spots": args.skip_spots,
            "centrifuge_spots": args.centrifuge_spots, "method": args.method,
            "fetch_interval": args.fetch_interval,
            "min_strand_fraction": args.min_strand_fraction, "keep_scratch": args.keep_scratch,
            "pipeline": git_commit()}
    os.makedirs(opts["scratch"], exist_ok=True)
    opts["centrifuge_db"] = os.path.join(opts["resources"], RESOURCES["centrifuge_index"])
    opts["centrifuge_mm"] = False
    if args.centrifuge_spots > 0 and todo:
        if not all(os.path.exists(opts["centrifuge_db"] + x) for x in CENTRIFUGE_SUFFIXES):
            print(f"prescreen: Centrifuge index not found at {opts['centrifuge_db']}", file=sys.stderr)
            return 1
        if args.index_cache != "none":
            t_copy = time.time()
            opts["centrifuge_db"] = cache_centrifuge_index(opts["resources"], args.index_cache)
            opts["centrifuge_mm"] = True
            print(f"Centrifuge index at {opts['centrifuge_db']} ({time.time() - t_copy:.0f} s)", file=sys.stderr)
    n_ok = n_err = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {}
        for s in todo:
            first_run = s["runs"].split(",")[0]
            futures[pool.submit(screen_sample, s, runs[first_run], opts)] = s["GSM"]
        for fut in as_completed(futures):
            rec = fut.result()
            write_record(rec, args.out)
            if "error" in rec:
                n_err += 1
                print(f"{rec['GSM']}: ERROR {rec['error']}", file=sys.stderr, flush=True)
            else:
                n_ok += 1
                m = rec["metrics"]
                print(f"{rec['GSM']}: {m.get('strand_call')} fwd {m.get('fraction_forward')} rev "
                      f"{m.get('fraction_reverse')}, aligned {m.get('pct_aligned')}%, E. coli {m.get('pct_ecoli')}%, "
                      f"length {m.get('mean_length_r1')}, {sum(rec['seconds'].values()):.0f} s", file=sys.stderr, flush=True)
    print(f"done: {n_ok} screened, {n_err} errors, {time.time() - t0:.0f} s", file=sys.stderr)
    if args.remove_index_cache and opts["centrifuge_mm"]:
        shutil.rmtree(args.index_cache, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
