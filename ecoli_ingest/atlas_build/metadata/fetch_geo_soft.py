#!/usr/bin/env python3
"""Fetch the GEO SOFT text (series + samples, no data tables) of every study with candidate samples.
Usage: fetch_geo_soft.py studies.tsv outdir [interval_s]"""
import csv, sys, time, urllib.request, pathlib
studies, out = sys.argv[1], pathlib.Path(sys.argv[2]); wait = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
gses = [r["GSE"] for r in csv.DictReader(open(studies), delimiter="\t") if int(r["n_candidate"]) > 0]
for i, g in enumerate(gses):
    f = out / f"{g}.soft"
    if f.exists() and f.stat().st_size > 200: continue
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g}&targ=all&form=text&view=brief"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geo-metadata-fetch/1.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
            f.write_bytes(data); break
        except Exception as e:
            print(g, "retry", attempt, e, flush=True); time.sleep(10 * (attempt + 1))
    else:
        print(g, "FAILED", flush=True)
    if i % 25 == 0: print(i, len(gses), g, flush=True)
    time.sleep(wait)
print("done", flush=True)
