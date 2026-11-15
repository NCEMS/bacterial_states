#!/usr/bin/env python3
"""Reduce a GEO SOFT file to a digest for label reading.

Usage: geo_digest.py SOFT_DIR OUT_DIR samples.tsv [GSE ...]
Writes OUT_DIR/<GSE>.md: study text, fields that are the same for every sample (once), and a
table of the candidate samples against the fields that differ between them.
"""
import csv, sys, pathlib, collections, re

FIELDS = ["title", "source_name_ch1", "characteristics_ch1", "growth_protocol_ch1",
          "treatment_protocol_ch1", "description", "extract_protocol_ch1",
          "library_selection", "library_strategy", "molecule_ch1", "organism_ch1"]

def parse(path):
    series, samples, cur = collections.defaultdict(list), {}, None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("^SAMPLE"):
            cur = collections.defaultdict(list); samples[line.split("=", 1)[1].strip()] = cur
        elif line.startswith("^"):
            cur = None
        elif line.startswith("!Series_") and " = " in line:
            k, v = line[8:].split(" = ", 1); series[k].append(v)
        elif cur is not None and line.startswith("!Sample_") and " = " in line:
            k, v = line[8:].split(" = ", 1); cur[k].append(v)
    return series, samples

def flat(sample):
    """field -> text; characteristics are split into one field per key."""
    out = {}
    for f in FIELDS:
        vals = sample.get(f, [])
        if not vals: continue
        if f == "characteristics_ch1":
            for v in vals:
                k, _, x = v.partition(":")
                key = "ch:" + k.strip().lower()
                out[key] = (out[key] + "; " if key in out else "") + x.strip()
        else:
            out[f.replace("_ch1", "")] = " ".join(vals)
    return out

def add_suppl(gse, rows, base=pathlib.Path(__file__).parent / "suppl_fields"):
    """Merge suppl_fields/<GSE>.tsv into the per-sample fields as `sup:<column>`: a table of GSM +
    columns taken from a metadata sheet the submitters deposited in GEO. Leading lines of the form
    `# key: value` give fields that are the same for every sample of the study."""
    f = base / f"{gse}.tsv"
    if f.exists():
        lines = f.read_text().splitlines()
        const = dict(l[1:].split(":", 1) for l in lines if l.startswith("#") and ":" in l)
        const = {"sup:" + k.strip(): v.strip() for k, v in const.items()}
        for r in csv.DictReader([l for l in lines if not l.startswith("#")], delimiter="\t"):
            if r["GSM"] in rows:
                rows[r["GSM"]].update({"sup:" + k: v for k, v in r.items() if k != "GSM" and v != ""})
        for fields in rows.values():
            fields.update(const)
    return rows


def main():
    soft_dir, out_dir, samples_tsv = map(pathlib.Path, sys.argv[1:4])
    want = sys.argv[4:] or [p.stem for p in sorted(soft_dir.glob("*.soft"))]
    status = {r["GSM"]: r["status"] for r in csv.DictReader(open(samples_tsv), delimiter="\t")}
    out_dir.mkdir(parents=True, exist_ok=True)
    for gse in want:
        p = soft_dir / f"{gse}.soft"
        if not p.exists(): print(gse, "no SOFT file"); continue
        series, samples = parse(p)
        rows = add_suppl(gse, {g: flat(s) for g, s in samples.items() if status.get(g) == "candidate"})
        skipped = len(samples) - len(rows)
        keys = sorted({k for r in rows.values() for k in r})
        const = {k: next(iter(rows.values())).get(k, "") for k in keys
                 if len({r.get(k, "") for r in rows.values()}) == 1}
        var = [k for k in keys if k not in const]
        L = [f"# {gse}: {' '.join(series['title'])}", "",
             f"candidate samples: {len(rows)} (other samples in the series, not shown: {skipped})", "",
             "## Summary", " ".join(series["summary"]), "", "## Overall design", " ".join(series["overall_design"]), "",
             "## Same for every sample"]
        L += [f"- **{k}**: {v}" for k, v in const.items() if v]
        L += ["", "## Differs between samples", "GSM\t" + "\t".join(var)]
        # long protocol texts: give each distinct text a number and print it once
        longtxt = {}
        for g, r in rows.items():
            cells = []
            for k in var:
                v = r.get(k, "")
                if len(v) > 160:
                    n = longtxt.setdefault(v, len(longtxt) + 1); v = f"[text {n}]"
                cells.append(v)
            L.append(g + "\t" + "\t".join(cells))
        if longtxt:
            L += ["", "## Long texts"] + [f"[text {n}] {t}" for t, n in longtxt.items()]
        (out_dir / f"{gse}.md").write_text("\n".join(L) + "\n")
        print(gse, len(rows), "samples", len(var), "varying fields", (out_dir / f"{gse}.md").stat().st_size, "bytes")

if __name__ == "__main__":
    main()
