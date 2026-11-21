"""GC bias of a library from its stranded coverage track, free of expression levels.

Within one gene every 100 bp bin carries the same expression, so the part of the variation of
coverage between the bins of a gene that follows their GC content is technical (amplification,
fragmentation, capture). Used by `sample_qc.py collect` (the QC record's `gc_bias`) and by the
atlas build for records made before it (`atlas_build/prescreen/qc_table.py`).

Fit, per sample: interior bins (entirely inside, without the outermost bin at each end) of CDS
features of at least MIN_LENGTH bp whose mean coverage is at least MIN_DEPTH; log coverage on
x, x^2 and the relative position in the gene (5' -> 3'), everything centred within gene, where x is
the GC share of a WINDOW bp window centred on the bin, in units of 10 percentage points from 50%.

  gc_slope  change of log coverage per 10 percentage points of GC, at the sample's mean GC
  gc_curv   curvature (coefficient of x^2); negative = bins of middling GC are covered best
  pos_slope within-gene 5' -> 3' trend of log coverage (the expression-free gene-body slope)
  gc_r2     share of the within-gene variance (after position) explained by the GC terms
"""
from __future__ import annotations

import numpy as np

BIN = 100            # bin size of the coverage track (sample_qc.py)
WINDOW = 300         # bp of sequence around a bin whose GC is used (fragments reach past the bin)
MIN_LENGTH = 600     # bp; shorter CDS features have too few interior bins
MIN_DEPTH = 5.0      # mean per-base coverage a gene needs over its interior bins
MIN_GENES = 50       # fewer usable genes: no estimate


def read_fasta(path: str) -> dict:
    """{contig (first word of the header): sequence as uint8 array, upper case}."""
    out, name, parts = {}, None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = np.frombuffer("".join(parts).upper().encode(), dtype=np.uint8)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        out[name] = np.frombuffer("".join(parts).upper().encode(), dtype=np.uint8)
    return out


def window_gc(seq: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """GC share of a `window` bp window centred on every BIN bp bin of `seq`."""
    gc = ((seq == ord("G")) | (seq == ord("C"))).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(gc)])
    n = len(seq)
    centre = np.arange(0, n, BIN) + BIN / 2
    lo = np.clip((centre - window / 2).astype(np.int64), 0, n)
    hi = np.clip((centre + window / 2).astype(np.int64), 0, n)
    return (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)


def interior_bins(bed: str, n_bins: dict, min_length: int = MIN_LENGTH) -> dict:
    """{contig: (strand +1/-1, gene index, bin, relative position)} of the interior bins of the CDS
    features of the counting annotation BED (name column `CDS:<id>`)."""
    rows: dict = {}
    with open(bed, encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            c = line.rstrip("\n").split("\t")
            if len(c) < 6 or not c[3].startswith("CDS:") or c[0] not in n_bins:
                continue
            start, end = int(c[1]), int(c[2])
            if end - start < min_length:
                continue
            first, last = -(-start // BIN), end // BIN - 1          # bins entirely inside the feature
            bins = np.arange(first + 1, last)
            bins = bins[bins < n_bins[c[0]]]
            if len(bins) < 4:
                continue
            pos = (bins * BIN + BIN / 2 - start) / (end - start)
            if c[5] == "-":
                pos = 1 - pos
            rows.setdefault(c[0], []).append((np.full(len(bins), 1 if c[5] == "+" else -1), np.full(len(bins), k), bins, pos))
    return {ctg: tuple(np.concatenate([r[i] for r in parts]) for i in range(4)) for ctg, parts in rows.items()}


def _demean(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    _, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv)
    if a.ndim == 1:
        return a - (np.bincount(inv, a) / n)[inv]
    return np.column_stack([a[:, j] - (np.bincount(inv, a[:, j]) / n)[inv] for j in range(a.shape[1])])


def fit(tracks: dict, bed: str, genome: dict, window: int = WINDOW, min_depth: float = MIN_DEPTH,
        min_length: int = MIN_LENGTH) -> dict:
    """tracks: {contig: {"plus": summed depth per bin, "minus": ...}} (or {"both": ...} for an
    unstranded track). Returns the fit (see the module text); {} when too few genes are covered."""
    n_bins = {ctg: len(next(iter(t.values()))) for ctg, t in tracks.items()}
    bins = interior_bins(bed, n_bins, min_length)
    depth, gcs, gene, pos = [], [], [], []
    for ctg, (strand, g, b, p) in bins.items():
        if ctg not in genome:
            continue
        t = tracks[ctg]
        plus, minus = (t["plus"], t["minus"]) if "plus" in t else (t["both"], t["both"])
        depth.append(np.where(strand == 1, plus[b], minus[b]).astype(np.float64) / BIN)
        gcs.append(window_gc(genome[ctg], window)[b])
        gene.append(np.char.add(ctg + ":", g.astype(str)))
        pos.append(p)
    if not depth:
        return {}
    depth, gcs, gene, pos = map(np.concatenate, (depth, gcs, gene, pos))
    _, inv = np.unique(gene, return_inverse=True)
    mean = (np.bincount(inv, depth) / np.bincount(inv))[inv]
    keep = (mean >= min_depth) & (depth > 0)
    if np.unique(inv[keep]).size < MIN_GENES:
        return {}
    y, g = np.log(depth[keep]), inv[keep]
    x = (gcs[keep] - 0.5) * 10
    X = _demean(np.column_stack([x, x ** 2, pos[keep]]), g)
    yc = _demean(y, g)
    beta = np.linalg.lstsq(X, yc, rcond=None)[0]
    b_pos = np.linalg.lstsq(X[:, 2:], yc, rcond=None)[0]
    rss_full, rss_pos = float(((yc - X @ beta) ** 2).sum()), float(((yc - X[:, 2:] @ b_pos) ** 2).sum())
    return {"gc_slope": round(float(beta[0] + 2 * beta[1] * x.mean()), 6), "gc_curv": round(float(beta[1]), 6),
            "pos_slope": round(float(beta[2]), 6), "gc_r2": round(1 - rss_full / rss_pos, 6) if rss_pos > 0 else None,
            "n_genes": int(np.unique(g).size), "n_bins": int(len(y)), "mean_gc": round(float(gcs[keep].mean()), 6),
            "window": window, "min_depth": min_depth, "min_length": min_length}


def fit_files(npz_path: str, bed: str, fasta: str, genome: dict | None = None) -> dict:
    """The fit from a `<sample>.cov100.npz` track (sample_qc.py bam) and the reference FASTA."""
    z = np.load(npz_path)
    tracks: dict = {}
    for key in z.files:
        if "__" in key:
            ctg, strand = key.rsplit("__", 1)
            tracks.setdefault(ctg, {})[strand] = z[key]
    return fit(tracks, bed, genome if genome is not None else read_fasta(fasta))
