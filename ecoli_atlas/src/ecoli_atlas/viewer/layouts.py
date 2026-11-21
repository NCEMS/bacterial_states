"""3-D layouts of the atlas embedding for `ecoli_atlas view`.

Every layout is fitted on the atlas samples only; samples of `embed` (the overlay) are projected with
the same fit. A layout is ``{"name", "label", "axes": [3 titles], "atlas": n x 3, "overlay": m x 3}``.

  pca_all, pca_residual   PCA of the standardised embedding (whole, or the residual block); axis titles
                          give the share of variance
  block_<head>            a head block (phase, medium, lab_strain): its first component (such a block is
                          in effect one axis) against residual PC1-2; a 3-D PCA of the block when it is not flat
  axes                    the three head-block axes together, each signed to rise with the call's positive
                          class (the logits of the calls when the bundle has no head blocks)
  umap_all, umap_residual 3-D UMAP (cosine, 30 neighbours, min_dist 0.3, as `compare`), optional
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config import HEADS

LOGIT_CLIP = 10.0
BLOCK_TITLES = {"phase": "phase block", "medium": "medium block", "lab_strain": "lab-strain block", "residual": "residual block"}


def _pad3(x: np.ndarray) -> np.ndarray:
    """Three columns (a block narrower than 3 gets zero columns)."""
    if x.shape[1] >= 3:
        return x[:, :3]
    return np.hstack([x, np.zeros((x.shape[0], 3 - x.shape[1]))])


def pca_layout(name: str, label: str, atlas: np.ndarray, overlay: np.ndarray | None, seed: int = 0) -> dict:
    scaler = StandardScaler().fit(atlas)
    a = scaler.transform(atlas)
    k = min(3, a.shape[1], a.shape[0])
    pca = PCA(n_components=k, random_state=seed).fit(a)
    var = list(pca.explained_variance_ratio_) + [0.0] * (3 - k)
    axes = [f"PC{i + 1} ({100 * v:.0f}%)" for i, v in enumerate(var)]
    out = {"name": name, "label": label, "axes": axes, "atlas": _pad3(pca.transform(a))}
    if overlay is not None and len(overlay):
        out["overlay"] = _pad3(pca.transform(scaler.transform(overlay)))
    return out


def umap_layout(name: str, label: str, atlas: np.ndarray, overlay: np.ndarray | None, seed: int = 77,
                n_neighbors: int = 30, min_dist: float = 0.3, metric: str = "cosine") -> dict:
    import umap  # optional dependency, checked by the caller
    reducer = umap.UMAP(n_components=3, n_neighbors=min(n_neighbors, len(atlas) - 1), min_dist=min_dist, metric=metric,
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")   # random_state makes UMAP single-threaded
        coords = reducer.fit_transform(atlas)
        out = {"name": name, "label": label, "axes": ["UMAP1", "UMAP2", "UMAP3"], "atlas": np.asarray(coords)}
        if overlay is not None and len(overlay):
            out["overlay"] = np.asarray(reducer.transform(overlay))
    return out


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.clip(np.log(p / (1 - p)), -LOGIT_CLIP, LOGIT_CLIP)


def axes_layout(atlas_probs: pd.DataFrame, overlay_probs: pd.DataFrame | None) -> dict | None:
    """Coordinates = logit of each call's positive class (columns ``p_<positive>``)."""
    cols = [f"p_{pos}" for _, (_, pos) in HEADS.items()]
    if not all(c in atlas_probs.columns for c in cols):
        return None
    titles = [f"logit p({pos}) — {h.replace('_', ' ')} call" for h, (_, pos) in HEADS.items()]
    out = {"name": "axes", "label": "Named axes: logits of the three calls", "axes": titles,
           "atlas": np.column_stack([logit(atlas_probs[c]) for c in cols])}
    if overlay_probs is not None and len(overlay_probs):
        out["overlay"] = np.column_stack([logit(overlay_probs[c]) if c in overlay_probs.columns
                                          else np.full(len(overlay_probs), np.nan) for c in cols])
    return out


def has_umap() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False


def _block_axis(a: np.ndarray, o: np.ndarray | None, orient: np.ndarray | None, seed: int):
    """First principal component of a block (standardised), signed to rise with ``orient``, and the
    variance shares of the components."""
    scaler = StandardScaler().fit(a)
    pca = PCA(n_components=min(3, a.shape[1]), random_state=seed).fit(scaler.transform(a))
    x = pca.transform(scaler.transform(a))[:, 0]
    sign = 1.0
    if orient is not None:
        ok = np.isfinite(orient)
        if ok.sum() > 2 and np.std(orient[ok]) > 0 and np.corrcoef(x[ok], orient[ok])[0, 1] < 0:
            sign = -1.0
    xo = None if o is None else sign * pca.transform(scaler.transform(o))[:, 0]
    return sign * x, xo, pca.explained_variance_ratio_


def build_layouts(embedding: pd.DataFrame, blocks: list[dict], atlas_probs: pd.DataFrame, overlay: pd.DataFrame | None = None,
                  overlay_probs: pd.DataFrame | None = None, umap: bool = True, seed: int = 77, log=print,
                  flat_share: float = 0.9) -> list[dict]:
    """All layouts; ``blocks`` is ``manifest["embedding_blocks"]`` (block, start, end).

    A head block whose first component holds more than ``flat_share`` of its variance is in effect one
    axis: it is drawn as that axis against residual PC1-2 instead of as a 3-D PCA of the block."""
    a = embedding.to_numpy(dtype=float)
    o = overlay[embedding.columns].to_numpy(dtype=float) if overlay is not None and len(overlay) else None
    sl = {b["block"]: slice(int(b["start"]), int(b["end"])) for b in blocks}
    out = [pca_layout("pca_all", f"PCA — whole embedding ({a.shape[1]}-d)", a, o, seed=seed)]
    res = None
    if "residual" in sl:
        w = sl["residual"].stop - sl["residual"].start
        res = pca_layout("pca_residual", f"PCA — {BLOCK_TITLES['residual']} ({w}-d)", a[:, sl["residual"]],
                         None if o is None else o[:, sl["residual"]], seed=seed)
    head_axes = {}
    for h, (_, pos) in HEADS.items():
        if h not in sl:
            continue
        orient = pd.to_numeric(atlas_probs[f"p_{pos}"], errors="coerce").to_numpy() if f"p_{pos}" in atlas_probs.columns else None
        blk, blk_o = a[:, sl[h]], None if o is None else o[:, sl[h]]
        x, xo, share = _block_axis(blk, blk_o, orient, seed)
        head_axes[h] = (x, xo)
        title = BLOCK_TITLES.get(h, h + " block")
        if share[0] < flat_share or res is None:
            out.append(pca_layout(f"block_{h}", f"PCA — {title} ({blk.shape[1]}-d)", blk, blk_o, seed=seed))
            continue
        L = {"name": f"block_{h}", "label": f"{title[0].upper()}{title[1:]} axis × residual PC1–2",
             "axes": [f"{h.replace('_', ' ')} axis → {pos} ({100 * share[0]:.0f}% of the block)",
                      "residual " + res["axes"][0], "residual " + res["axes"][1]],
             "atlas": np.column_stack([x, res["atlas"][:, 0], res["atlas"][:, 1]])}
        if xo is not None:
            L["overlay"] = np.column_stack([xo, res["overlay"][:, 0], res["overlay"][:, 1]])
        out.append(L)
    if res is not None:
        out.append(res)
    if len(head_axes) == len(HEADS) == 3:
        L = {"name": "axes", "label": "Named axes: phase × medium × lab-strain blocks",
             "axes": [f"{h.replace('_', ' ')} axis → {HEADS[h][1]}" for h in head_axes],
             "atlas": np.column_stack([v[0] for v in head_axes.values()])}
        if o is not None:
            L["overlay"] = np.column_stack([v[1] for v in head_axes.values()])
        out.append(L)
    else:
        ax = axes_layout(atlas_probs, overlay_probs)
        if ax is not None:
            out.append(ax)
    if umap:
        if not has_umap():
            log("umap-learn is not installed: UMAP layouts skipped (pip install umap-learn)")
        else:
            parts = [("all", "whole embedding", slice(0, a.shape[1]))]
            if "residual" in sl:
                parts.append(("residual", BLOCK_TITLES["residual"], sl["residual"]))
            for key, title, s_ in parts:
                log(f"UMAP of the {title} ...")
                out.append(umap_layout(f"umap_{key}", f"UMAP — {title}", a[:, s_], None if o is None else o[:, s_], seed=seed))
    return out
