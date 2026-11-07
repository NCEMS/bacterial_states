"""Inference: raw counts -> preprocessing -> factor encoders -> residual autoencoder -> embedding.

The steps are exactly those recorded in the bundle's manifest:
  1. drop rRNA/tRNA, map locus tags to gene symbols (feature table)
  2. reindex to the model's gene list; absent genes become 0 counts (check the coverage report!)
  3. per-sample CPM, per-gene z-score with the saved training statistics
  4. each factor encoder: block embedding + class probabilities
  5. concatenate blocks in factor order, standardize with the saved block statistics
  6. residual autoencoder -> final embedding = [standardized factor blocks | residual]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .artifacts import ModelBundle
from .preprocess import CoverageReport, GeneMap, align_to_genes, clean_counts, cpm_array

EMBED_BATCH = 1024


@dataclass
class EmbedResult:
    embedding: pd.DataFrame      # samples x embedding columns ("growth_phase[0]", ..., "residual[23]")
    predictions: pd.DataFrame    # samples x ({f}_pred, {f}_confidence, {f}_prob_{class} ...)
    block_layout: pd.DataFrame   # block, kind, start, end, width
    coverage: CoverageReport
    blocks: dict                 # factor -> raw (unstandardized) block embedding
    proba: dict                  # factor -> class probabilities


def prepare_input(counts: pd.DataFrame, bundle: ModelBundle, genemap: GeneMap | None):
    """genes x samples counts -> (z-scored float32 matrix samples x model genes, sample ids, coverage)."""
    X = clean_counts(counts, genemap) if genemap is not None else counts.T.copy()
    X.columns = X.columns.astype(str)
    aligned, coverage = align_to_genes(X, bundle.genes)
    Z = bundle.zscore.apply(cpm_array(aligned.values.astype(np.float64)))
    return Z, list(aligned.index), coverage


@torch.no_grad()
def encode(Z: np.ndarray, bundle: ModelBundle, batch: int = EMBED_BATCH):
    """Run the factor encoders and the residual AE. Returns (final embedding, blocks, probabilities)."""
    dev = bundle.device
    finals, blocks, probas = [], {f: [] for f in bundle.factor_order}, {f: [] for f in bundle.factor_order}
    tb_mu = torch.tensor(bundle.tb_mu, device=dev)
    tb_sd = torch.tensor(bundle.tb_sd, device=dev)
    for i in range(0, Z.shape[0], batch):
        x = torch.tensor(Z[i:i + batch], dtype=torch.float32, device=dev)
        parts = []
        for f in bundle.factor_order:
            z, logits = bundle.factors[f].model(x)
            parts.append(z)
            blocks[f].append(z.cpu().numpy())
            probas[f].append(torch.softmax(logits, dim=1).cpu().numpy())
        tb = (torch.cat(parts, dim=1) - tb_mu) / tb_sd
        _, z_final, _ = bundle.residual(x, tb)
        finals.append(z_final.cpu().numpy())
    final = np.concatenate(finals) if finals else np.zeros((0, bundle.embedding_dim), np.float32)
    return final, {f: np.concatenate(v) for f, v in blocks.items()}, {f: np.concatenate(v) for f, v in probas.items()}


def predict_labels(proba: dict, bundle: ModelBundle, sample_ids: list, apply_thresholds=()) -> pd.DataFrame:
    """Class calls per factor: argmax, or for factors in ``apply_thresholds`` the saved operating
    threshold on P(positive class)."""
    df = pd.DataFrame(index=pd.Index(sample_ids, name="sample"))
    for f in bundle.factor_order:
        fa = bundle.factors[f]
        p = proba[f]
        classes = fa.classes
        use_thr = (f in apply_thresholds and fa.threshold is not None and fa.positive_class in classes
                   and len(classes) == 2)
        if use_thr:
            pos = classes.index(fa.positive_class)
            labels = np.where(p[:, pos] >= fa.threshold, classes[pos], classes[1 - pos])
        else:
            labels = np.array([classes[i] for i in p.argmax(1)])
        df[f"{f}_pred"] = labels
        df[f"{f}_confidence"] = p.max(1)
        for j, c in enumerate(classes):
            df[f"{f}_prob_{c}"] = p[:, j]
    return df


def embed_counts(counts: pd.DataFrame, bundle: ModelBundle, genemap: GeneMap | None = None,
                 apply_thresholds=(), batch: int = EMBED_BATCH) -> EmbedResult:
    Z, sample_ids, coverage = prepare_input(counts, bundle, genemap)
    final, blocks, proba = encode(Z, bundle, batch)
    emb = pd.DataFrame(final, index=pd.Index(sample_ids, name="sample"), columns=bundle.embedding_columns())
    preds = predict_labels(proba, bundle, sample_ids, apply_thresholds)
    return EmbedResult(emb, preds, bundle.layout_frame(), coverage, blocks, proba)


def embed_zscored(Z: np.ndarray, sample_ids: list, bundle: ModelBundle, apply_thresholds=(),
                  batch: int = EMBED_BATCH) -> EmbedResult:
    """Same as embed_counts but starting from an already z-scored matrix (used after training)."""
    final, blocks, proba = encode(Z, bundle, batch)
    emb = pd.DataFrame(final, index=pd.Index(sample_ids, name="sample"), columns=bundle.embedding_columns())
    preds = predict_labels(proba, bundle, sample_ids, apply_thresholds)
    return EmbedResult(emb, preds, bundle.layout_frame(), CoverageReport(len(bundle.genes), len(bundle.genes), []),
                       blocks, proba)
