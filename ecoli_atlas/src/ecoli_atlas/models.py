"""The atlas models. Two layouts of the embedding, with the same count decoder.

``AtlasModel`` (fits ``two_stage`` and ``one_pass``): one supervised block per call and a residual block.

    block encoders   z-scored log expression -> the call's block of the embedding; a linear head on
                     the block gives the class logits
    residual encoder [z-scored log expression, covariates] -> mean and log variance of the residual block
                     (expression only with ``enc_cov`` False, ``train --no-encoder-covariates``)
    embedding        z = [blocks | residual block]

``UnstructuredModel`` (fit ``unstructured``): one embedding without blocks.

    encoder          [z-scored log expression, covariates] -> mean and log variance of z
    heads            linear classifiers on the mean of z

Both:

    decoder          log rate_g = a_g + f_g(z) + (B c)_g      B has low rank
    counts           negative binomial with mean s * rate_g + k * d_g and a dispersion per feature;
                     d_g are the genomic DNA reads expected on the feature from the sample's QC
                     (``data.dna_density``) and k one learned factor (``count_log_mean``)

With the ordinal phase head (``phase_head="ordinal"``) the phase head is an ``OrdinalHead``: one
score and a threshold between neighbouring phase levels. Its call, the probability of `stationary`,
is the probability of lying above the last threshold.

In the two-stage fit the blocks are fitted first, standardised with training statistics and frozen
while the residual encoder and the decoder are fitted. In the one-pass fit the blocks move together
with everything else and are not standardised (``block_mu`` 0, ``block_sd`` 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import HEADS, PHASE_LEVELS, HParams


def mlp(n_in: int, width: int, depth: int, dropout: float) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
        d = width
    return nn.Sequential(*layers)


class OrdinalHead(nn.Module):
    """Proportional-odds head: output column k is the logit of P(level > k), a linear score minus
    the k-th threshold. The first threshold is 0 and the gaps to the next ones are positive."""

    def __init__(self, dim: int, n_levels: int = len(PHASE_LEVELS)):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.gaps = nn.Parameter(torch.zeros(n_levels - 2))

    def cuts(self) -> torch.Tensor:
        return torch.cat([self.gaps.new_zeros(1), torch.cumsum(nn.functional.softplus(self.gaps), 0)])

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return self.score(b) - self.cuts()


def make_head(dim: int, ordinal: bool) -> nn.Module:
    return OrdinalHead(dim) if ordinal else nn.Linear(dim, 2)


def call_logits(head: nn.Module, out: torch.Tensor) -> torch.Tensor:
    """Two-class logits of a head's call from its output: for an ordinal head [0, logit of P(last level)]."""
    return torch.stack([torch.zeros_like(out[:, -1]), out[:, -1]], dim=1) if isinstance(head, OrdinalHead) else out


def split_outputs(heads: dict, outs: dict) -> tuple[dict, dict]:
    """(two-class logits of every head, cumulative logits of the ordinal heads)."""
    return ({h: call_logits(heads[h], o) for h, o in outs.items()},
            {h: o for h, o in outs.items() if isinstance(heads[h], OrdinalHead)})


class BlockEncoder(nn.Module):
    """encoder(x) -> block; head(block) -> class logits (cumulative logits for an ordinal head)."""

    def __init__(self, n_features: int, dim: int, hp: HParams, ordinal: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(mlp(n_features, hp.block_width, hp.block_depth, hp.block_dropout), nn.Linear(hp.block_width, dim))
        self.head = make_head(dim, ordinal)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.encoder(x)
        return b, self.head(b)


class GCTerm(nn.Module):
    """Mechanistic GC bias in the decoder's log rate (``train --gc-offset``): the library's own GC fit
    (inGEST ``gc_bias``: log coverage = b1 x + b2 x^2 inside genes, x = (GC - 0.5) * 10, reported as
    gc_slope = b1 + 2 b2 mean(x) and gc_curv = b2) at every feature's GC content, centred over the mRNA
    features, times one learned scale (starts at 1). It reads gc_slope and gc_curv from the scaled
    covariates; ``keep`` takes them out of the linear covariate term, so the decoder explains GC once.
    The values are set by ``configure`` before training and travel with the weights."""

    def __init__(self, n_features: int, n_cov: int, n_bins: int = 0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.n_bins = n_bins                                                   # 0: the fit at the feature's mean GC
        self.register_buffer("x", torch.zeros(n_features))
        if n_bins:                                                             # window profile (technical.GCProfile)
            self.register_buffer("P", torch.zeros(n_features, n_bins))
            self.register_buffer("xk", torch.zeros(n_bins))
        self.register_buffer("mrna", torch.ones(n_features, dtype=torch.bool))
        self.register_buffer("xbar", torch.zeros(()))
        self.register_buffer("idx", torch.zeros(2, dtype=torch.long))           # columns of gc_slope, gc_curv in c
        self.register_buffer("mu", torch.zeros(2))                             # their scaling: raw = c * sd + mu
        self.register_buffer("sd", torch.ones(2))
        self.register_buffer("keep", torch.ones(n_cov))                        # 0 for the two GC columns

    @torch.no_grad()
    def configure(self, gc, is_mrna: np.ndarray, xbar: float, idx: tuple, mu: np.ndarray, sd: np.ndarray,
                  P: np.ndarray | None = None, bins: np.ndarray | None = None) -> None:
        self.x.copy_(torch.as_tensor((np.asarray(gc, dtype=np.float64) - 0.5) * 10, dtype=self.x.dtype))
        if self.n_bins:
            self.P.copy_(torch.as_tensor(P, dtype=self.P.dtype))
            self.xk.copy_(torch.as_tensor((np.asarray(bins, dtype=np.float64) - 0.5) * 10, dtype=self.xk.dtype))
        self.mrna.copy_(torch.as_tensor(np.asarray(is_mrna, dtype=bool)))
        self.xbar.fill_(float(xbar))
        self.idx.copy_(torch.as_tensor(idx, dtype=torch.long))
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(sd, dtype=self.sd.dtype))
        self.keep.fill_(1.0)
        self.keep[self.idx] = 0.0

    def offset(self, c: torch.Tensor) -> torch.Tensor:
        """The GC offset at scale 1 (samples x features): at the feature's mean GC, or with a window profile the
        log of the mean of exp(b1 x + b2 x^2) over the feature's windows (``technical.gc_offset``)."""
        raw = c[:, self.idx] * self.sd + self.mu
        slope, curv = raw[:, 0:1], raw[:, 1:2]
        b1 = slope - 2 * curv * self.xbar
        if self.n_bins:
            off = torch.log(torch.exp(b1 * self.xk + curv * self.xk ** 2) @ self.P.T)
        else:
            off = b1 * self.x + curv * self.x ** 2
        return off - off[:, self.mrna].mean(dim=1, keepdim=True)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.scale * self.offset(c)


def covariate_term(model: nn.Module, c: torch.Tensor) -> torch.Tensor:
    """The decoder's covariate part of the log rate: the low-rank linear term (without the GC columns
    when the model has a GC term) plus the GC term."""
    gc = getattr(model, "gc_term", None)
    if gc is None:
        return model.cov_out(model.cov_in(c))
    return model.cov_out(model.cov_in(c * gc.keep)) + gc(c)


class AtlasModel(nn.Module):
    def __init__(self, n_features: int, n_cov: int, hp: HParams, log_mean: torch.Tensor | None = None, phase_head: str = "binary",
                 dna: bool = False, gc: bool = False, enc_cov: bool = True, gc_bins: int = 0):
        super().__init__()
        if gc:
            self.gc_term = GCTerm(n_features, n_cov, gc_bins)
        self.n_features, self.n_cov, self.hp, self.enc_cov = n_features, n_cov, hp, enc_cov
        if dna:
            self.log_dna_scale = nn.Parameter(torch.zeros(()))        # log of the factor on the expected DNA reads
        self.blocks = nn.ModuleDict({h: BlockEncoder(n_features, hp.block_dims[h], hp, ordinal=h == "phase" and phase_head == "ordinal")
                                     for h in HEADS})
        d_block = sum(hp.block_dims[h] for h in HEADS)
        self.register_buffer("block_mu", torch.zeros(d_block))
        self.register_buffer("block_sd", torch.ones(d_block))
        self.encoder = mlp(n_features + (n_cov if enc_cov else 0), hp.enc_width, hp.enc_depth, hp.dropout)
        self.to_mu = nn.Linear(hp.enc_width, hp.res_dim)
        self.to_logvar = nn.Linear(hp.enc_width, hp.res_dim)
        self.decoder = nn.Sequential(mlp(hp.z_dim, hp.dec_width, hp.dec_depth, hp.dropout), nn.Linear(hp.dec_width, n_features))
        self.cov_in = nn.Linear(n_cov, hp.cov_rank, bias=False)
        self.cov_out = nn.Linear(hp.cov_rank, n_features, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(n_features) if log_mean is None else log_mean.clone().float())
        self.log_theta = nn.Parameter(torch.zeros(n_features))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.zeros_(self.cov_out.weight)

    def residual_parameters(self) -> list:
        """Everything that is fitted after the blocks."""
        return [p for name, p in self.named_parameters() if not name.startswith("blocks.")]

    @torch.no_grad()
    def raw_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """The blocks of all calls side by side, before standardisation (block encoders in eval mode)."""
        was_training = self.blocks.training
        self.blocks.eval()
        out = torch.cat([self.blocks[h](x)[0] for h in HEADS], dim=1)
        self.blocks.train(was_training)
        return out

    @torch.no_grad()
    def block_part(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """(standardised blocks, output of each head); no gradient reaches the block encoders."""
        was_training = self.blocks.training
        self.blocks.eval()
        outs = {h: self.blocks[h](x) for h in HEADS}
        self.blocks.train(was_training)
        b = (torch.cat([outs[h][0] for h in HEADS], dim=1) - self.block_mu) / self.block_sd
        return b, {h: outs[h][1] for h in HEADS}

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log variance of the residual block (from expression alone without ``enc_cov``)."""
        h = self.encoder(torch.cat([x, c], dim=1) if self.enc_cov else x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 5.0)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """log rate per feature (before the size factor)."""
        return self.gene_bias + self.decoder(z) + covariate_term(self, c)

    def forward(self, x: torch.Tensor, c: torch.Tensor, sample: bool = True, train_blocks: bool = False) -> dict:
        """``mu`` is the embedding (blocks and the mean of the residual); ``kl_mu`` / ``kl_logvar``
        describe the residual block alone. With ``train_blocks`` the block encoders are part of the
        graph (one-pass fit); otherwise no gradient reaches them and they run in eval mode."""
        if train_blocks:
            outs = {h: self.blocks[h](x) for h in HEADS}
            b = (torch.cat([outs[h][0] for h in HEADS], dim=1) - self.block_mu) / self.block_sd
            raw = {h: outs[h][1] for h in HEADS}
        else:
            b, raw = self.block_part(x)
        logits, cum = split_outputs({h: self.blocks[h].head for h in HEADS}, raw)
        res_mu, res_logvar = self.encode(x, c)
        r = res_mu + torch.randn_like(res_mu) * torch.exp(0.5 * res_logvar) if sample else res_mu
        z = torch.cat([b, r], dim=1)
        return {"mu": torch.cat([b, res_mu], dim=1), "kl_mu": res_mu, "kl_logvar": res_logvar, "z": z,
                "log_rate": self.decode(z, c), "logits": logits, "cum": cum}


class UnstructuredModel(nn.Module):
    def __init__(self, n_features: int, n_cov: int, hp: HParams, log_mean: torch.Tensor | None = None, phase_head: str = "binary",
                 dna: bool = False, gc: bool = False, enc_cov: bool = True, gc_bins: int = 0):
        super().__init__()
        if gc:
            self.gc_term = GCTerm(n_features, n_cov, gc_bins)
        self.n_features, self.n_cov, self.hp, self.enc_cov = n_features, n_cov, hp, enc_cov
        if dna:
            self.log_dna_scale = nn.Parameter(torch.zeros(()))        # log of the factor on the expected DNA reads
        self.encoder = mlp(n_features + (n_cov if enc_cov else 0), hp.enc_width, hp.enc_depth, hp.dropout)
        self.to_mu = nn.Linear(hp.enc_width, hp.z_dim)
        self.to_logvar = nn.Linear(hp.enc_width, hp.z_dim)
        self.decoder = nn.Sequential(mlp(hp.z_dim, hp.dec_width, hp.dec_depth, hp.dropout), nn.Linear(hp.dec_width, n_features))
        self.cov_in = nn.Linear(n_cov, hp.cov_rank, bias=False)
        self.cov_out = nn.Linear(hp.cov_rank, n_features, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(n_features) if log_mean is None else log_mean.clone().float())
        self.log_theta = nn.Parameter(torch.zeros(n_features))
        self.heads = nn.ModuleDict({h: make_head(hp.z_dim, h == "phase" and phase_head == "ordinal") for h in HEADS})
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.zeros_(self.cov_out.weight)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=1) if self.enc_cov else x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 5.0)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """log rate per feature (before the size factor)."""
        return self.gene_bias + self.decoder(z) + covariate_term(self, c)

    def forward(self, x: torch.Tensor, c: torch.Tensor, sample: bool = True, train_blocks: bool = False) -> dict:
        mu, logvar = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        logits, cum = split_outputs(self.heads, {h: head(mu) for h, head in self.heads.items()})
        return {"mu": mu, "kl_mu": mu, "kl_logvar": logvar, "z": z, "log_rate": self.decode(z, c), "logits": logits, "cum": cum}


def build_model(fit: str, n_features: int, n_cov: int, hp: HParams, log_mean: torch.Tensor | None = None,
                phase_head: str = "binary", dna: bool = False, gc: bool = False, enc_cov: bool = True, gc_bins: int = 0) -> nn.Module:
    """The model of a fit (``config.FITS``) with a phase head (``config.PHASE_HEADS``); ``dna``: with
    the learned scale of the DNA term (``count_log_mean``); ``gc``: with the GC term (``GCTerm``); ``enc_cov``:
    the residual (or unstructured) encoder sees the covariates; ``gc_bins``: GC bins of the GC term's window profile (0: none)."""
    return (UnstructuredModel if fit == "unstructured" else AtlasModel)(n_features, n_cov, hp, log_mean, phase_head, dna, gc, enc_cov,
                                                                        gc_bins)


def count_log_mean(model: nn.Module, log_rate: torch.Tensor, log_s: torch.Tensor, log_dna: torch.Tensor | None = None) -> torch.Tensor:
    """log of the expected counts: size factor x RNA rate, plus the genomic DNA reads when the
    model has a DNA term (``log_dna``: log of the reads expected from the QC estimate, -inf for a
    sample without DNA; the model's one learned factor scales them)."""
    lm = log_rate + log_s[:, None]
    if log_dna is None or not hasattr(model, "log_dna_scale"):
        return lm
    has = torch.isfinite(log_dna)
    with_dna = torch.logaddexp(lm, model.log_dna_scale + torch.where(has, log_dna, torch.zeros_like(log_dna)))
    return torch.where(has, with_dna, lm)
