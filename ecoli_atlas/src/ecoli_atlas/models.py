"""Network definitions. Constructor signatures and submodule names match the original notebook so
that previously saved ``state_dict`` files load unchanged."""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(d_in: int, width: int, depth: int, dropout: float, d_out: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
        d = width
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


class FactorEncoder(nn.Module):
    """encoder(x) -> block embedding; head(embedding) -> class logits. No decoder."""

    def __init__(self, n_genes: int, emb_dim: int, n_out: int, width: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = _mlp(n_genes, width, depth, dropout, emb_dim)
        self.head = nn.Linear(emb_dim, n_out)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.head(z)

    def arch(self) -> dict:
        lins = [l for l in self.encoder if isinstance(l, nn.Linear)]
        drops = [l.p for l in self.encoder if isinstance(l, nn.Dropout)]
        return dict(n_genes=lins[0].in_features, width=lins[0].out_features, emb_dim=lins[-1].out_features,
                    depth=len(lins) - 1, dropout=float(drops[0]) if drops else 0.0, n_out=self.head.out_features)

    @classmethod
    def from_arch(cls, a: dict) -> "FactorEncoder":
        return cls(a["n_genes"], a["emb_dim"], a["n_out"], a["width"], a["depth"], a["dropout"])


class ResidualAE(nn.Module):
    """The encoder sees only the expression profile and emits the residual block. The factor
    blocks are concatenated to it for the decoder and form the final embedding ``z``."""

    def __init__(self, d_raw: int, d_block: int, d_res: int, width: int, depth: int, dropout: float):
        super().__init__()
        self.encoder = _mlp(d_raw, width, depth, dropout, d_res)
        self.decoder = _mlp(d_block + d_res, width, depth, dropout, d_raw)

    def forward(self, x_raw, blocks):
        r = self.encoder(x_raw)
        z = torch.cat([blocks, r], dim=1)
        return r, z, self.decoder(z)

    def arch(self) -> dict:
        e = [l for l in self.encoder if isinstance(l, nn.Linear)]
        dec = [l for l in self.decoder if isinstance(l, nn.Linear)]
        drops = [l.p for l in self.encoder if isinstance(l, nn.Dropout)]
        return dict(d_raw=e[0].in_features, width=e[0].out_features, d_res=e[-1].out_features,
                    depth=len(e) - 1, dropout=float(drops[0]) if drops else 0.0,
                    d_block=dec[0].in_features - e[-1].out_features)

    @classmethod
    def from_arch(cls, a: dict) -> "ResidualAE":
        return cls(a["d_raw"], a["d_block"], a["d_res"], a["width"], a["depth"], a["dropout"])
