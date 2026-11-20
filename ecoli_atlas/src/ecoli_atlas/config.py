"""Training configuration and hyper-parameters, with round-trips to/from ``manifest.json``."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

# The supervised calls: head name -> (negative class, positive class). The probability a head
# reports is that of the positive class. Each call has its own block of the embedding.
HEADS = {"phase": ("exponential", "stationary"), "medium": ("rich", "minimal"), "lab_strain": ("non_lab", "lab")}


# The phase head (TrainConfig.phase_head):
#   binary   exponential against stationary, as the other calls
#   ordinal  one score with a threshold between neighbouring levels of PHASE_LEVELS; the call
#            (probability of `stationary`) is the probability of lying above the last threshold
PHASE_HEADS = ("binary", "ordinal")
PHASE_LEVELS = ("exponential", "transition", "stationary")


def fit_classes(head: str, phase_head: str = "binary") -> tuple:
    """The classes a head is fitted on, in the order of their codes."""
    return PHASE_LEVELS if head == "phase" and phase_head == "ordinal" else HEADS[head]


# How the model is fitted (TrainConfig.fit):
#   two_stage     block encoders on their labelled samples, then the residual encoder and the decoder with the blocks frozen
#   one_pass      the same blocks and residual, all fitted together in one loss
#   unstructured  one embedding without blocks, heads on the whole of it, all fitted together
FITS = ("two_stage", "one_pass", "unstructured")


@dataclass
class HParams:
    # the embedding is [one block per call | residual block]; the sizes are those of the first atlas model
    block_dims: dict = field(default_factory=lambda: {"phase": 6, "medium": 12, "lab_strain": 4})
    res_dim: int = 24
    # block encoders (one small classifier per call, fitted on its labelled samples only)
    block_width: int = 256
    block_depth: int = 2
    block_dropout: float = 0.2
    block_lr: float = 1e-3
    # residual encoder and count decoder (fitted on all training samples, blocks frozen)
    enc_width: int = 512
    enc_depth: int = 2
    dec_width: int = 512
    dec_depth: int = 1
    dropout: float = 0.1
    cov_rank: int = 8            # rank of the decoder's covariate term
    lr: float = 1e-3
    weight_decay: float = 1e-5
    kl_w: float = 1e-3           # weight of the KL term (per dimension of the residual block, or of the unstructured embedding)
    head_w: float = 1.0          # weight of each head's cross-entropy (one_pass and unstructured fits)
    push_w: float = 0.5          # within-study push
    ctrl_w: float = 1.0          # cross-study control pull
    push_margin: float = 0.5

    @property
    def z_dim(self) -> int:
        return sum(self.block_dims[h] for h in HEADS) + self.res_dim


@dataclass
class TrainConfig:
    seed: int = 77
    fit: str = "two_stage"       # one of FITS
    phase_head: str = "binary"   # one of PHASE_HEADS
    # study-level splits
    test_frac: float = 0.10
    n_folds: int = 3
    n_candidates: int = 500
    fold_study_weight: float = 1.0
    withheld_studies: tuple = ("GSE224463",)                                  # projected only, in no split
    test_studies: tuple = ("GSE128997", "GSE189154", "GSE78756", "GSE80451")  # case studies, forced into TEST
    # a study's samples above this number are drawn at random each epoch, so that one very large
    # study does not weigh more than a large ordinary one; None = size of the second-largest study
    study_cap: int | None = None
    # the same for the labels of a call: within one class, a study counts at most as this many samples;
    # None = for every class the size of the second-largest study in that class
    label_cap: int | None = None
    # preprocessing
    min_gene_presence: float = 0.5   # a gene enters the size-factor reference if nonzero in this share of training samples
    sd_floor: float = 0.1            # per-gene sd of the log expression is floored here
    # genes a strain does not have (preprocess.missing_genes): left out of the size factor, the gene
    # statistics and the count likelihood; the encoders see the gene's mean instead
    mask_missing: bool = True
    mask_min_expected: float = 10.0  # reads a run of depleted genes must have expected at the low end of expression
    mask_low_fraction: float = 0.02  # a gene is depleted at or below this share of its low-end expectation
    mask_quantile: float = 0.25      # the low end: this quantile of a gene's normalised count over the reference samples that have it
    mask_transplant: float = 0.5     # share of the samples of a batch that also get the missing genes of another training sample
    # training
    block_epochs: int = 40       # of each block encoder
    block_batch: int = 256
    epochs: int = 150            # of the residual encoder and the decoder
    batch: int = 512
    knn_k: int = 15
    hparams: HParams = field(default_factory=HParams)

    def __post_init__(self):
        if self.fit not in FITS:
            raise ValueError(f"fit must be one of {FITS}, not {self.fit!r}")
        if self.phase_head not in PHASE_HEADS:
            raise ValueError(f"phase_head must be one of {PHASE_HEADS}, not {self.phase_head!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["withheld_studies"], d["test_studies"] = sorted(self.withheld_studies), sorted(self.test_studies)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known and k != "hparams"}
        for k in ("withheld_studies", "test_studies"):
            if k in kw:
                kw[k] = tuple(kw[k])
        hp_known = {f.name for f in fields(HParams)}
        return cls(hparams=HParams(**{k: v for k, v in (d.get("hparams") or {}).items() if k in hp_known}), **kw)
