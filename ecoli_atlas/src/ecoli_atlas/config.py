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

# The covariates the model takes (TrainConfig.covariates; data.covariate_frame computes these and
# more). Technical values of the library and the sequencing run only (user decision 2026-11-21).
# Left out as biology, since they shift by 0.8-1.5 within-study SD between exponential and stationary
# samples (`work/qc_blocks/`): the rRNA / tRNA / mRNA composition, duplication rate, gene-body slope,
# GC content, share of reads on short features. Left out as redundant or handled elsewhere: depth on
# mRNA (the size factor), the DNA estimate (samples above 10% are gated out), DNA leakage (strand
# purity), intergenic share, trimming shares, cycles and mate-2 length (read-1 lengths), other genus.
# The GC bias is the within-gene fit of the QC record (inGEST gc_bias.py; `work/gc_bias/`): it shifts
# by 0.17 SD with phase, against 0.79 for the read GC share, which it replaces. The instrument flags
# were dropped after models_v2_techqc: shuffling them changed TEST NLL by -0.005..+0.002 over seeds 1-4
# (train +0.003), i.e. they fitted the training studies only (`work/qc_blocks/NOTES.md`).
COVARIATES = ("paired", "fragment_q25", "fragment_q50", "fragment_q75",           # layout, fragment sizes (paired-end)
              "length_q05_r1", "length_q50_r1", "length_q95_r1",                  # read lengths after trimming
              "strand_purity", "log_substitution_rate",
              "gc_slope", "gc_curv")                                              # GC bias of the coverage inside genes


GC_SCALES = ("learned", "fixed")     # the scale of the decoder's GC term (TrainConfig.gc_scale)


@dataclass
class HParams:
    # the embedding is [one block per call | residual block]. One dimension per call (user decision 2026-11-21):
    # v1's 6 / 12 / 4 held nothing more than the call's score (`work/phase_dims/NOTES.md`, sweeps
    # models_v2_cov11_order_w1{,_phase1,_blocks1}); the residual keeps v1's 24
    block_dims: dict = field(default_factory=lambda: {"phase": 1, "medium": 1, "lab_strain": 1})
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

    def with_block_dims(self, text: str) -> "HParams":
        """A copy with some block sizes changed: ``text`` is ``name=size[,name=size...]`` over the
        call blocks (``phase``, ``medium``, ``lab_strain``) and ``residual``; blocks not named keep their size."""
        from dataclasses import replace
        dims, res = dict(self.block_dims), self.res_dim
        for part in filter(None, (p.strip() for p in text.split(","))):
            name, _, size = part.partition("=")
            name = name.strip()
            if name not in (*HEADS, "residual"):
                raise ValueError(f"--block-dims: unknown block {name!r} (blocks: {', '.join(HEADS)}, residual)")
            try:
                n = int(size)
            except ValueError:
                raise ValueError(f"--block-dims: {part!r} is not name=size") from None
            if n < 1:
                raise ValueError(f"--block-dims: block {name} needs at least 1 dimension")
            if name == "residual":
                res = n
            else:
                dims[name] = n
        return replace(self, block_dims=dims, res_dim=res)


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
    # forced into TEST: the case studies, and GSE67402, the worked example (examples/GSE67402_*), kept out of training so that
    # the example stays out of sample (it was already TEST in the split of 2026-11-21)
    test_studies: tuple = ("GSE128997", "GSE189154", "GSE78756", "GSE80451", "GSE67402")
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
    mask_veto_reads: float = 2.0     # with antisense counts: a gene with more antisense reads than this ...
    mask_veto_fraction: float = 0.25 # ... and than this share of the sample's antisense background is present
    mask_pool: bool = True           # also call the genes missing from a study's samples of one strain on their summed counts
    mask_pool_min_calls: int = 20    # samples of a study with fewer calls than this form one group ...
    mask_pool_jaccard: float = 0.5   # ... the others are joined when their call sets overlap this much
    mask_transplant: float = 0.5     # share of the samples of a batch that also get the missing genes of another training sample
    # genomic DNA in the library (data.dna_density, from the QC table): reads on every gene in proportion to its length.
    # Samples above `max_dna_fraction` are kept out of the model when it is built (qc_table.py mark-model) and flagged by `embed`.
    # The two corrections are off: on the atlas they gave no gain and the input correction harms shallow libraries.
    max_dna_fraction: float = 0.10
    # the depth gate of the atlas (qc_table.py mark-model --min-cds-reads): no sample with fewer reads on CDS features was
    # trained on, and `embed` flags samples below it (`low_depth`). Below it the calls are not reliable: exponential
    # libraries of GSE67402 thinned to 200 k CDS reads, and its total-RNA libraries (~100 k), were called stationary
    # (`work/qc_module/depth_checks.txt`); deep atlas samples thinned to 200 k mostly kept their call.
    min_cds_reads: int = 200_000
    covariates: tuple = COVARIATES   # columns of data.covariate_frame given to the residual encoder and the decoder
    # phase-order term of the phase block (stage 1; 0 = off): a pairwise ranking loss on the phase score over pairs of
    # samples whose order along the growth curve is known, inside one study (labels.ORDER_LEVEL) or one culture series
    # (`train --time-series`, atlas_data/time_courses.tsv). Every (source, study) counts the same.
    phase_order_weight: float = 0.0
    phase_order_batch: int = 256     # pairs drawn per step of the phase block
    dna_decoder: bool = False        # the count mean is size factor x RNA rate + expected DNA reads
    dna_inputs: bool = False         # the encoder inputs and size factors are made from the counts minus the expected DNA reads
    # mechanistic GC bias in the decoder (models.GCTerm): the library's GC fit at every feature's GC content, one learned scale;
    # gc_slope and gc_curv then leave the linear covariate term. feature_gc: table of the features' GC content (default: the
    # package table for the inGEST counting annotation, technical.load_feature_gc)
    gc_offset: bool = False
    # learned: the scale is fitted (it settled at 0.62 on the atlas, the rest of the GC effect going into z: `work/batch_correction/`);
    # fixed: held at 1, the gene-level value of the measured fit, so the decoder cannot split GC between the term and z
    gc_scale: str = "learned"
    feature_gc: str | None = None
    # what the encoders see (default since 2026-11-22, user decision; `work/qc_module/NOTES.md`): clean_inputs takes each
    # sample's measured GC offset (technical.gc_offset) off the counts the encoders see; encoder_covariates False keeps
    # the covariates out of the residual encoder, which then sees expression only. Together they took GC out of z (R2
    # 0.24 -> 0.03 within replicates, 0.33 -> 0.02 over all samples) with TEST NLL 6.206 against 6.217. The decoder keeps
    # its linear covariate term on all covariates. qc_module (not the default) adds the decoder's GC term at fixed scale
    # and the genomic-DNA term (``apply_qc_module``); it kept GC in z across samples (0.33)
    clean_inputs: bool = True
    encoder_covariates: bool = False
    # with the GC term or the cleaned inputs: size factors (and the size-factor reference) from the GC-cleaned counts, so
    # that the GC part cannot shift a library's size factor (a mean-centred offset against a median-of-ratios size factor
    # leaves a per-sample constant, SD 0.04, correlated 0.62 with gc_slope, that only z could fit). False in bundles
    # made before 2026-11-22
    gc_size_factors: bool = True
    # genomic-DNA penalty on the residual block (default since 2026-11-22, user decision; 0 = off; `work/qc_module/`):
    # each step also encodes the samples of dna_penalty_groups biological replicate groups (metadata_v2
    # `replicate_group`) of the training rows and adds dna_penalty_w x the in-sample R2 of the regression of their DNA
    # estimate on the residual block, both centred within the groups. Replicates share their biology, so what the
    # residual holds of DNA between them is technical. On the atlas: z -> DNA within replicates 0.38 -> -0.03, decoded
    # treatment contrasts kept (r 0.80), kNN phase / medium / lab strain 0.74 / 0.86 / 0.73 -> 0.78 / 0.89 / 0.86. Without
    # replicate groups or DNA estimates in the data the term is left out (manifest `dna_penalty`)
    dna_penalty_w: float = 1.0
    dna_penalty_groups: int = 256
    qc_module: bool = False
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
        if self.gc_scale not in GC_SCALES:
            raise ValueError(f"gc_scale must be one of {GC_SCALES}, not {self.gc_scale!r}")
        self.apply_qc_module()

    def apply_qc_module(self) -> None:
        """With ``qc_module``: the decoder's GC term at fixed scale, its genomic-DNA term, GC-cleaned encoder
        inputs and an encoder without covariates."""
        if self.qc_module:
            self.gc_offset, self.gc_scale, self.dna_decoder = True, "fixed", True
            self.clean_inputs, self.encoder_covariates = True, False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["withheld_studies"], d["test_studies"] = sorted(self.withheld_studies), sorted(self.test_studies)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known and k != "hparams"}
        for k in ("withheld_studies", "test_studies", "covariates"):
            if k in kw:
                kw[k] = tuple(kw[k])
        hp_known = {f.name for f in fields(HParams)}
        return cls(hparams=HParams(**{k: v for k, v in (d.get("hparams") or {}).items() if k in hp_known}), **kw)
