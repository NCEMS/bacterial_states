# Atlas model v2: design

Status: design of 2026-11-15. A first stage is implemented in `ecoli_atlas` (2026-11-19, see
`ecoli_atlas/README.md`): feature groups (section 1), normalization (2), the rRNA / tRNA and QC
covariates (3, 5), the jointly trained model with its two heads and alignment terms (6) and part of
the evaluation (8). Not implemented yet: missing-gene inference (4), the DNA term (5a), the
corrected and denoised counts (7), the deletion and twin checks of section 8. The model is trained
on counts from the revised inGEST (`docs/counting.md`); the QC the recount saves for it is listed
in `docs/atlas_qc_inputs.md`.


> **As implemented (2026-11-20).** To keep the embedding comparable with the first atlas model, the
> implementation keeps that model's block layout and two-stage fit instead of the single jointly
> trained embedding described below: supervised blocks for growth phase (6), medium type (12) and
> lab strain (4), each fitted on its labelled samples, then a residual block (24) fitted with the
> count decoder on all training samples while the blocks are frozen. This brings back a lab-strain
> call (decisions 3 and 4 below are superseded on that point). TEST and the folds hold reference
> samples only, as before. The count likelihood, feature groups and covariates are as described here.

## Goals

1. The embedding places samples in similar physiological states next to each other, whatever
   study, library protocol or sequencing run they come from.
2. A decoder returns batch-corrected counts.
3. Two supervised calls: growth phase (exponential or stationary) and medium type (complex or
   defined). The v1 `is_lab_strain` call is dropped.
4. No strain label enters the model. Genes a strain does not have are inferred from the counts.
   Regulatory differences between strains stay in the embedding.

## What v1 does and why it changes

Numbers are from the v1 training data (`atlas_data/`: 3,393 metadata rows, of which 3,306 samples
in 159 studies have counts; label counts refer to the metadata rows) and the v1 code
(`ecoli_atlas/src/ecoli_atlas/`).

| v1 | Consequence | v2 |
|---|---|---|
| Input is a per-gene z-score of linear CPM (`preprocess.py`) | A few induced samples set each gene's sd; repression is squeezed between zero and the mean; samples outside the 926 fit samples get extreme values | z-score of log values |
| CPM denominator keeps ssrA (5% of counts at the median, up to 38%) | Every gene shifts when one RNA does | Size factor from CDS and pseudogenes, median of ratios |
| Decoder reconstructs z-scored CPM with MSE (`losses.py`) | Cannot return counts; low counts in shallow samples weigh as much as high counts | Negative binomial on raw counts |
| A gene missing from the strain is a zero, i.e. a strongly negative input | 773 samples lack 200 or more MG1655 genes; K-12 versus other strains is the second expression PC | Missing genes inferred and masked |
| No QC or protocol information is used | Study explains 50–90% of the variance of every leading PC; growth phase 7% or less | QC covariates go to encoder and decoder |
| Three classifier encoders trained separately, frozen, joined to a residual autoencoder (46 dims) | Reconstruction and alignment losses shape only the residual block | One jointly trained model |
| `early_stationary` (229) counted as stationary | Transitional samples blur the phase call | Clean classes only |

## 1. Feature groups

Groups follow the `Type`, `Source` and `Rfam` columns of
`GCF_000005845.2_ASM584v2_counting_v2.features.tsv`.

| Group | Features | Encoder input | Size factor | Decoder and loss |
|---|---|---|---|---|
| mRNA | CDS (4,290), pseudo (145) | yes | yes | yes |
| other RNA | NCBI ncRNA (108) and Rfam hits (127), including ssrA, rnpB, ssrS, ffs | yes | no | yes |
| depleted | rRNA (22), tRNA (86) | no | no | no; summarized as covariates |

Only rRNA and tRNA are removed as inputs, because those are the classes that depletion and size
selection change by orders of magnitude (rRNA is 1.4% of v1 counts at the median and above 50% in
630 samples). The other RNAs are inputs like any gene. They are left out of the size factor only,
so that one abundant RNA cannot move the scale of every gene.

**Check before training.** For each non-mRNA input feature, compute the between-study share of
variance of its log ratio to mRNA reads and compare it with mRNA genes of similar abundance.
Features that behave like rRNA are moved to the depleted group. RNAs shorter than about 200 nt are
the likely candidates, since size selection removes them in some protocols. v1 cannot answer this:
it contains ssrA only, and the ssrA shares computed so far are shares of all counts, so they are
mixed with the rRNA share.

## 2. Normalization

- **Size factor `s_i`.** Median over genes of `count_gi / ref_g`, where `ref_g` is the geometric
  mean of gene `g` over the training samples. Only mRNA features that are called present (section
  4) and have a nonzero count are used. The reference profile is stored in the model bundle, so a
  new sample is normalized without reference to the other samples in its table.
- **Encoder input.** `log2(count_gi / s_i + 1)`, then a per-gene z-score with the training mean
  and sd of the log values (sd floored). A masked gene is set to 0, the gene's mean.
- **Decoder target.** The raw count, with `log s_i` as an offset.
- **Depth.** The log of the number of mRNA reads is one of the covariates (section 5), so the
  encoder knows how noisy the low values of a shallow sample are.

Raw counts are not the encoder input. Depth differs about 100-fold between samples (0.8 to 91
million counts between the 1st and 99th percentile) and 74% of that variance is between studies,
so an encoder given raw counts would have to learn the division itself and could use depth to
recognize the study. Gene length needs no correction because every gene has its own intercept.

## 3. rRNA and tRNA

Their reads are not model features. Each sample gets four covariates: the log ratio of 16S, 23S,
5S and tRNA reads to mRNA reads. 5S and tRNA are separate from 16S/23S because depletion probes
and size selection act on them differently (in v1 the tRNA share correlates with the third
expression PC, Spearman −0.55).

## 4. Genes missing from a strain

Counting is against MG1655, so a gene that a strain lacks gives a zero that means "no gene", not
"gene off". v1 examples: BW25113 has no `araA`, `araB`, `rhaA`, `rhaB` reads; DH10B none for
`fimH`, `leuD`, `mrr`; C strains none for `ompT`, `gatZ`; clinical and ECOR isolates lack the e14,
CP4-6 and CP4-57 prophages.

The calls are made from the counts alone:

1. **Expected count if the gene is present:** `e_gi = s_i · q_g`, with `q_g` the 5th percentile of
   the normalized expression of gene `g` among training samples.
2. **Evidence:** log-likelihood ratio of the observed count under a negative binomial with mean
   `e_gi`, against a low background rate that allows for a few mis-mapped reads.
3. **Smoothing along the genome:** a two-state hidden Markov model over genes in chromosome
   order. Deleted islands are contiguous, so a weakly expressed gene inside one is called with its
   neighbours. An isolated single gene is called only when `e_gi` is large (gene knockouts).
4. **Result:** `p_absent` for every gene and sample. The weight `1 − p_absent` is applied when the
   size factor is computed, to the encoder input (masked genes are set to the mean) and to the
   reconstruction loss. A shallow sample gives little evidence and is left unmasked.
5. **Refinement:** after training, the decoder's predicted mean replaces `q_g` and the calls are
   made once more.
6. **Coverage evidence:** when the stranded coverage track from the recount is available
   (`docs/atlas_qc_inputs.md`, item 6), zero coverage on both strands across a gene is added to
   the evidence. This separates a deleted gene from a silent one better than the sense count alone.

During training, random contiguous blocks of genes are masked in addition, so that the embedding
does not depend on which genes happen to be missing. `ecoli_atlas embed` writes the calls
(`absent_genes.tsv`).

Strain metadata is used only to check the calls against the known deletions above. Genes that a
strain has and MG1655 lacks are not counted and are outside this model.

## 5. QC covariates

A vector `c_i` of about 25 values per sample, standardized on the training set. A value that
cannot exist for a sample (insert size of a single-end library) is set to 0 and flagged in a
companion indicator.

| Group | Covariates |
|---|---|
| Depth | log mRNA reads |
| Reads | trimmed read-length quantiles (10, 50, 90%) per mate, sequencing cycles, share of reads dropped as too short |
| Library | paired or single, fragment-size quantiles, strand fraction |
| Composition | log ratios of 16S, 23S, 5S, tRNA to mRNA |
| DNA contamination | log estimated DNA fraction, strand-leakage slope, intergenic share (section 5a) |
| Bias | duplication rate, gene-body 5′/3′ slope, GC content, share of reads in features under 200 nt |
| Other | share of reads from other genera, instrument family |

Not inputs: study identity, strain, and quantities that mostly reflect strain (mismatch rate
against MG1655, number of genes called absent). They are saved in the QC table but kept away from
the decoder, which could otherwise explain away regulatory differences between strains.

## 5a. DNA contamination

Genomic DNA left in a library gives reads spread evenly along the genome on both strands. It adds
counts to every gene in proportion to the gene's length and independent of its expression, so it
raises weakly expressed genes most and differs between studies with the DNase treatment used.

DNA is seen as reads between genes and reads antisense to genes. Both have other sources as well:
unannotated transcripts and untranslated regions between genes, real antisense RNA, and incomplete
strand marking in the library protocol, which gives antisense reads in proportion to a gene's
sense expression. The estimate therefore separates an even background from an
expression-dependent one:

| Value | Definition | Reflects |
|---|---|---|
| `intergenic_share` | reads in no feature / (assigned + in no feature) | DNA and unannotated transcription |
| `antisense_share` | antisense assigned / (sense + antisense assigned) | DNA, strand leakage, antisense RNA |
| `a` (floor) | intercept of a robust fit over genes of `antisense_g / L_g = a + b · sense_g / L_g` | even background: DNA |
| `b` (leakage) | slope of the same fit | antisense that scales with expression: strand leakage |
| `dna_fraction_est` | `2 · a · genome length / aligned reads` | share of the library that is DNA |

`L_g` is the gene length. The coverage track gives two independent checks of `a`: the low quantile
of bin depth, and the median depth of bins between genes, each per strand (DNA gives the same
value on both strands).

Use in the model:

- `log dna_fraction_est`, `b` and `intergenic_share` are covariates in `c_i`.
- The effect is additive, not a fold change, so the decoder mean carries it explicitly:
  `mean_gi = s_i · rate_gi + d_i · L_g`, where `d_i` is the sample's DNA read density per base and
  strand taken from the QC (a single learned scale factor is allowed). Batch-corrected counts are
  computed without this term.
- The missing-gene calls of section 4 are unaffected: a gene the strain lacks gets no DNA reads
  either.

These definitions have not been tried on data; the fit and the two cross-checks have to agree on
samples of known DNA content (see the spike-in twins in `docs/atlas_qc_inputs.md`) before
`dna_fraction_est` is used as an input.

## 6. Architecture

One model, trained jointly.

- **Encoder:** MLP of the masked, z-scored log input and `c_i`, giving `z_i` of about 32
  dimensions (Gaussian, small KL weight).
- **Decoder:** `log rate_gi = a_g + f_g(z_i) + B_g · c_i`, count ~ NB(`s_i · rate_gi + d_i · L_g`,
  dispersion per gene); the second term is the DNA background of section 5a. `f` is an MLP; `B` is a low-rank gene × covariate matrix. The biological and
  technical terms add in log space, which is what makes the correction in section 7 well defined.
- **Heads on `z`:** two linear classifiers with class-weighted cross-entropy; samples without a
  label do not contribute.
  - Growth phase: `exponential` (1,742 in v1) against `stationary` (234). `lag`,
    `early_exponential` and `transition` (which includes late exponential) carry no label in training and serve
    as a check: they should fall between the two classes. `late_stationary` carries no label
    either and is not a check: it is left out of every phase evaluation (decided 2026-11-19).
  - Medium type: complex against defined (`medium_type`, 1,117 and 1,080 in v1).
- **Alignment terms**, as in v1 (`losses.py`, `train_residual.py`): samples of matching control
  conditions from different studies are pulled together, samples of different conditions within a
  study are pushed apart, and batches are built from whole studies.
- **Output of `embed`:** `z`, the two head probabilities, the absent-gene calls. The per-factor
  embedding blocks of v1 no longer exist.

Two additions are held back until the diagnostics in section 8 ask for them:

1. A predictor of the purely technical covariates (read length, layout, depletion ratios, depth)
   from `z`, trained with gradient reversal, if `z` still carries them.
2. A per-study term in the decoder, shrunk towards zero and dropped at random during training, if
   studies still separate. It comes last because it cannot be estimated for a study the model has
   not seen, and because it can absorb the biology of a study that has only one condition.

## 7. Batch-corrected counts

Reference covariates `c*` are the training medians; the reference depth is fixed (10 million mRNA
reads).

| File | Value |
|---|---|
| `corrected_counts.tsv` | observed count × rate(`z_i`, `c*`) / rate(`z_i`, `c_i`), rescaled to the reference depth |
| `denoised_counts.tsv` | decoder mean at `c*` and the reference depth |

The first keeps what is specific to the sample and was not reconstructed; the second is the
model's own estimate. Genes called absent are written as missing values in both.

## 8. Evaluation

Everything is measured on studies held out from training (`splits.py`, `metrics.py`).

| Question | Measure |
|---|---|
| Do similar samples from different studies meet? | k-nearest-neighbour label agreement for phase and medium with same-study neighbours excluded |
| Is study still visible? | share of neighbours from the sample's own study against chance (`same_study_rate`) |
| Is within-study biology kept? | correlation of exponential-versus-stationary log fold changes before and after correction, in the 9 studies that contain both phases |
| Does `z` carry protocol information? | R² of predicting each covariate from `z` |
| Are missing genes called correctly? | the known deletions of section 4 |
| Does the decoder fit? | negative binomial deviance |
| Do read length, layout and depth move a sample? | distance between technical twins (`docs/atlas_qc_inputs.md`) |

## Limits known in advance

- Medium type is nearly constant within a study (3 of 115 labelled studies contain both types),
  and only 9 studies contain both growth phases. A protocol effect that coincides with a
  condition across all its studies cannot be separated from the condition.
- 33% of the v1 metadata rows have no growth phase label and 35% no medium type.
- The check of section 1 and all covariates of section 5 depend on the recount; nothing in this
  document has been trained or measured yet.
