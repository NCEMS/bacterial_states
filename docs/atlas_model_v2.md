# Atlas model v2: design

Status: design of 2026-11-15. A first stage is implemented in `ecoli_atlas` (2026-11-19, see
`ecoli_atlas/README.md`): feature groups (section 1), normalization (2), the rRNA / tRNA and QC
covariates (3, 5), the jointly trained model with its two heads and alignment terms (6) and part of
the evaluation (8). Missing genes (4) are called and masked in a simpler form than designed (see the
note in that section). The DNA term (5a) was built, tried and left off; libraries above 10% DNA are left out instead (note there). Not implemented yet: the
corrected and denoised counts (7), the deletion and twin checks of section 8. The model is trained
on counts from the revised inGEST (`docs/counting.md`); the QC the recount saves for it is listed
in `docs/atlas_qc_inputs.md`.


> **As implemented.** Instead of the single jointly trained embedding described below, the model has
> a block layout and a two-stage fit: one supervised block per call (growth phase, medium type and
> lab strain, the laboratory lineages K-12, B, C and W against all other known strain groups; one
> dimension each since 2026-11-21), each fitted on its labelled samples, then a residual block (24)
> fitted with the count decoder on all training samples while the blocks are frozen. There is a
> lab-strain call (goals 3 and 4 below are superseded on that point). TEST and the folds hold
> reference samples only. The count likelihood, feature groups and covariates are as described here.

## Goals

1. The embedding places samples in similar physiological states next to each other, whatever
   study, library protocol or sequencing run they come from.
2. A decoder returns batch-corrected counts.
3. Two supervised calls: growth phase (exponential or stationary) and medium type (complex or
   defined).
4. No strain label enters the model. Genes a strain does not have are inferred from the counts.
   Regulatory differences between strains stay in the embedding.

## Design choices and why

| Choice | Why |
|---|---|
| Input is a per-gene z-score of log values | A z-score of linear CPM lets a few induced samples set each gene's sd and squeezes repression between zero and the mean |
| Size factor from CDS and pseudogenes, median of ratios | A CPM denominator that keeps ssrA (5% of counts at the median, up to 38%) shifts every gene when one RNA does |
| Negative binomial on raw counts | A decoder of z-scored CPM with MSE cannot return counts, and low counts in shallow samples would weigh as much as high counts |
| Genes missing from a strain inferred and masked | A missing gene would otherwise be a zero, a strongly negative input; hundreds of samples lack 200 or more MG1655 genes |
| QC covariates | Study explains 50–90% of the variance of every leading expression PC, growth phase 7% or less |
| Clean phase classes only | Transitional samples counted as stationary blur the phase call |

## 1. Feature groups

Groups follow the `Type`, `Source` and `Rfam` columns of
`GCF_000005845.2_ASM584v2_counting_v2.features.tsv`.

| Group | Features | Encoder input | Size factor | Decoder and loss |
|---|---|---|---|---|
| mRNA | CDS (4,290), pseudo (145) | yes | yes | yes |
| other RNA | NCBI ncRNA (108) and Rfam hits (127), including ssrA, rnpB, ssrS, ffs | yes | no | yes |
| depleted | rRNA (22), tRNA (86) | no | no | no; summarized as covariates |

Only rRNA and tRNA are removed as inputs, because those are the classes that depletion and size
selection change by orders of magnitude (rRNA is a small share of counts at the median and above 50% in
hundreds of samples). The other RNAs are inputs like any gene. They are left out of the size factor only,
so that one abundant RNA cannot move the scale of every gene.

**Check before training.** For each non-mRNA input feature, compute the between-study share of
variance of its log ratio to mRNA reads and compare it with mRNA genes of similar abundance.
Features that behave like rRNA are moved to the depleted group. RNAs shorter than about 200 nt are
the likely candidates, since size selection removes them in some protocols.

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
and size selection act on them differently (the tRNA share correlates with a leading expression PC).

## 4. Genes missing from a strain

Counting is against MG1655, so a gene that a strain lacks gives a zero that means "no gene", not
"gene off". Examples: BW25113 has no `araA`, `araB`, `rhaA`, `rhaB` reads; DH10B none for
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

**As implemented (2026-11-20, `preprocess.missing_genes`, on by default, `train --no-mask-missing`
switches it off).** A rule in place of the likelihood ratio and the hidden Markov model:

- The low end `q_g` is the 25th percentile of the normalized count over the reference training
  samples in which the gene is not called missing (found in two rounds).
- A gene is depleted in a sample when its count is at most 2% of `s_i · q_g`. Neighbouring depleted
  genes in chromosome order form a run; the run is called missing when its genes together expected
  at least 10 reads.
- Missing genes are left out of the size factor, of the gene means and standard deviations and of
  the count likelihood; the encoders see 0 (the gene's mean) for them.
- In place of random blocks, half of the samples of every training batch also get the missing genes
  of another training sample, as input only: their counts are still reconstructed.
- `embed` makes the same calls with the low end saved in the bundle and writes `missing_genes.tsv`;
  `predictions.tsv` of a bundle has `n_missing_genes`.
- With the antisense count table (`train --antisense`, `embed --antisense`) a depleted gene that has
  more than 2 antisense reads, and more than a quarter of the sample's antisense background for a
  gene of its length, is present and not called (`preprocess.background_rate`). Antisense reads only
  remove calls: adding "no antisense reads" as evidence for a deletion was tried and calls the
  silenced gene clusters of K-12 (`yddJKL`, `ypjBC`, `ygeF`-`ygeN`), which have no reads on either strand.
- Pooled calls (`call_groups`, `pooled_low_profile`, `group_calls`; on by default, `--no-pool`): within
  a study, samples whose calls overlap (intersection over union at least 0.5; samples with fewer
  than 20 calls together) form a group, found without strain metadata. The group's counts are summed
  and called like one deep sample against a low end made from pooled groups, and every sample gets
  its own calls and its group's. Weakly expressed genes of a missing island, which give a single
  sample too little evidence, are so found: of deletions put into MG1655 studies 99% are called
  (79% per sample), with a median of 2 calls in K-12 samples as before.
- Not done: the refinement with the decoder's mean (5) and the coverage tracks (6).

On the atlas the calls give a median of 3 genes in K-12 samples (mostly known deletions: `araBA`
and `rhaBA` of BW25113, the `lac` region) and about 280 to 480 in the larger other strain groups. They find about
80% of what calls pooled over a whole study and strain find, and 80% of them are confirmed by
those.

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

As implemented, the log substitution rate against MG1655 is a covariate as well (2026-11-20): among
K-12 samples it spans a wider range (whole studies at 2 to 4%) than the medians of the strain
groups differ (0.27% for K-12, 0.30 to 0.59% for the others), so it describes the sequencing run.

**Revised 2026-11-21 (user decision): technical covariates only.** The list above was the
design; the model now takes 15 (`config.COVARIATES`): layout, fragment-size quantiles, read-1 length
quantiles, strand purity, log substitution rate, GC slope and curvature, instrument family. The composition ratios,
duplication rate, gene-body slope, GC content and short-feature share are left out because they
move with the cells' state: within a study, stationary samples differ from exponential ones by
0.8 to 1.5 SD in each of them (`work/qc_blocks/covariate_biology.tsv`). Depth is carried by the
size factor, the DNA estimate by the 10% gate. Leakage, intergenic share, trimming shares, cycles,
mate-2 length and other genus were dropped as redundant or not about the library. The permutation and
forward-selection analyses behind the choice are in `work/qc_blocks/NOTES.md`. The GC bias replaces
the read GC share. It is the within-gene fit of the QC record (inGEST `gc_bias.py`; for the atlas
recount, whose records predate it, computed from the coverage tracks by `qc_table.py add-full
--genome`). It shifts by 0.17 SD with phase (read GC share 0.79), has split-half reliability 0.99,
and predicts the gene-level GC trend of the counts within studies (factor 1.05, R2 0.39;
`work/gc_bias/NOTES.md`).

New samples get the same covariates from their inGEST QC records (`ecoli_atlas/qc_record.py`, the
formulas of `qc_metrics.py` repeated so that the packages stay apart; identical to `sample_qc.tsv`
for all 7,731 atlas samples, checked 2026-11-21 with `work/qc_parity/record_parity.py`). `embed`
refuses samples without them (`data.check_qc`: the values every atlas sample of its layout has)
except the Centrifuge value, which a run with `centrifuge: false` lacks and which is then set to the
training mean.

Not inputs: study identity, strain, and quantities that mostly reflect strain (share of reads off
the MG1655 reference, number of genes called absent). They are saved in the QC table but kept away from
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

These definitions were then tried on data (`docs/atlas_qc_inputs.md`, calibration): the estimate
used is the coverage-based `dna_fraction_est`.

**As built (2026-11-21; `data.dna_density`, `models.count_log_mean`), and what was decided.**

- DNA fragments per base and strand of a sample: `dna_fraction_est` × aligned fragments /
  (2 × genome length), with the aligned fragments taken from the QC table (reads kept × aligned
  share, halved for paired-end). A feature of length `L` expects that density × (`L` + median
  fragment size, or read length for single-end) DNA reads.
- Decoder: `mean = s · rate + k · DNA reads`, with one learned factor `k` (`train --no-dna-decoder`
  switches the term off).
- Encoder inputs and size factors are made from the counts minus the expected DNA reads, because
  the block encoders see no covariates and cannot correct for DNA themselves (`--no-dna-inputs`).
- Samples with an estimate of 0 (below the detection limit; 1,620 of 7,769) get no correction.
- Check on the spiked twins: expected over added reads 0.93–1.04 at 5 and 20% DNA (0.80–0.96 at
  1%); after the input correction 0–6% of the added reads are left (up to 23% over-removed at 1%).
  Libraries below the limit (250,000 pairs with 1 or 5% DNA) are not corrected.
- DNA is taken as even along the chromosome. In the 127 atlas samples above 5% DNA the background
  near the replication origin is 1.2 times that near the terminus (median; 0.9–1.5 for 80% of
  them), so a position term would change the DNA reads by about ±10% and is left out.
- **Outcome.** Two 8-seed sweeps against the model without the term (`outputs/sweeps/models_v2_dna`,
  `outputs/sweeps/models_v2_dna_decoder`): the decoder term alone left the calls unchanged and the count
  fit slightly worse, with a learned factor of 0.31 instead of 1. With the input correction, twins
  with 5 and 20% DNA at normal depth landed near their DNA-free parent (strain block 42 → 2
  replicate distances at 20%), but shallow libraries moved much further away (250,000 pairs with
  20% DNA: P(stationary) changed by 0.66 instead of 0.11), and the atlas showed no net gain (phase
  TEST AUROC 0.952 → 0.934).
- **Decision (user).** Both corrections are off by default. Samples with `dna_fraction_est` above
  10% are left out of the model (38 of 7,769) and `embed` warns for such samples.

## 6. Architecture

One model, trained jointly.

- **Encoder:** MLP of the masked, z-scored log input and `c_i`, giving `z_i` of about 32
  dimensions (Gaussian, small KL weight).
- **Decoder:** `log rate_gi = a_g + f_g(z_i) + B_g · c_i`, count ~ NB(`s_i · rate_gi + d_i · L_g`,
  dispersion per gene); the second term is the DNA background of section 5a. `f` is an MLP; `B` is a low-rank gene × covariate matrix. The biological and
  technical terms add in log space, which is what makes the correction in section 7 well defined.
- **Heads on `z`:** two linear classifiers with class-weighted cross-entropy; samples without a
  label do not contribute.
  - Growth phase: `exponential` against `stationary`. `lag`,
    `early_exponential` and `transition` (which includes late exponential) carry no label in training and serve
    as a check: they should fall between the two classes. `late_stationary` carries no label
    either and is not a check: it is left out of every phase evaluation (decided 2026-11-19).
  - Phase-order term (2026-11-21, `train --phase-order-weight`, off by default): the two-class head
    left the phase block effectively one-dimensional and saturated within each class, while the
    growth-curve movement of the time courses sat in the residual block (`work/phase_dims/NOTES.md`).
    The term is a pairwise ranking loss on the phase score over pairs whose order along the growth
    curve is known: phase levels inside a study (early exponential < mid exponential < transition
    < stationary) and time points inside a culture series (`atlas_data/time_courses.tsv`, 10
    studies, 43 series; GSE224463, GSE67402 and GSE226643 held out). It never compares studies.
    Scored by `series_order` / `cv_series_order` (within-series time order of the phase score).
  - Medium type: rich against minimal (`medium_type`).
- **Alignment terms** (`losses.py`): samples of matching control
  conditions from different studies are pulled together, samples of different conditions within a
  study are pushed apart, and batches are built from whole studies.
- **Output of `embed`:** `z`, the head probabilities, the absent-gene calls.

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
- The check of section 1 and all covariates of section 5 depend on the recount; nothing in this
  document has been trained or measured yet.
