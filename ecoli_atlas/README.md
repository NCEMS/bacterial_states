# Ecoli_ATLAS

Embeds *E. coli* RNA-seq samples with an encoder–decoder trained on thousands of public samples
counted with Ecoli_inGEST, and places new samples in that embedding.

The embedding has one block per call and a residual block, 27 dimensions, and is fitted in two
stages (count model and covariates: [`docs/atlas_model_v2.md`](../docs/atlas_model_v2.md)):

| Block | Dimensions | Fitted on |
|---|---|---|
| growth phase (exponential / stationary) | 1 | training samples with a phase label |
| medium type (rich / minimal) | 1 | training samples with a medium label |
| lab strain (`lab_strain`: lab / non-lab) | 1 | training samples with a known strain group |
| residual | 24 | all training samples, blocks frozen |

Sweeps with call blocks of 6 / 12 / 4 dimensions against one dimension each found that one dimension
loses nothing: the same calls, the same neighbours inside each block, the residual no richer; the decoder's TEST
NLL rose by about one seed SD (`work/phase_dims/NOTES.md`). `train --block-dims` changes the sizes
(e.g. `phase=6,medium=12,lab_strain=4`, `residual=16`; blocks not named keep theirs). The bundle records them (`embedding_blocks` in `manifest.json`), and `embed`, `evaluate` and
`figures` read them from there.

- **Block encoders:** z-scored log expression → the block; a linear head on the block gives the
  call. Class-weighted cross-entropy, labelled samples only.
- **Residual encoder:** z-scored log expression plus library covariates → the residual block.
- **Decoder:** the whole embedding and the covariates → expected counts, with a negative binomial
  likelihood on the raw counts and the size factor as an offset.
- **Alignment terms** (residual stage, on the whole embedding): samples of one study known to
  differ are kept apart; matching control conditions of different studies are drawn together.

This is the first stage of the design. Not yet in the model: batch-corrected counts.

## Install

```bash
mamba env create -f ecoli_atlas/environment.yaml
conda activate ecoli_atlas
pip install -e "./ecoli_atlas[dev]"
pytest ecoli_atlas/tests
```

The environment has the CPU build of PyTorch, and gseapy (bioconda) for the gene-set enrichment of
`figures` and `drivers --gsea`; with pip it comes with the `figures` extra (`pip install -e "./ecoli_atlas[figures]"`). A single training run takes minutes on 8 CPU threads
(about 3 s per epoch on 6,800 samples); a GPU is worth it for seed sweeps and cross-validation. For
a GPU node:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Without installing, `bin/ecoli_atlas` runs the CLI from the checkout if the dependencies are present.

## Data bundles

```bash
ecoli_atlas download models          # outputs/models: the model, its atlas embedding, all seeds
ecoli_atlas download training-data   # atlas_data: everything needed to retrain the model or check its labels
ecoli_atlas download all --dry-run   # what would be fetched, and where
```

Both are assets of the GitHub release
[`resources-v2`](https://github.com/NCEMS/bacterial_states/releases/tag/resources-v2); each archive's SHA-256 is pinned in
`download.py` and checked before anything is unpacked. `--dest` unpacks elsewhere; by default the target is under
`$BS_DATA_ROOT`, or the current directory.

| Bundle | Files | Size |
|---|---|---|
| `models` | `models/`: the selected seed (`model.pt`, `preprocessing.pt`, `manifest.json`, `splits.csv`, `evaluation.json`), the atlas it defines (`embedding.tsv`, `predictions.tsv`, `cv_predictions.tsv`, `missing_genes.tsv`), and `seed_1` ... `seed_8` (each a full bundle; `drivers` averages over them) | 312 MB download, 378 MB unpacked |
| `training-data` | `atlas_v2_counts.tsv`, `atlas_v2_antisense_counts.tsv` (sense and antisense counts, inGEST feature ids x GSM), `metadata_v2.tsv` (labels per sample, with replicate groups), `sample_qc.tsv` (QC values; `in_model` = the model's samples), `time_courses.tsv` (the phase-order term's series), `atlas_v2_missing_genes.tsv` (genes called missing in each sample, `sample` / `feature`), and the metadata's sources: `metadata_v2_studies.tsv` and `metadata_v2_sample_text.tsv` (GEO text), `metadata_v2_evidence.tsv` (the passage behind each label), `metadata_v2_label_scheme.yaml` | 100 MB download, 301 MB unpacked |

`ecoli_atlas download modules` fetches the curated gene modules for `drivers --modules` from their own sites (they are
not shipped) and builds `atlas_data/modules/modules.tsv`; see [Curated gene modules](#curated-gene-modules).

The counting annotation (`*_counting_v2.features.tsv`, needed by `train`, `figures` and `casestudy`) is in the inGEST
core bundle (`ecoli_ingest download core`). The archives are built by `release/make_release.py` (reproducible: the same
files give the same bytes).

## Inputs

| Table | Made by | Content |
|---|---|---|
| `atlas_v2_counts.tsv` | `ecoli_ingest/atlas_build/full_run/collect_results.py` | features × samples, inGEST feature ids (`CDS_b0002_thrA`), GSM columns |
| `metadata_v2.tsv` | `ecoli_ingest/atlas_build/metadata/apply_labels.py` | labels per sample ([`docs/metadata_v2.md`](../docs/metadata_v2.md)) |
| `sample_qc.tsv` | `ecoli_ingest/atlas_build/prescreen/qc_table.py` | QC values per sample; `in_model` names the samples of the model |
| `*_counting_v2.features.tsv` | the inGEST counting annotation | type, symbol and length of every feature |

**Features.** mRNA features (CDS, pseudogenes) and the other RNAs (ncRNA genes, Rfam hits) are
encoder inputs and decoder outputs. Only mRNA features enter the size factor. rRNA and tRNA are
neither and are not used.

**Covariates** (`config.COVARIATES`, recorded in the manifest as `config.covariates`): technical
values of the library and the sequencing run only. Layout (paired or single), fragment-size
quantiles (paired-end), read-1 length quantiles after trimming, strand purity, substitution rate
against MG1655 (log; a property of the sequencing run more than of the strain), GC bias (slope
and curvature of log coverage against GC inside genes, from the QC record's `gc_bias`, so
expression levels drop out; it replaces the read GC share) and instrument family; 15 in all. A
value that does not exist for a sample (fragment sizes of a single-end
library) is set to the training mean. Left out because they move with the cells' state (0.8 to 1.5
within-study SD between exponential and stationary samples; `work/qc_blocks/`): rRNA, tRNA and
mRNA composition, duplication rate, gene-body slope, GC content, share of reads on short features.
Left out as redundant or handled elsewhere: depth (the size factor), the DNA estimate (samples
above 10% are gated out), DNA leakage, intergenic share, trimming shares, cycles, mate-2 length,
other genus. Study and strain are not covariates. Covariates go to the residual encoder and the
decoder, not to the block encoders. Bundles made before the choice keep the 32 covariates they were
trained with.

**Labels** (`labels.py`).

- Phase: `mid_exponential` → exponential, `stationary` → stationary, only for untreated standard
  RNA-seq samples (`phase2` set). `late_stationary` has no label and is not used in any phase
  evaluation. Lag, early exponential and transition have no label; the head's output on them is
  reported as a check. (With `--phase-head ordinal` transition is a fitted level and late
  stationary counts as stationary; see below.) The phase call is **scored** (TEST, out-of-fold, nearest
  neighbours) only on samples whose phase is stated in the study's text (`growth_phase_basis`
  `stated`); a phase inferred from an OD or a culture time is used for fitting only.
- Medium: `medium_type` (rich / minimal), the two extremes only. Rich is a complex broth (LB,
  Mueller-Hinton, TSB, BHI); minimal is salts with a carbon source (M9, MOPS, M63). Minimal medium
  with a hydrolysate, defined rich media (DMEM, MOPS EZ rich) and natural fluids have no label; the
  head's output on them is reported as a check.
- Lab strain (`lab_strain`): `strain_group` K-12, B, C or W (the laboratory lineages) → `lab`; any other
  known group (pathogens, natural isolates, probiotic strains) → `non_lab`; no label where the group is
  empty. The probability column is `p_lab`. A bundle trained for other calls (`outputs/sweeps/models_v2_k12`
  has a `k12` call, K-12 against the rest) is refused by this version.

**Sample selection.** `qc_table.py mark-model --metadata metadata_v2.tsv --min-cds-reads 200000 --max-dna-fraction 0.10`
marks the samples that are usable by the metadata, were counted by the full inGEST run, have at
least 200,000 reads on CDS features and a genomic-DNA estimate of at most 10%.

## Commands

```bash
T="--counts atlas_data/atlas_v2_counts.tsv --metadata atlas_data/metadata_v2.tsv --qc atlas_data/sample_qc.tsv \
   --features resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"

# the study split only (splits.csv), then the figures of the data
ecoli_atlas train $T --out outputs/models --split-only
ecoli_atlas figures --metadata atlas_data/metadata_v2.tsv --splits outputs/models --out examples/atlas_model_figures/figures

# train on that split and evaluate on the held-out studies
ecoli_atlas train $T --out outputs/models --splits outputs/models/splits.csv [--device cuda]
ecoli_atlas train $T --out outputs/models --splits outputs/models/splits.csv --seeds 1,2,3,4 --cv

# evaluate a bundle again; embed new samples
ecoli_atlas evaluate $T --models outputs/models --out work/evaluation
ecoli_atlas embed --ingest-results my_run --models outputs/models --out embedded     # an inGEST run directory
ecoli_atlas prepare --ingest-results my_run --out prepared [--metadata my_samples.tsv]   # the same tables, to look at or join
ecoli_atlas embed --counts prepared/counts.tsv --qc prepared/qc.tsv --antisense prepared/antisense_counts.tsv \
    --models outputs/models --out embedded

# new samples against the atlas; studies of the atlas against the rest of it
ecoli_atlas compare --embeddings embedded --models outputs/models --out compared \
    [--atlas-metadata atlas_data/metadata_v2.tsv] [--metadata prepared/samples.tsv --group-by time_h --order-by time_h] [--umap]
ecoli_atlas neighbors GSE224463 --models outputs/models --out neighbors [--metadata atlas_data/metadata_v2.tsv] [--umap]

# an interactive 3-D map of the atlas (one HTML file), optionally with your embedded samples
ecoli_atlas view --models outputs/models --metadata atlas_data/metadata_v2.tsv \
    --studies atlas_data/metadata_v2_studies.tsv --sample-text atlas_data/metadata_v2_sample_text.tsv \
    --counts atlas_data/atlas_v2_counts.tsv --qc atlas_data/sample_qc.tsv \
    [--embeddings embedded=my_samples] --out atlas_view.html
```

### Ways of fitting (`train --fit`)

| `--fit` | Embedding (27 dimensions) | Fitting |
|---|---|---|
| `two-stage` (default) | blocks 1 / 1 / 1 and residual 24 | block encoders on their labelled samples, then the residual encoder and the decoder on all training samples with the blocks frozen and standardised |
| `one-pass` | the same blocks and residual | everything together in one loss: the heads' cross-entropy, the count likelihood, the KL term and the two alignment terms. The block encoders also receive the count and alignment gradients; the blocks are not standardised |
| `unstructured` | one embedding without blocks | the encoder sees expression and covariates, the three heads read the whole embedding, all terms together |

The tables, labels, split, cross-validation and evaluation are the same for all three, so runs on
one `splits.csv` can be compared directly. The manifest records `fit` and `embedding_blocks`.

```bash
for FIT in two-stage one-pass unstructured; do
  ecoli_atlas train $T --fit $FIT --out outputs/sweeps/models_${FIT//-/_} --splits outputs/models/splits.csv \
      --seeds 1,2,3,4,5,6,7,8 --cv --device cuda
done
```

### Phase head (`train --phase-head`)

| `--phase-head` | Fitted on | Loss |
|---|---|---|
| `binary` (default) | `mid_exponential` against `stationary` | class-weighted cross-entropy, as the other calls |
| `ordinal` | three ordered levels: exponential (`mid_exponential`) < transition (`transition`) < stationary (`stationary` and `late_stationary`) | proportional odds: one score of the phase block (or of the whole embedding in the unstructured fit) with a threshold between neighbouring levels |

- The call is the same in both: `p_stationary`, with the ordinal head the probability of lying above
  the upper threshold. The scores on TEST and out of fold (exponential against stationary, stated
  phases only) are therefore comparable between the two heads.
- The levels are given to untreated standard RNA-seq samples only (`labels.phase_levels`). Lag,
  early exponential and long-term stationary have no level.
- The label cap applies to each level as it does to each class.
- `predictions.tsv` has `phase_logit`, the log-odds of stationary, which is still graded where
  `p_stationary` is 0 or 1.
- **Within-study order** (`phase_order` in `evaluation.json`; `test_order` and `cv_order` in
  `seed_sweep.tsv`): among pairs of untreated samples of one study at different levels, the share
  whose `phase_logit` is in the order of the levels, as the mean over studies. It is reported for
  both heads, on the TEST studies and study-blind over the folds. Stated and inferred phases count
  alike. The seeds are still ranked on the mean out-of-fold AUROC.

```bash
ecoli_atlas train $T --phase-head ordinal --out outputs/sweeps/models_ordinal --splits outputs/models/splits.csv --seeds 1,2,3,4 --cv
```

### Phase-order term (`train --phase-order-weight`)

The two-class phase head has no reason to order samples within a class, and the phases between the
classes do not train it. The order term adds a pairwise ranking loss on the phase score to the
fit of the phase block (stage 1), `log(1 + exp(-(s_j - s_i)))` for pairs where sample j is
further along the growth curve than sample i:

- **phase levels inside a study:** early exponential < mid exponential < transition < stationary
  (`labels.ORDER_LEVEL`; untreated samples). Lag is left out (lag cells come from a stationary
  inoculum), and late and long-term stationary as everywhere else;
- **time inside a culture series** (`--time-series atlas_data/time_courses.tsv`, built by
  `atlas_build/metadata/time_courses.py`; one series = one culture line followed along its growth
  curve): earlier against later time points, without lag, late and long-term stationary samples.

Pairs are only made within a study or a series, never across studies, and only among the
samples a fit uses (folds included). Each step draws `phase_order_batch` (256) pairs so that every
(source, study) group counts the same. Weight 0 (the default) is the model without the term; then
the table changes nothing. The manifest records the weight and the pairs (`phase_order_term`).
With `--time-series` the evaluation also reports `series_order` (TEST, withheld and training
studies) and, with `--cv`, `cv_series_order`: along every series of 3 or more samples, the share of
time pairs whose phase score is in time order, and the Spearman correlation, as means over
series within a study and then over studies. `seed_sweep.tsv` gets `test_series_order`,
`withheld_series_order` and `cv_series_order`. The held-out growth curves are GSE224463
(withheld), GSE67402 and GSE226643 (TEST).

```bash
ecoli_atlas train $T --time-series atlas_data/time_courses.tsv --phase-order-weight 0.3 --out outputs/sweeps/models_order_w0.3 \
    --splits outputs/models/splits.csv --seeds 1,2,3,4,5,6,7,8 --cv
```

### Split

Whole studies only, and TEST and the folds are made of **reference samples** only, as in the first
atlas model: untreated samples of the wild type or of the baseline genotype a study compares against
(`treated` false and `is_reference_genotype` true; `labels.is_reference`). The stricter choice, unmodified wild type only
(`is_wild_type`), was tried on the same studies (2026-11-21): scored on the same samples the calls are
equivalent, and it leaves 8 of the 44 TEST studies and 75 of the 302 fold studies without a scored sample.

| `splits.csv` | Samples |
|---|---|
| `split` 3 | TEST: the reference samples of the TEST studies (about 10% of all reference samples, chosen to mirror the class proportions of the three labels) |
| `split` 1, `cv_fold` 1–3 | reference samples of the other studies, in three study folds |
| `split` 1, no `cv_fold` | the other samples of those studies: fitted, never scored; in cross-validation they are left out of the fit together with their study's fold |
| `split` 0 | projected only: the withheld study `GSE224463`, and the samples of TEST studies that are not reference samples |

The studies of the examples and case studies (`GSE128997`, `GSE189154`, `GSE271069`, `GSE67402`) are
always TEST (`config.py`). A study with more than the TEST share of the reference samples stays in the training pool.

All statistics (size-factor reference, z-scores, covariate scaling) come from the reference samples
of the training pool.

### Large studies

A study contributes at most `study_cap` samples per epoch, drawn afresh each epoch; the default cap
is the size of the second-largest study. The same weights are used for the training statistics and
the class weights. With the present atlas only GSE206047 (1,072 samples) is affected.

That cap does not stop one study from supplying most of a class: GSE206047 has 300 of the 424
stationary training labels. So the labels of each call have a cap of their own (`label_cap`,
`--label-cap`): within one class, a study counts for at most as many samples as the second-largest
study of that class has (19 for stationary, 249 for exponential). It is applied as a weight on each
sample's label in the call's cross-entropy, in all three fits, and the class weights are made from
the weighted class sizes. The manifest records the caps as `label_caps`.

### `train` output (the bundle)

| File | Content |
|---|---|
| `model.pt`, `preprocessing.pt` | weights; reference profile, z-score and covariate statistics (tensors only: `torch.load(weights_only=True)`) |
| `manifest.json` | features, covariates, `embedding_blocks` (columns of each block), hyper-parameters, split settings, TEST metrics, versions (`schema_version` 2) |
| `splits.csv` | study, split and fold of every sample |
| `evaluation.json` | the report below |
| `embedding.tsv`, `predictions.tsv` | `z` and the head probabilities of every training-table sample |
| `phase_lfc.tsv`, `history.tsv` | per-study fold-change check; loss per epoch of every stage (`stage`: a call, or `residual`) |

With `--cv`, `cv_predictions.tsv` has the out-of-fold scores: every sample of the fold studies
(reference or not), scored by the model that was fitted without its study, in the columns of
`predictions.tsv` plus `cv_fold`, `reference` and `phase_level` (0 exponential, 1 transition,
2 stationary, -1 not on the scale). The out-of-fold metrics use its `reference` rows, the
within-study order all rows.

With `--seeds`, every seed is in `seed_<s>/` and `seed_sweep.tsv` lists their TEST and (with `--cv`)
out-of-fold metrics. One seed is copied to the output directory: the one with the best mean
out-of-fold AUROC with `--cv`, otherwise the first. Seeds are never ranked on TEST.

### Evaluation report

| Key | Meaning |
|---|---|
| `heads` | balanced accuracy (at 0.5) and AUROC of the three calls, on TEST and on the training pool |
| `knn_agreement` | share of a TEST sample's 15 nearest training neighbours with its label, class-balanced |
| `same_study_rate` | share of a sample's nearest neighbours from its own study (lower = better mixing) |
| `covariate_r2` | R² on TEST of a ridge regression from `z` to each covariate (lower = less technical signal) |
| `counts` | negative binomial log-likelihood per feature; correlation of observed and expected log counts |
| `phase_lfc` | studies with both phase classes: correlation of the observed stationary-vs-exponential log fold change with the decoder's at reference covariates |
| `phase_check` | median P(stationary) of untreated samples in the in-between phases |
| `medium_check` | median P(minimal) and share called minimal for the in-between media |
| `series_order` | with `--time-series`: time order of the phase score along culture series (TEST, withheld, training studies; see the phase-order term) |

### `embed`

The model takes every sample's QC values as covariates, so `embed` needs them. With an inGEST run,
`--ingest-results DIR` (the run directory or its results directory) is all it needs: it reads the
counts (`all_samples_raw_counts.txt`), the antisense counts (`all_samples_antisense_counts.txt`)
and the QC record of every counted sample (`<sample>/qc/<sample>.qc.json`), and writes the QC table
it used as `qc.tsv`. Otherwise give `--counts` and `--qc` (the `qc.tsv` of `prepare`, or a table with
the QC columns of `sample_qc.tsv`, as `full_<name>` or `<name>`, indexed by sample), and
`--antisense` when the bundle was trained with it. `prepare --ingest-results` writes the same four tables
(`counts.tsv`, `antisense_counts.tsv`, `qc.tsv`, `samples.tsv`).

Only strand-specific libraries are accepted, by `embed`, `prepare` and `train` alike: a sample with
less than 80% of its reads on one strand (strand purity = the larger of `fraction_forward` and
`fraction_reverse` of its QC record), or without strand fractions, is refused (`data.check_stranded`,
the same bar as inGEST, which counts no other libraries).

`embed` refuses a sample without a QC row, or one that lacks a covariate of its bundle that every
atlas sample has (read-1 lengths, strand purity, substitution rate, GC slope and curvature;
fragment sizes for paired-end libraries). The GC bias is in records made with the reference
genome in the inGEST resources (`annotation/GCF_000005845.2_ASM584v2_genomic.fna`). For an older
run, add the genome and repeat the record step (`ecoli_ingest run -- --forcerun sample_qc`).
A record made with the genome can still lack the GC bias when the library is too shallow for the fit (fewer than 50
genes of at least 600 bp at depth 5). `embed --fill-missing-gc study` embeds such samples with the median GC slope
and curvature of the other samples of their study (`--studies`, else the whole table), `--fill-missing-gc atlas` with
the training mean; `samples.tsv` says which (`gc_filled`) and `embed` warns. Such libraries are usually also below
the depth gate (`low_depth`), outside the depths the model was trained on.

`embed --seeds` also embeds the samples with every `seed_<n>` model of a bundle trained with several seeds:
`seeds/seed_<n>/` (embedding and predictions of each seed) and `predictions_seeds.tsv` (per call: median, min and
max probability over the seeds, `<p>_n_called` = the number of seeds at 0.5 or above, and the median, min and max of
the log-odds). The bundle's own model is one of the seeds (the one chosen on out-of-fold scores), and
`predictions.tsv` stays its call. The spread is how much a call depends on the randomness of training, with the data
and the split fixed; it does not include the sampling noise of the sample's counts, and every seed can agree on a
wrong call for a sample unlike the training data.
Centrifuge is not needed. For a bundle made before the covariate choice, the older list
applies, and a run made with `centrifuge: false` gets the atlas mean for the other-genus value, with
a warning. The
QC table made from the records gives the same covariates as `sample_qc.tsv` for all 7,731 atlas
samples, and embedding a recount batch this way gives the training embedding to 1e-5.

Feature ids must be those of the counting annotation (fails below `--min-feature-coverage`, default
0.9). Output: `embedding.tsv`, `predictions.tsv`, `samples.tsv` (size factor, reads on mRNA
features, number of genes called missing, DNA estimate), `covariates.tsv` (the composition and QC covariates
before scaling, as the model sees them), `missing_genes.tsv` (the genes called
missing from a sample's strain, see below) and `feature_coverage.tsv`.

### Corrected and decoded counts (`embed --corrected-counts`, `--decoded-counts`)

Both are tables features x samples, each sample scaled to its own reads on mRNA features, with
the technical part of the library swapped for that of a reference library: paired-end, the atlas's
mean technical values and no GC bias (`technical.py`).

- `corrected_counts.tsv`: the **observed** counts with (1) the decoder's covariate term moved from
  the sample's covariates to the reference (it is additive in the log rate, so this changes exactly
  what the model learned the covariates do) and (2) the library's GC bias taken off by its own fit
  (inGEST `gc_bias`: slope and curvature of log coverage on GC inside genes) at every feature's GC
  content. The GC covariates stay at the sample's values in (1), so GC is not taken off twice. Every
  sample keeps its own detail.
- `decoded_counts.tsv`: the decoder's expected counts at the sample's embedding, with the same swap.
  A denoised summary, not corrected counts: on the atlas they keep within-study stationary against
  exponential fold changes at r 0.88 against a noise ceiling of about 0.98, and 0.71 on the TEST
  study. Genes called missing from a strain are 0 here; in the corrected counts they keep their
  observed count. Genomic-DNA reads are not in the decoder's rate, so they are not in these counts.

What is corrected: the measured technical covariates, and fully the GC bias (on the atlas the
per-sample GC trend falls from r 0.61 against the library's GC slope to 0.02 with the offset alone).
What is not: batch effects the covariates do not describe. Within one strain and condition the
between-study share of the variance is still about four times its no-study-effect value, and the
covariates remove 0.01-0.02 of it (`work/batch_correction/`). Within-study contrasts are kept
(r 0.999 against the observed counts).

`train --gc-offset` puts the same GC offset into the decoder (`models.GCTerm`): each library's GC
fit at every feature's GC content, with one learned scale (starting at 1; the manifest reports it
under `gc_term`). `gc_slope` and `gc_curv` then leave the linear covariate term, so the decoder
explains GC once; the residual encoder still sees them. Corrected counts from such a bundle use the
decoder's own term and need no feature table. With a learned scale the atlas fit settled at 0.62 and
the embedding took the rest of the GC effect (its GC-slope R2 rose from 0.33 to 0.57): the split
between the term and z is not identified. `--gc-scale fixed` holds the scale at 1, the gene-level
value of the measured fit (subtracting it leaves no GC trend in the observed counts). On the atlas it left less GC in the decoded
counts (trend 0.13 against 0.42) but z still carried more GC than with the plain linear term,
so neither variant is the default.

What the encoders see (default since 2026-11-22): counts with each library's measured GC offset taken off
(`--clean-inputs`; the size factors are then made from these counts too), and a residual encoder that sees expression
only (`--no-encoder-covariates`). The decoder is unchanged: it predicts the raw counts, with its linear covariate term
on all covariates. On the atlas this took GC out of the embedding (GC-slope R2 from z 0.24 -> 0.03 within biological
replicate groups, 0.33 -> 0.02 over all samples) with a slightly better TEST NLL (6.206 against 6.217;
`work/qc_module/NOTES.md`). `--no-clean-inputs --encoder-covariates` give the fit without the cleaning; bundles made that way keep
it. `embed` repeats the cleaning and needs each sample's GC fit, which the QC record has. `train --qc-module` adds the
decoder's GC term at fixed scale and the genomic-DNA term on top; it is not the default (it kept GC in z across
samples). The manifest reports the encoder side under `encoder_inputs`.

`train --dna-penalty W` (default 1 since 2026-11-22; 0 = off) adds W x the R2 of each sample's genomic-DNA estimate from the
residual block, both centred within biological replicate groups (metadata_v2 `replicate_group`), to the loss: every
step also encodes up to 256 replicate groups of the training rows. Replicates share their biology, so DNA that the
residual holds between them is technical; the input cleaning does not reach it (`work/qc_module/NOTES.md`). On the atlas it took the DNA estimate out of
z within replicates (R2 0.38 -> -0.03), kept the decoded within-study treatment contrasts (r 0.80) and raised kNN
agreement (phase / medium / lab strain 0.74 / 0.86 / 0.73 -> 0.78 / 0.89 / 0.86); study clustering within
conditions shared by several studies fell (0.67 -> 0.60). Without replicate groups in the metadata the term is left
out; the manifest reports it under `dna_penalty`.

The GC offset is the library's fitted curve averaged over each feature's 300 bp windows, the window the inGEST fit
uses (log of the mean of exp(b1 x + b2 x^2) over the windows); within replicate groups it explains more of a
sample's deviation than the curve at the feature's mean GC, with a slope close to 1 (`work/qc_module/gc_fragment.txt`;
windows of the library's own fragment size add nothing). The GC offset needs each feature's GC content and window profile: the package ships it for the inGEST counting
annotation (`reference/feature_gc_counting_v2.tsv`, made by `scripts/feature_gc.py`); `--feature-gc`
takes a table for another annotation, and `embed` refuses a table whose features or lengths do not
match the bundle.

### Genomic DNA in the library

DNA left in a library adds reads to every feature in proportion to its length, whatever its
expression. It is handled by leaving the worst libraries out:

- **Atlas:** samples whose DNA estimate (`dna_fraction_est` of the QC table) is above 10% are not
  in the model (`qc_table.py mark-model --max-dna-fraction 0.10`; 38 samples, `high_dna`).
- **New samples:** `embed` writes `dna_fraction_est` and `high_dna` to `samples.tsv` and prints a
  warning for samples above the bundle's limit (`max_dna_fraction` in the manifest); they are
  still embedded.
- A shallow library reads 0 (below the estimate's detection limit) whatever its DNA content, so
  it passes.
- **Library depth:** `embed` also writes `cds_reads` (reads on CDS features) and `low_depth` (1 below 200,000, the
  atlas's depth gate `qc_table.py mark-model --min-cds-reads`; the bundle's `min_cds_reads`) and warns for such samples;
  it does not refuse them. No atlas sample is that shallow. The model has not learned that shallow means stationary in
  general (deep atlas samples thinned to 200,000 CDS reads kept their call; within replicate groups depth barely
  tracks the phase score), but single studies can be fragile: exponential libraries of GSE67402 thinned to 200,000,
  and its total-RNA libraries (about 100,000), were called stationary (`work/qc_module/depth_checks.txt`).

Two corrections exist in the code and are **off** (`train --dna-decoder`, `--dna-inputs`): an
additive DNA term in the decoder's count mean (`models.count_log_mean`, expected reads from
`data.dna_density`) and taking the expected DNA reads off the counts the encoders see. Tried on
the atlas (2026-11-21, `outputs/sweeps/models_v2_dna`, `outputs/sweeps/models_v2_dna_decoder`): the decoder term
changed nothing useful; the input correction brought libraries with 5 and 20% added DNA back to
their DNA-free twins at normal depth, but made shallow libraries much worse and gave no net gain
on the atlas.

### Genes missing from a strain

Counts are against MG1655, so a gene a strain lacks has no reads. Such genes are called from the
counts (`preprocess.missing_genes`: runs of neighbouring genes far below the low end of their
expression, `docs/atlas_model_v2.md` section 4) and left out of the size factor, the gene statistics
and the count likelihood; the encoders see the gene's mean. In training, half of the samples of a
batch also get the missing genes of another sample as input. `train --no-mask-missing` treats the
zeros as measured, as before. A bundle trained without the calls is embedded without them.

`train` writes the calls of the atlas samples to `missing_genes.tsv` in the output directory (one row
per sample and gene; once per seed sweep, since the calls do not depend on the seed), and their
number per sample to `n_missing_genes` in `predictions.tsv`.

The samples of a study that lack the same genes are also called together on their summed counts
(found from the calls themselves, no strain metadata), which finds the weakly expressed genes of a
missing island; `--no-pool` switches this off. In `embed` all samples of the count table count as
one study unless `--studies` gives a table sample -> study, so a sample's calls can depend on the
samples embedded with it; one sample alone, or `embed --no-pool`, is called on its own.

With `--antisense` (the antisense count table of the same samples: `atlas_v2_antisense_counts.tsv`
for the atlas, inGEST's `all_samples_antisense_counts.txt` for new samples) a silent gene that has
antisense reads is not called missing. Give the table to `evaluate`, `figures` and `embed` as well
when the bundle was trained with it (`missing_genes.antisense_veto` in the manifest); without it
`embed` calls from the sense counts alone.

### `compare`: new samples against the atlas

The atlas is the embedding of the samples a bundle was trained on (`embedding.tsv`, `predictions.tsv`
of the bundle), so the new samples must have been embedded with the same bundle. `--embeddings` is
the output directory of `embed`.

| File | Content |
|---|---|
| `neighbors.tsv` | per new sample its `-k` (15) nearest atlas samples: distance, study, curated labels, the model's calls; with `--atlas-metadata` also strain, medium, growth phase, carbon source, oxygen, treatment, genotype |
| `summary.tsv` | per new sample: nearest atlas sample, most frequent study among the neighbours, the neighbours' vote on every call (`<call>_knn_vote`, `_knn_frac`, `_knn_n_labelled`) and whether the model's own call agrees (`<call>_agree`), mean neighbour distance within every block (`<block>_knn_dist`), novelty |
| `query_distances.tsv` | distances between the new samples |
| `group_neighbors.tsv` | with `--metadata` and `--group-by`: the pooled neighbours of each group by atlas study, strain, genotype class, treatment, growth phase and base medium (the last five with `--atlas-metadata`): number and share of the group's neighbours, how many of its samples have one there, mean distance |
| `group_summary.tsv`, `fig_*` | with `--metadata` and `--group-by`: one row per group (calls, votes, top studies, distance within the group and to the previous group) and the figures `fig_query_distances`, `fig_head_by_group`, `fig_neighbor_labels`, with `--umap` also `fig_umap_groups` |
| `umap_atlas.tsv`, `umap_query.tsv`, `umap.png` | with `--umap`: a UMAP fitted on the atlas with the new samples projected onto it |

The votes use the curated labels of the atlas samples (the labels the calls were trained on);
neighbours without a label do not vote. `novelty_percentile` places a sample's distance to its
k-th neighbour within the same distance of every atlas sample to the samples of other studies: near
1, the sample is farther from the atlas than almost any atlas sample is from the rest of it.
`--blocks phase,residual` restricts the distances to those blocks; `--metric` is cosine (default) or
euclidean. Without `--metadata` / `--group-by` / `--plots` only the tables are written.
`--exclude-study GSE` (repeatable) leaves that study's atlas samples out of every table: when the new samples come from
a study the atlas holds already (a TEST study recounted, as in `examples/GSE147611_greAB_deletion`), their own atlas
copies would otherwise be their nearest neighbours.

### `extract`: atlas studies as new samples

```bash
ecoli_atlas extract GSE224463 --models outputs/models --seeds --out atlas
```

The bundle holds the embedding and the calls of every atlas sample (training, TEST and withheld studies). `extract`
writes those of the named studies in the layout of `embed` (`embedding.tsv`, `predictions.tsv`, `samples.tsv` with
each sample's study and split, `extract.json`; with `--seeds` also `seeds/seed_<n>/` and `predictions_seeds.tsv`), so
`axes --query`, `compare` and the call figures can be used on a study the atlas holds. Give the study to their
`--exclude-study`, or its own atlas rows are its nearest neighbours (`axes` refuses query samples that are also atlas
samples). Example: `examples/GSE224463_rpoS_stress_time_courses`.

### `neighbors`: studies of the atlas against the rest of it

For every sample of the named studies: its nearest atlas samples outside its own study
(`--within-study` allows them; the sample itself is always left out), the neighbours' votes against
the sample's curated label (`<call>_agree_meta`) and the model's call (`<call>_agree_pred`), and the
novelty. `summary.tsv` has one row per sample, `study_summary.tsv` one per study,
`neighbors_manifest.json` the settings. `query_split_bucket` / `atlas_split_bucket` say where a
sample was in training: `test`, `fold<k>`, `train_only`, `withheld`, `projected` (in no split, e.g.
the treated samples of a TEST study).

### `view`: an interactive 3-D map of the atlas

Writes one self-contained HTML page that opens in any web browser, with no server. The
`ecoli_atlas` conda environment (`environment.yaml`) has everything it needs (plotly and
umap-learn); an environment made before they were added gets them with
`mamba env update -n ecoli_atlas -f ecoli_atlas/environment.yaml`, and a pip install with
`pip install "ecoli-atlas[view]"`. Without plotly the page loads plotly.js from a CDN (`--plotly cdn`, smaller file, needs a network
connection), or `--plotly PATH` inlines a local `plotly.min.js`. `--no-umap` skips the UMAP layouts,
which take a few minutes for the whole atlas. The page holds:

- **Layouts.** PCA of the whole embedding, PCA of the residual block, UMAP of each
  (`umap_all`, `umap_residual`), one layout per head block, and "named axes", which puts the phase,
  medium and lab-strain blocks on x, y and z. A trained head block is in effect one axis: its first
  component holds over 99% of its variance. So `block_<head>` draws that axis against residual
  PC1–2, and a block that is not flat (first component under 90%) gets its own 3-D PCA. Each block
  axis is signed so that it rises toward the call's positive class (stationary, minimal, lab).
  The PCA layouts keep the first 10 components (`--n-pcs`), and X / Y / Z menus next to the layout
  menu draw any three of them (e.g. PC2 × PC4 × PC5). In a `block_<head>` layout, x stays the block
  axis and y and z can be any two residual PCs. The chosen axes are part of the page's link
  (`#layout=pca_all&pcs=1,3,4`; the numbers count from 0).
- **Hover and click.** Hovering shows the sample, its GEO title, the model's calls and the curated
  labels. Clicking a sample opens a panel with every curated label and its basis, the GEO sample
  record, its SRA runs (linked), and the study.
- **QC covariates** (with `--counts` and `--qc`). These are the composition and QC values of every sample, computed
  as in `train` (the model itself uses 11 of them, `config.COVARIATES`): the composition values (mRNA reads, 16S/23S/5S/tRNA-to-mRNA ratios) from the count
  table, and the QC values (read and fragment lengths, adapter trimming, duplication, GC, strand
  purity, DNA fraction, leakage, intergenic share, gene-body slope, other-genus share, substitution
  rate, paired-end, instrument) from the QC table. They form a "QC covariates" group in the colour
  menu. The sample panel lists them with each value's percentile among the atlas samples. Your own
  samples get the same table from the `covariates.tsv` that `embed` writes.
- **Study mode** (the "Show this study" button, the search box, or a URL ending in `#study=GSE…`).
  The study is drawn over the faded atlas; "hide other samples" removes the rest of the atlas. The
  study is grouped by its *condition*: the label columns that differ between its samples, or the
  GEO sample title without its replicate part. You can also group by any single varying column or
  by a call. Clicking a group shows it alone; hovering a sample in the list marks it in the plot.
  The study card shows the GEO title, summary and overall design, links to PubMed, BioProject and
  SRA, and the curation notes.
- **Examples.** These are preset views of single studies (growth-phase series, a biofilm, rich
  against minimal medium, a strain panel, the withheld study, the case studies). They are kept
  in `src/ecoli_atlas/viewer/presets.json` (`id`, `title`, `gse`, `layout`, `color_by`,
  `caption`, optionally `pcs`: the three columns to draw, counted from 0); `--presets FILE` replaces them, and presets whose studies are not in the bundle are
  dropped.
- **Your samples** (`--embeddings DIR[=NAME]`, repeatable: output directories of `embed` made
  with the same bundle). They are drawn as diamonds, projected onto the atlas fit of every layout.
  Clicking one lists its `-k` nearest atlas samples in the whole embedding (`--metric`, default
  cosine as in `compare`) and their studies.

The study and sample text come from `ecoli_ingest/atlas_build/metadata/study_text.py` (see
`docs/metadata_v2.md`). Without `--studies` / `--sample-text` the page still works, but without the
GEO text.

### `figures`

Without a bundle, two figures of the data, from the metadata and a `splits.csv`:
`fig_split_balance` (class proportions of the three labels per fold and TEST, by samples and by
studies, with `split_balance.tsv`) and `fig_corpus_composition` (pies of phase, medium, carbon
source, genotype class, stress family and strain group of the samples in the split).

With `--models` and the three tables, also the model figures (`--only` selects by the names below):

```bash
ecoli_atlas figures $T --models outputs/models --out examples/atlas_model_figures/figures [--only NAMES] [--strict]
```

| Name | Files | What it shows |
|---|---|---|
| `pca_batch` | `fig_pca_mg1655_batch`, `fig_pca_bw_batch`, `fig_pca_strain_all_complex`, `pca_log_expression_coords.tsv` | PCA of the log2 normalised expression (no model): MG1655 and BW25113 in one condition (exponential, untreated wild type, LB: one medium, `pca.MATCHED_MEDIUM`) coloured by study, and the rich-medium samples by strain |
| `batch_genes` | `fig_batch_genes`, `batch_genes.tsv` | the genes that differ most between studies within those two subsets |
| `gsea` | `fig_gsea_batch_go`, `gsea_batch.tsv` | GO-term enrichment (one-sided by default) of one pooled between-study ranking: every LB exponential, untreated wild-type study of a laboratory strain (MG1655, BW25113 and the other K-12, B, C and W strains; each strain of a study apart), genes ranked by the SD of their per-study means, a gene left out of the studies whose strain lacks it (missing-gene calls); bars of the 20 terms with the largest NES (see [Gene-set enrichment](#gene-set-enrichment)). With `--modules TSV` also the module table: `gsea_batch_modules.tsv` and one bar figure per database, `fig_gsea_batch_{regulondb,kegg_pathway,kegg_module}`; RegulonDB as whole regulons (activated and repressed targets together, `modules.whole_regulons`), since the ranking has no direction |
| `batch_tracking` | `fig_batch_tracking`, `batch_tracking.tsv`, `batch_tracking_sets.tsv` | with `--modules`, after `gsea`: per study of the same lab-strain subsets (MG1655, BW25113, other lab strains as a third colour), the 9 RegulonDB regulons with the largest NES in `gsea_batch_modules.tsv`, in that order, against the ribosomal proteins (a growth-rate marker; `--tracking-x phase`: the study's mean phase ordinal score), one y span for every panel, R² and residual SD of one line: which study differences follow the growth rate and which do not |
| `head_test_eval` | `fig_head_test_eval`, `phase_misclassified.tsv` | confusion matrix of each call on TEST; the samples whose phase call is wrong |
| `head_scores` | `fig_head_scores`, `head_scores.tsv` | each head's score (the log-odds of its call, i.e. its one-dimensional block) by class, training pool and TEST, with the AUROC of each split |
| `head_gene_correlates` | `fig_head_gene_correlates`, `head_gene_correlations.{txt,tsv}` | Spearman rho of every gene's expression with each head's probability (the `.txt` lists the top 50 each way, for gene-set enrichment) |
| `study_leakage` | `fig_study_leakage` | share of a TEST sample's 25 nearest neighbours from its own study, in the z-scored expression and in the embedding |
| `threshold_curves` | `fig_threshold_curves`, `cv_predictions.tsv` | balanced accuracy against the threshold on the out-of-fold probabilities |

`figure_metrics.json` holds the numbers of every figure and two checks against the bundle: the
embedding made from the tables against `embedding.tsv`, and the TEST metrics against the manifest
(`--strict` exits 1 when one of them differs or a figure fails). `threshold_curves` reads
`cv_predictions.tsv` from the bundle (`train --cv`); for a bundle without it the study folds are
fitted once and the file is kept in the output directory. Those out-of-fold metrics are compared
with the manifest's `cv_metrics` but are not expected to be identical on another device.

### `casestudy`

```bash
ecoli_atlas casestudy {uti,oxygen,glu-cas,all} $T --models outputs/models --out OUT     # the examples write into examples/<study>/casestudy
```

Case studies on the embedding, each on a held-out (TEST) study. Each writes its figures (if any), its
sample and gene tables and `<name>_metrics.json`; `casestudy_metrics.json` collects them.

- `uti` (GSE128997): infection against urine culture, paired by patient isolate. The isolates'
  samples are centred per patient, and each patient is scored on a discriminant fitted without it
  (sign test, Wilcoxon, permutation of the labels within pairs). The genes along that axis are in
  `uti_axis_genes.tsv`; with `--gsea` / `--modules` their pathway analysis as for `oxygen` (positive NES = towards UTI).
- `oxygen` (default GSE189154; `--oxygen-studies`): aerobic against anaerobic or microaerobic,
  untreated samples of studies outside the training pool. One discriminant
  (`fig_oxygen_lda_projection`; the genes along it in `oxygen_axis_genes.tsv`); the AUROC is given in sample and with
  every sample scored by an axis fitted without it. Pathway analysis of the genes along the axis (preranked GSEA of their
  mean within-study rho, two-sided, positive NES = towards low oxygen): `--gsea` against the GO terms
  (`oxygen_gsea_go.tsv`, `fig_oxygen_gsea_go`), `--modules atlas_data/modules/modules.tsv` against KEGG pathways and
  modules, RegulonDB regulons and iModulons (`oxygen_gsea_modules.tsv`, with `database` and `category` columns; one
  figure per database, `fig_oxygen_gsea_<database>`); `--gene-sets`,
  `--gsea-permutations` as for `drivers`.
- `glu-cas` (GSE271069): glucose against casamino acids, paired by growth phase
  (`fig_glu_cas_lda_projection`; the genes along the axis in `glu_cas_gene_correlations.tsv`). The study is a
  forced TEST study; the metrics say whether the bundle was fitted on its samples (`fitted_by_the_model`).

### `axes`: ordered categories as dimensions of the embedding

Takes a bundle's embedding and one or more ordered axes (a metadata column and its values in
order) and gives every sample a position on each axis.

```bash
ecoli_atlas axes --models outputs/models --metadata atlas_data/metadata_v2.tsv \
    --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
    --where treated=false --where standard_rnaseq=true --holdout lag --holdout late_stationary --holdout long_term_stationary \
    --axis medium:medium_class:complex,defined_rich,minimal_plus_hydrolysate,defined_minimal \
    --series time_courses.tsv --out examples/atlas_axes/axes
```

- **Ordinal score:** a proportional-odds model, one linear score of the embedding with a threshold
  between neighbouring categories. It is not bounded, so a sample can lie beyond the last fitted
  category. A sample is always scored by a model that has seen no sample of its study.
- **Anchor score:** the direction that separates the two end categories, fitted on those alone;
  everything in between is placed without having been fitted.
- Both are scaled so that the first fitted category has median 0 and the last median 1.
- `--holdout CATEGORY` leaves a category out of both fits and only scores it: the test of
  extrapolation (late stationary beyond stationary) or interpolation. An end category needs samples
  of at least two studies, otherwise it is refused.
- A category can group several values of the column: `label=value1+value2`, e.g.
  `--axis carbon:carbon_source:sugar=glucose+other_sugar,non_sugar=glycerol+organic_acid`. The score table then
  keeps each sample's own value (`value`) and the summary has the median of every value (`value_medians`), so the
  values inside a category can be compared along the axis.
- `--where COLUMN=VALUE` restricts which samples get a category; `--blocks` restricts the embedding
  to named blocks (`phase`, `medium`, `lab_strain`, `residual`).
- `--series` is a table `sample, series, x[, x_label]` of ordered series (time courses), drawn as
  trajectories of the ordinal score, or `PATH:SERIES_COLUMN:X_COLUMN` for any sample table (CSV or TSV,
  first column the sample id), e.g. `samples.csv:replicate:time_h`; a fourth field names a panel column
  (`samples.csv:replicate:time_min:panel`, or a column `panel` of the table) and draws one panel per value instead of one
  per study. One colour per series, marker shape = the
  sample's category. `--series-x` sets the x axis: `categorical` (the positions as categories labelled with their
  values, the series side by side within each), `linear`, `log`, or `auto` (default: categorical with at most 12
  positions in a panel, else log when they span more than 20-fold, else linear).
- `--query DIR` places new samples (an `embed` output made with the same bundle) on the axes: they are
  only scored, by the model of all fitted samples, get `role` = query in the score tables and are drawn
  over the map of two axes, joined along their series. `--query-study` names their study (default
  `query`); `--exclude-study GSE` leaves an atlas study out of the fits and the tables, e.g. the study the
  new samples come from (the worked example `examples/GSE67402_*` does both). With the `predictions.tsv` of the
  `embed` folder the calls are drawn too, on their 0-1 scale: `fig_heads_series` (along `--series`) and
  `fig_heads_phase_by_medium` (P(stationary) against P(minimal), the atlas from the bundle's `predictions.tsv` behind).
- `--seeds` fits every axis again on the embedding of each `seed_<n>` model of the bundle (same samples, same
  scaling) and adds `ordinal_seed_median/min/max`, `anchor_seed_median/min/max` and `n_seeds` to the score tables,
  `axis_<name>_seed_scores.tsv` (every seed's ordinal score) and every seed's score as a small dot in the series
  figures. `--query` samples must then be embedded with `embed --seeds`; with `--series` the share of seeds making each
  call is also drawn as one stacked bar per sample (`fig_heads_seed_votes`).

The example axes of the paper (growth phase, oxygen, sugar against non-sugar carbon sources and glycerol, the last two in
defined minimal media) and their drivers are the example `examples/atlas_axes/` (`bash examples/atlas_axes/run.sh` in the
`ecoli_atlas` environment; `MODELS=<bundle>` for another bundle; one folder per axis under `axes/` and `drivers/`).

Output per axis: `axis_<name>_scores.tsv` (category, role, both scores per sample),
`axis_<name>_summary.json` (rank correlation; share of within-study sample pairs in the order of
their categories, over all pairs and as the mean over studies, also per pair of categories;
category medians; thresholds; the held-out categories; all of it also for the samples the
embedding never fitted), `fig_axis_<name>_distributions`, `fig_axis_<name>_series`, and for two
axes `fig_axes_<a>_by_<b>`.

### `drivers`: the genes that move a call

Takes a bundle and the training tables and, for a head and a pair of its classes, ranks the genes,
and the groups of co-transcribed genes, whose change moves the head's score from the low class to
the high class.

```bash
ecoli_atlas drivers $T --models outputs/sweeps/models_ordinal --out work/drivers_ordinal   # all seeds of the sweep
ecoli_atlas drivers $T --models outputs/models --head medium --gsea --out examples/atlas_drivers/drivers/medium
```

- **Method:** integrated gradients of the head's score (`phase_logit` for the phase head) along the
  line from the mean expression profile of the low class to each sample of the high class. The
  attributions of a sample add up to its score minus the score of that baseline; they are averaged
  with every study counting once.
- **Contrasts:** by default the first class against the last, and with the ordinal phase head also
  `exponential:transition` and `transition:stationary`; `--contrast LOW:HIGH` names others. It works
  for either phase head, any fit and any head (`--head`).
- **Seeds:** when `--models` holds the `seed_<n>` bundles of a sweep, all of them are used
  (`--one-model` uses only the bundle itself). Attributions are the mean over the seeds;
  `seeds_same_sign` and `seeds_in_top` say in how many seeds a row has that sign and is among the
  top rows (`--top-genes` 200, `--top-units` 100), `attribution_sd` is the spread. Models that
  differ only in the seed are equally good, so a row that is high in one seed only is not a finding.
- **Expression units:** co-transcribed genes carry the same information, and a model may put its
  weight on any of them. Neighbouring genes on the same strand, at most 500 bp apart, whose
  expression has a correlation of at least `--unit-r` (0.6) within studies are joined into a unit, a
  stand-in for operons made from the atlas itself, and the attributions of a unit are added up.
  **The unit table is the one to read**; the gene table shows which members a model used.
- **Studies:** the high class is a sample of studies, and with few of them the choice of studies
  moves the ranking more than the seed does (phase, 33 stationary studies: seeds give about a
  quarter of the uncertainty of the unit attributions, studies the rest; medium and lab strain,
  with 130 and 300 studies, about half each). `--study-boot N` (200) draws the high-class studies
  again with replacement, on the mean over the seeds, and gives `study_in_top`, `study_same_sign`
  and `study_sd`. The low-class baseline is held fixed, so the study uncertainty is if anything
  underestimated. A unit with a `study_in_top` near 1 does not hang on a few studies.

`units_<head>_<low>_to_<high>.tsv` (one row per unit) and `drivers_<head>_<low>_to_<high>.tsv` (one
row per model feature), strongest first:

| Column | Meaning |
|---|---|
| `unit`, `genes`, `n_genes`, `products` | units only: a short name (`pspABCDE`, or `first-last (n)`), the members in genome order and their products |
| `attribution` | contribution of the change of the gene, or of all genes of the unit, to the score; positive = moves the score towards the high class |
| `attribution_per_gene` | units only: `attribution` divided by the number of genes, since a large unit adds up more |
| `share` | its part of the summed absolute attributions |
| `direction`, `log2fc` | whether the gene is higher (`up`) or lower (`down`) in the high class in the data, and by how much (mean of the members for a unit) |
| `n_seeds`, `seeds_same_sign`, `seeds_in_top`, `attribution_sd` | with several seeds, see above |
| `study_in_top`, `study_same_sign`, `study_sd` | share of the study draws in which the row is among the top rows (`--top-genes`, `--top-units`), share with the sign of `attribution`, spread of `attribution` over the draws |
| `same_sign` | genes only: share of the high-class samples (studies weighted equally) in which the attribution has that sign |
| `score_spearman` | genes only: rank correlation of the gene's expression with the score over the samples of both classes |

The sign of the attribution is not the direction of the gene: a ribosomal protein operon that falls
in stationary phase and so raises the phase score has a positive attribution and direction `down`.
A negative attribution is a gene whose change pulls the score back towards the low class.

`drivers_<head>.json` has, per contrast, the sample and study numbers, how concentrated the
attribution is, the split between units that are up and down, the rank correlation of the unit
attributions between seeds, the top units of each kind that hold in at least three of four
seeds, and the number of units among the top rows in at least three of four seeds and in at least
three quarters of the study draws. `fig_units_*` and `fig_drivers_*` show the top `--top` rows of each direction; with several
seeds a row that does not hold in three of four seeds is drawn pale.

**Drivers of an axis** (`--axis axis_<name>_summary.json`): the same for a dimension made by
`axes` on this bundle. The score is the axis's score (`--axis-score ordinal`, default, or `anchor`),
its weights applied to the embedding columns standardised over the axis's fitted samples, as `axes`
fits them; the classes are the axis's fitted categories (default contrast: first against last, and
every neighbouring pair). With the `seed_<n>` bundles of a sweep the axis is fitted again on each
seed's embedding (the same fitted samples, model and 0-1 scaling as `axes`, so the seeds' scores are
on one scale) and the seeds are averaged as for a head; `--one-model` uses the axis as fitted on the
bundle itself. Axis drivers depend more on the seed than head drivers (phase axis, units, one seed
against another: top-50 overlap 0.72, phase head 0.80), and averaging the 8 seeds brings the phase
axis's drivers close to the phase head's (top-50 overlap 0.68 with one seed, 0.76 with eight, against a
ceiling of 0.78). Output names start with `axis_<name>` (`drivers_axis_<name>_<low>_to_<high>.tsv`, ...).
An axis that leans on the residual block also brings in what the residual carries besides the axis
(culture environment, strain): read its drivers with `study_in_top`.

```bash
ecoli_atlas axes --models outputs/models --metadata atlas_data/metadata_v2.tsv --axis oxygen:oxygen:aerobic,anaerobic_fermentation --out work/axes_oxygen
ecoli_atlas drivers $T --models outputs/models --axis work/axes_oxygen/axis_oxygen_summary.json --gsea --out work/drivers_oxygen
```

**GO terms** (`--gsea`): two tests, one for the genes higher in the high class (`direction` up) and one for those
lower in it (down), each ranking its genes by their signed attribution (seed mean) against the GO sets (see
[Gene-set enrichment](#gene-set-enrichment)): `gsea_<contrast>_up.tsv` / `_down.tsv` and `fig_gsea_<contrast>_up` /
`_down`, each with the terms whose genes' change moves the score towards the high class on the left and towards the
low class on the right; the counts and top terms per direction in the JSON (`gsea`). A ribosomal protein term, for
example, is in the `down` test (lower in stationary phase) on the side that moves the score towards stationary. Split
this way, a term says both which way its genes change and what that change does to the call.

### Gene-set enrichment

`figures` (`gsea`), `drivers --gsea` and the case studies (`--gsea`, `--modules`) run a preranked GSEA (every gene
weighted the same) of gene sets of 5 to 500 ranked genes, 10,000 permutations (`--gsea-permutations`), so the
smallest p-value is 1e-4. Two tests (`--gsea-sides`):

- `one` (default of the `figures` panel `gsea`): are a set's genes at the top of the ranking? fgsea's
  `scoreType = "pos"` test with `gseaParam = 0`, written in numpy (`gsea.prerank_one_sided`): ES = the largest
  positive deviation of the running sum, null = random sets of the same size, NES = ES / mean null ES (a set with no
  enrichment near 1, one at the bottom near 0, never negative), `fdr` = Benjamini-Hochberg. It gives the same ES as
  gseapy for sets at the top of a ranking and agrees with R fgsea (NES r 0.9998 on the same rankings).
- `two` (default of `drivers` and the case studies, gseapy): top or bottom; a positive NES means the term's genes sit
  at the top of the ranking, a negative NES at the bottom, and a set with no enrichment still gets an NES near +1 or
  -1; `fdr` is gseapy's FDR q-value.

Genes are matched by lower-case symbol; about 90% of the CDS features are in a GO set. Result tables have `term`,
`database`, `name`, `namespace`, `category`, `size`, `es`, `nes`, `pval`, `fdr` and `lead_genes`.

The GO sets ship with the package (`reference/go_gene_sets_ecocyc.tsv.gz`, 4,112 terms, 126 KB):
the EcoCyc annotations of E. coli K-12 (`ECOLI-mod.gaf`, direct annotations, NOT qualifiers left
out) with the term names and namespaces of `go-basic.obo`; the file's header gives both versions.
Gene Ontology data are CC BY 4.0 (geneontology.org). Rebuild with newer files:

```bash
curl -LO https://current.geneontology.org/annotations/gaf/ECOLI-mod.gaf.gz
curl -LO https://current.geneontology.org/ontology/go-basic.obo
python ecoli_atlas/scripts/go_gene_sets.py --gaf ECOLI-mod.gaf.gz --obo go-basic.obo \
    --out ecoli_atlas/src/ecoli_atlas/reference/go_gene_sets_ecocyc.tsv.gz
```

`--gene-sets TSV` uses another table of the same columns (`term`, `namespace`, `name`, `genes` as
comma-separated symbols), e.g. the module table of `download modules` below.

### Curated gene modules

`drivers --modules TSV` scores curated gene modules (regulons, iModulons, pathways) with the drivers' attributions,
for a head or an axis: `modules_<contrast>.tsv`, `fig_modules_<contrast>` and the top modules per source in the JSON
(`modules`). A module's `net` is the sum of its members' attributions, averaged over the seeds and the high-class
studies: its part of the score gap (positive = it moves the score towards the high class). It is compared with random
modules of the same size drawn as whole expression units, so co-transcribed genes are not counted as independent
(`--module-null`, 10,000 draws): `z_net`, `p_net`, and `fdr_net` (Benjamini-Hochberg within a source). The null is a
little anti-conservative and heavy-tailed (a few genes carry large attributions), so read `z_net` together with
`boot_in_top` (the share of study draws in which the module stays among the top modules of its source) and
`seeds_same_sign`, rather than `fdr_net` alone. `mean_log2fc` and `frac_same_dir` give the members' change in the data;
`lead_genes` the members with the largest attributions. Modules of 3 to 500 model genes are scored
(`--module-min`, `--module-max`).

```bash
ecoli_atlas download modules                       # atlas_data/modules/modules.tsv (818 modules)
ecoli_atlas drivers $T --models outputs/models --head phase --modules atlas_data/modules/modules.tsv --out work/drivers_phase
```

`download modules` builds the table from:

| Source | Modules | Fetched from |
|---|---|---|
| `regulondb` | RegulonDB regulons of transcription factors and sigma factors split by direction (290 modules): the genes a regulator activates, `X (activated)`, and those it represses, `X (repressed)`; links of unknown or dual effect are left out | `TRN.csv` of [SBRG/precise1k](https://github.com/SBRG/precise1k) (MIT licence), pinned to one commit and checked by SHA-256 |
| `imodulon` | PRECISE-1K iModulons (200): the `imodulon_size` genes with the largest weight | `M.csv`, `imodulon_table.csv` of the same commit |
| `kegg` | KEGG pathways (global maps left out) and modules of E. coli K-12 MG1655 (241), each with its KEGG class B as `category` (BRITE br08901 for pathways, ko00002 for modules) | the KEGG REST service (`rest.kegg.jp`; free for academic use, not to be redistributed; the retrieval date is in the table's header and in `modules_manifest.json`) |

Members are mapped to the counting annotation by locus tag (b-number), else by symbol (`--features`, default the
annotation under `resources/`). EcoCyc is not fetched (it needs a licence): an EcoCyc export, or any list of gene sets,
can be given as a table with the same columns (`term`, `namespace`, `name`, `genes` as comma-separated symbols or locus
tags; the part of `namespace` before its first `_` is the source). The same table works as `--gene-sets` for `--gsea`.

## On a SLURM cluster

`scripts/slurm_train.sh` trains on one GPU. Name an x86 GPU type in `--gpus`; the environment does
not run on ARM GPU nodes.

```bash
sbatch ecoli_atlas/scripts/slurm_train.sh
SEEDS=1,2,3,4,5,6,7,8 CV=1 sbatch ecoli_atlas/scripts/slurm_train.sh
FIT=one-pass OUT=outputs/sweeps/models_one_pass SPLITS=outputs/models/splits.csv sbatch ecoli_atlas/scripts/slurm_train.sh
```
