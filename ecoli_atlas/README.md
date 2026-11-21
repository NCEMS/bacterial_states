# Ecoli_ATLAS

Embeds *E. coli* RNA-seq samples with an encoder–decoder trained on thousands of public samples
counted with Ecoli_inGEST, and places new samples in that embedding.

The embedding has the block layout of the first atlas model, 46 dimensions, and is fitted in the
same two stages (count model and covariates: [`docs/atlas_model_v2.md`](../docs/atlas_model_v2.md)):

| Block | Dimensions | Fitted on |
|---|---|---|
| growth phase (exponential / stationary) | 6 | training samples with a phase label |
| medium type (rich / minimal) | 12 | training samples with a medium label |
| lab strain (`lab_strain`: lab / non-lab) | 4 | training samples with a known strain group |
| residual | 24 | all training samples, blocks frozen |

- **Block encoders:** z-scored log expression → the block; a linear head on the block gives the
  call. Class-weighted cross-entropy, labelled samples only.
- **Residual encoder:** z-scored log expression plus library covariates → the residual block.
- **Decoder:** the whole embedding and the covariates → expected counts, with a negative binomial
  likelihood on the raw counts and the size factor as an offset.
- **Alignment terms** (residual stage, on the whole embedding): samples of one study known to
  differ are kept apart; matching control conditions of different studies are drawn together.

This is the first stage of the design. Not yet in the model: batch-corrected counts. The earlier model and its `verify-models` / `download` commands are on
branch `main`.

## Install

```bash
mamba env create -f ecoli_atlas/environment.yaml
conda activate ecoli_atlas
pip install -e "./ecoli_atlas[dev]"
pytest ecoli_atlas/tests
```

The environment has the CPU build of PyTorch. A single training run takes minutes on 8 CPU threads
(about 3 s per epoch on 6,800 samples); a GPU is worth it for seed sweeps and cross-validation. For
a GPU node:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Without installing, `bin/ecoli_atlas` runs the CLI from the checkout if the dependencies are present.

## Inputs

| Table | Made by | Content |
|---|---|---|
| `atlas_v2_counts.tsv` | `ecoli_ingest/atlas_build/full_run/collect_results.py` | features × samples, inGEST feature ids (`CDS_b0002_thrA`), GSM columns |
| `metadata_v2.tsv` | `ecoli_ingest/atlas_build/metadata/apply_labels.py` | labels per sample ([`docs/metadata_v2.md`](../docs/metadata_v2.md)) |
| `sample_qc.tsv` | `ecoli_ingest/atlas_build/prescreen/qc_table.py` | QC values per sample; `in_model` names the samples of the model |
| `*_counting_v2.features.tsv` | the inGEST counting annotation | type, symbol and length of every feature |

**Features.** mRNA features (CDS, pseudogenes) and the other RNAs (ncRNA genes, Rfam hits) are
encoder inputs and decoder outputs. Only mRNA features enter the size factor. rRNA and tRNA are
neither: their abundance relative to mRNA goes to the model as covariates.

**Covariates** (`data.py`): reads on mRNA features and the 16S, 23S, 5S and tRNA ratios (always
available, from the counts); read-length quantiles, cycles, share of reads kept, adapter share,
duplication, GC, strand purity, fragment-size quantiles, DNA-background estimate, leakage,
intergenic share, gene-body slope, share of reads on short features, largest other genus, substitution
rate against MG1655 (log; a property of the sequencing run more than of the strain); layout
and instrument family. A value that does not exist for a sample is set to the training mean, with
flags for "no QC row" and "DNA estimate below its limit". Study and strain are not covariates.
Covariates go to the residual encoder and the decoder, not to the block encoders.

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
  empty. The probability column is `p_lab`. A bundle trained for other calls (`outputs/models_v2_k12`
  has a `k12` call, K-12 against the rest) is refused by this version.

**Sample selection.** `qc_table.py mark-model --metadata metadata_v2.tsv --min-cds-reads 200000 --max-dna-fraction 0.10`
marks the samples that are usable by the metadata, were counted by the full inGEST run, have at
least 200,000 reads on CDS features and a genomic-DNA estimate of at most 10%.

## Commands

```bash
T="--counts atlas_data/atlas_v2_counts.tsv --metadata atlas_data/metadata_v2.tsv --qc atlas_data/sample_qc.tsv \
   --features resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"

# the study split only (splits.csv), then the figures of the data
ecoli_atlas train $T --out outputs/models_v2 --split-only
ecoli_atlas figures --metadata atlas_data/metadata_v2.tsv --splits outputs/models_v2 --out outputs/figures_v2

# train on that split and evaluate on the held-out studies
ecoli_atlas train $T --out outputs/models_v2 --splits outputs/models_v2/splits.csv [--device cuda]
ecoli_atlas train $T --out outputs/models_v2 --splits outputs/models_v2/splits.csv --seeds 1,2,3,4 --cv

# evaluate a bundle again; embed new samples
ecoli_atlas evaluate $T --models outputs/models_v2 --out outputs/models_v2_eval
ecoli_atlas embed --ingest-results my_run --models outputs/models_v2 --out embedded     # an inGEST run directory
ecoli_atlas prepare --ingest-results my_run --out prepared [--metadata my_samples.tsv]   # the same tables, to look at or join
ecoli_atlas embed --counts prepared/counts.tsv --qc prepared/qc.tsv --antisense prepared/antisense_counts.tsv \
    --models outputs/models_v2 --out embedded

# new samples against the atlas; studies of the atlas against the rest of it
ecoli_atlas compare --embeddings embedded --models outputs/models_v2 --out compared \
    [--atlas-metadata atlas_data/metadata_v2.tsv] [--metadata prepared/samples.tsv --group-by time_h --order-by time_h] [--umap]
ecoli_atlas neighbors GSE224463 --models outputs/models_v2 --out neighbors [--metadata atlas_data/metadata_v2.tsv] [--umap]
```

### Ways of fitting (`train --fit`)

| `--fit` | Embedding (46 dimensions) | Fitting |
|---|---|---|
| `two-stage` (default) | blocks 6 / 12 / 4 and residual 24 | block encoders on their labelled samples, then the residual encoder and the decoder on all training samples with the blocks frozen and standardised |
| `one-pass` | the same blocks and residual | everything together in one loss: the heads' cross-entropy, the count likelihood, the KL term and the two alignment terms. The block encoders also receive the count and alignment gradients; the blocks are not standardised |
| `unstructured` | one embedding without blocks | the encoder sees expression and covariates, the three heads read the whole embedding, all terms together |

The tables, labels, split, cross-validation and evaluation are the same for all three, so runs on
one `splits.csv` can be compared directly. The manifest records `fit` and `embedding_blocks`.

```bash
for FIT in two-stage one-pass unstructured; do
  ecoli_atlas train $T --fit $FIT --out outputs/models_v2_${FIT//-/_} --splits outputs/models_v2/splits.csv \
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
ecoli_atlas train $T --phase-head ordinal --out outputs/models_v2_ordinal --splits outputs/models_v2/splits.csv --seeds 1,2,3,4 --cv
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

The four case-study studies (`GSE128997`, `GSE189154`, `GSE78756`, `GSE80451`) are always TEST
(`config.py`). A study with more than the TEST share of the reference samples stays in the training pool.

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

### `embed`

The model takes every sample's QC values as covariates, so `embed` needs them. With an inGEST run,
`--ingest-results DIR` (the run directory or its results directory) is all it needs: it reads the
counts (`all_samples_raw_counts.txt`), the antisense counts (`all_samples_antisense_counts.txt`)
and the QC record of every counted sample (`<sample>/qc/<sample>.qc.json`), and writes the QC table
it used as `qc.tsv`. Otherwise give `--counts` and `--qc` (the `qc.tsv` of `prepare`, or a table with
the QC columns of `sample_qc.tsv`, as `full_<name>` or `<name>`, indexed by sample), and
`--antisense` when the bundle was trained with it. `prepare --ingest-results` writes the same four tables
(`counts.tsv`, `antisense_counts.tsv`, `qc.tsv`, `samples.tsv`).

`embed` refuses a sample without a QC row, or one that lacks a value every atlas sample has (read
lengths and cycles, kept and duplicated shares, GC, strand purity, DNA leakage, intergenic share,
gene-body slope, short-feature share, substitution rate; fragment sizes and mate-2 length for
paired-end libraries). The adapter-trimmed share and the DNA estimate may be missing. A run made
with `centrifuge: false` has no other-genus value: it is set to the atlas mean, with a warning. The
QC table made from the records gives the same covariates as `sample_qc.tsv` for all 7,731 atlas
samples, and embedding a recount batch this way gives the training embedding to 1e-5.

Feature ids must be those of the counting annotation (fails below `--min-feature-coverage`, default
0.9). Output: `embedding.tsv`, `predictions.tsv`, `samples.tsv` (size factor, reads on mRNA
features, number of genes called missing, DNA estimate), `missing_genes.tsv` (the genes called
missing from a sample's strain, see below) and `feature_coverage.tsv`.

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

Two corrections exist in the code and are **off** (`train --dna-decoder`, `--dna-inputs`): an
additive DNA term in the decoder's count mean (`models.count_log_mean`, expected reads from
`data.dna_density`) and taking the expected DNA reads off the counts the encoders see. Tried on
the atlas (2026-11-21, `outputs/models_v2_dna`, `outputs/models_v2_dna_decoder`): the decoder term
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
| `group_summary.tsv`, `fig_*` | with `--metadata` and `--group-by`: one row per group (calls, votes, top studies, distance within the group and to the previous group) and the figures `fig_query_distances`, `fig_factor_by_group`, `fig_neighbor_labels`, with `--umap` also `fig_umap_groups` |
| `umap_atlas.tsv`, `umap_query.tsv`, `umap.png` | with `--umap`: a UMAP fitted on the atlas with the new samples projected onto it |

The votes use the curated labels of the atlas samples (the labels the calls were trained on);
neighbours without a label do not vote. `novelty_percentile` places a sample's distance to its
k-th neighbour within the same distance of every atlas sample to the samples of other studies: near
1, the sample is farther from the atlas than almost any atlas sample is from the rest of it.
`--blocks phase,residual` restricts the distances to those blocks; `--metric` is cosine (default) or
euclidean. Without `--metadata` / `--group-by` / `--plots` only the tables are written.

### `neighbors`: studies of the atlas against the rest of it

For every sample of the named studies: its nearest atlas samples outside its own study
(`--within-study` allows them; the sample itself is always left out), the neighbours' votes against
the sample's curated label (`<call>_agree_meta`) and the model's call (`<call>_agree_pred`), and the
novelty. `summary.tsv` has one row per sample, `study_summary.tsv` one per study,
`neighbors_manifest.json` the settings. `query_split_bucket` / `atlas_split_bucket` say where a
sample was in training: `test`, `fold<k>`, `train_only`, `withheld`, `projected` (in no split, e.g.
the treated samples of a TEST study).

### `figures`

Without a bundle, two figures of the data, from the metadata and a `splits.csv`:
`fig_split_balance` (class proportions of the three labels per fold and TEST, by samples and by
studies, with `split_balance.tsv`) and `fig_corpus_composition` (pies of phase, medium, carbon
source, genotype class, stress family and strain group of the samples in the split).

With `--models` and the three tables, also the figures of the earlier model's paper, made for this
model (`--only` selects by the names below):

```bash
ecoli_atlas figures $T --models outputs/models_v2 --out outputs/figures_v2 [--only NAMES] [--strict]
```

| Name | Files | What it shows |
|---|---|---|
| `pca_batch` | `fig_pca_mg1655_batch`, `fig_pca_bw_batch`, `fig_pca_strain_all_complex`, `pca_log_expression_coords.tsv` | PCA of the log2 normalised expression (no model): MG1655 and BW25113 in one condition (exponential, untreated wild type, rich medium) coloured by study, and the rich-medium samples by strain |
| `batch_genes` | `fig_batch_genes`, `batch_genes.tsv` | the genes that differ most between studies within those two subsets |
| `factor_test_eval` | `fig_factor_test_eval`, `phase_misclassified.tsv` | confusion matrix of each call on TEST; the samples whose phase call is wrong |
| `factor_umaps` | `fig_factor_umaps`, `fig_factor_umaps_overlay`, `umap_factor_coords.tsv` | UMAP of each head's block, by label and by training pool / TEST |
| `factor_gene_correlates` | `fig_factor_gene_correlates`, `factor_gene_correlations.{txt,tsv}` | Spearman rho of every gene's expression with each head's probability (the `.txt` lists the top 50 each way, for gene-set enrichment) |
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
ecoli_atlas casestudy {uti,oxygen,glu-cas,all} $T --models outputs/models_v2 --out outputs/figures_v2
```

The case studies of the earlier model, on this model's embedding. Each writes its figures, its
sample and gene tables and `<name>_metrics.json`; `casestudy_metrics.json` collects them.

- `uti` (GSE128997): infection against urine culture, paired by patient isolate. The isolates'
  samples are centred per patient, and each patient is scored on a discriminant fitted without it
  (sign test, Wilcoxon, permutation of the labels within pairs). `fig_uti_axis` shows the genes
  along that axis.
- `oxygen` (default GSE189154; `--oxygen-studies`): aerobic against anaerobic or microaerobic,
  untreated samples of studies outside the training pool. One discriminant
  (`fig_oxygen_lda_projection`, `fig_oxygen_axis_genes`); the AUROC is given in sample and with
  every sample scored by an axis fitted without it. GSE78756, the second study of the earlier
  model's figure, did not pass the prescreen and is not in this atlas.
- `glu-cas` (GSE271069): glucose against casamino acids, paired by growth phase
  (`fig_glu_cas_axis`, `fig_glu_cas_lda_projection`, `fig_glu_cas_axis_genes`). For the earlier
  model this study was an outside test. It is now in the training pool, and the metrics say so
  (`fitted_by_the_model`).

### `axes`: ordered categories as dimensions of the embedding

Takes a bundle's embedding and one or more ordered axes (a metadata column and its values in
order) and gives every sample a position on each axis.

```bash
ecoli_atlas axes --models outputs/models_v2 --metadata atlas_data/metadata_v2.tsv \
    --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
    --where treated=false --where standard_rnaseq=true --holdout lag --holdout late_stationary --holdout long_term_stationary \
    --axis medium:medium_class:complex,defined_rich,minimal_plus_hydrolysate,defined_minimal \
    --series time_courses.tsv --out outputs/axes_v2
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
- `--where COLUMN=VALUE` restricts which samples get a category; `--blocks` restricts the embedding
  to named blocks (`phase`, `medium`, `lab_strain`, `residual`).
- `--series` is a table `sample, series, x[, x_label]` of ordered series (time courses), drawn as
  trajectories of the ordinal score.

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
ecoli_atlas drivers $T --models outputs/models_v2_phase_ordinal --out outputs/drivers_v2_ordinal   # all seeds of the sweep
ecoli_atlas drivers $T --models outputs/models_v2 --head medium --out outputs/drivers_v2_medium
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
| `same_sign` | genes only: share of the high-class samples (studies weighted equally) in which the attribution has that sign |
| `score_spearman` | genes only: rank correlation of the gene's expression with the score over the samples of both classes |

The sign of the attribution is not the direction of the gene: a ribosomal protein operon that falls
in stationary phase and so raises the phase score has a positive attribution and direction `down`.
A negative attribution is a gene whose change pulls the score back towards the low class.

`drivers_<head>.json` has, per contrast, the sample and study numbers, how concentrated the
attribution is, the split between units that are up and down, the rank correlation of the unit
attributions between seeds, and the top units of each kind that hold in at least three of four
seeds. `fig_units_*` and `fig_drivers_*` show the top `--top` rows of each direction; with several
seeds a row that does not hold in three of four seeds is drawn pale.

## On a SLURM cluster

`scripts/slurm_train.sh` trains on one GPU. Name an x86 GPU type in `--gpus`; the environment does
not run on ARM GPU nodes.

```bash
sbatch ecoli_atlas/scripts/slurm_train.sh
SEEDS=1,2,3,4,5,6,7,8 CV=1 sbatch ecoli_atlas/scripts/slurm_train.sh
FIT=one-pass OUT=outputs/models_v2_one_pass SPLITS=outputs/models_v2/splits.csv sbatch ecoli_atlas/scripts/slurm_train.sh
```
