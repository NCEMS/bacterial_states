# Ecoli_ATLAS

Embeds *E. coli* RNA-seq samples with a conditional encoder–decoder trained on thousands of public
samples, and places new samples in the resulting **atlas**.

The model is three supervised *factor encoders* (growth phase, medium type, lab strain) whose
embedding blocks are concatenated with the output of a *residual autoencoder* that reconstructs
expression from `[factor blocks | residual]` under a within-study push and a cross-study
control pull. The final embedding has 46 dimensions (`growth_phase[0..5]`, `medium_type[0..11]`,
`is_lab_strain[0..3]`, `residual[0..23]`).

## Install

```bash
mamba env create -f ecoli_atlas/environment.yaml
conda activate ecoli_atlas
pip install -e "./ecoli_atlas[dev]"
pytest ecoli_atlas/tests
```

The environment uses the CPU build of PyTorch, enough for `embed` and `compare`. For training on a
GPU node install a CUDA build of `pytorch` instead. Without installing you can run
`bin/ecoli_atlas` from the checkout if the dependencies are present.

## Data

Code and data are separate. Two bundles, downloaded with `ecoli_atlas download` (URLs are being set
up; until then the command tells you what to copy where):

| Bundle | Files | Default location |
|---|---|---|
| `models` | `models/{growth_phase,medium_type,is_lab_strain}_encoder.pt`, `residual_ae.pt`, `preprocessing.pt`, `manifest.json`, `splits.csv`; `atlas/final_embedding.csv`, `sample_predictions.csv`, `embedding_block_layout.csv`, `atlas_metadata.csv`, `atlas_manifest.json` | `outputs/` |
| `training-data` | `merged_count_table.txt` (4493 genes × 3342 samples), `metadata_revised.csv`, `GCF_000005845.2_ASM584v2_feature_table.txt` | `atlas_data/` |

Defaults are relative to `$BS_DATA_ROOT` (or the current directory); override with
`$ECOLI_ATLAS_DATA`, `$ECOLI_ATLAS_MODELS`, or the `--models` / `--atlas` / `--feature-table` options.

## Commands

Everyday use: `prepare`, `embed`, `compare`. Reproduction: `train`, `verify-models`, `figures`,
`casestudy`, `neighbors` (the last three are described further down).

```bash
# 1. inGEST results -> atlas input format (counts.tsv genes x samples, samples.tsv)
ecoli_atlas prepare --ingest-results my_run/tutorial_results --out prepared [--models outputs/models]

# 2. embed new samples -> final_embedding.csv, sample_predictions.csv, embedding_block_layout.csv, gene_coverage.tsv
ecoli_atlas embed --counts prepared/counts.tsv --models outputs/models --out embedded \
                  --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt

# 3. compare with the atlas -> neighbors.tsv, summary.tsv, query_distances.tsv [, umap_atlas.csv, umap_query.csv, umap.png]
ecoli_atlas compare --embeddings embedded/final_embedding.csv --atlas outputs/atlas --out compared -k 15 [--umap] \
                    [--metadata prepared/samples.tsv --group-by condition [--order-by time_h] [--color-by replicate]]

# 4. (re)train the model from the training bundle
ecoli_atlas train --counts atlas_data/merged_count_table.txt --metadata atlas_data/metadata_revised.csv \
                  --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt \
                  --out outputs/models_new --hparams outputs/models/manifest.json --splits outputs/models/splits.csv \
                  --no-tune --device cuda --write-atlas

# 5. compare two bundles
ecoli_atlas verify-models --old outputs/models --new outputs/models_new \
                  --counts atlas_data/merged_count_table.txt --metadata atlas_data/metadata_revised.csv \
                  --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt --report verify.json
```

`embed` accepts featureCounts output, the inGEST `*_gene_symbols.tsv` / `*_counts_extended.tsv`
tables, or any genes × samples TSV whose first column holds gene symbols or K-12 locus tags
(b-numbers). It drops rRNA/tRNA, renames locus tags to symbols with the feature table, aligns to the
model's 4385 genes (absent genes become 0 counts) and refuses to continue if fewer than
`--min-gene-coverage` (default 90 %) of the model genes are present. Check `gene_coverage.tsv`.

Class calls use argmax; `--apply-thresholds growth_phase` uses the saved operating threshold instead.

### `compare` output (`summary.tsv`, one row per new sample)

- `nearest_atlas_sample`, `nearest_distance`, `mean_knn_distance`, `top_study`, `top_study_frac`
- per factor: `<f>_knn_vote` (majority label among the k neighbours, from curated atlas metadata
  when available), `<f>_knn_frac`, `<f>_encoder_pred`, `<f>_agree`
- per block: `<block>_knn_dist` (mean distance to the neighbours inside that block's sub-space)
- `novelty_knn_dist` and `novelty_percentile`: distance to the k-th neighbour, as a percentile of
  the same statistic for atlas samples against other studies. Values near 1 flag samples unlike
  anything in the atlas.

`neighbors.tsv` lists every (sample, rank) pair with the neighbour's study, strain, growth phase,
medium and predicted labels. `--blocks growth_phase,residual` restricts distances to chosen blocks.

`query_distances.tsv` is the distance matrix between the new samples themselves (same metric and
blocks), i.e. how they relate to each other rather than to the atlas.

### Groups of samples: `--metadata`, `--group-by`, `--order-by`, `--color-by`

`--metadata` takes a table of the new samples with the sample id in the first column: a CSV, or the
`samples.tsv` that `prepare --metadata` writes. Its columns are put in front of `summary.tsv`, and
every sample must have a row. With `--group-by COL` (or `COL1,COL2`) the samples are summarised per
group, e.g. per condition or time point:

- `group_summary.tsv`, one row per group: `n`, `samples`; per factor `<f>_called_<class>` (fraction
  of the group's samples the encoder calls that class), `<f>_prob_<class>` (mean probability),
  `<f>_knn_vote` / `_knn_frac` (majority label among the group's pooled neighbours) and `<f>_agree`;
  `top_study`, `top_studies`; `mean_nearest_distance`, `mean_knn_distance`,
  `mean_novelty_percentile`; `within_group_distance` (mean distance between the group's own
  samples: replicate agreement) and `distance_to_previous` (between the centroids of consecutive
  groups).
- Groups are listed in the order of `--order-by COL` (numerically when the column holds numbers,
  such as a time), otherwise by their label.
- Figures (`--formats`, default pdf and png; `--plots` writes the first two without groups):

| Figure | Shows |
|---|---|
| `fig_query_distances` | heatmap of `query_distances.tsv`, samples sorted by group |
| `fig_umap_groups` | atlas UMAP in grey, new samples coloured by `--color-by` (default: the groups); with `--order-by` a line joins the group centroids in order. Needs `--umap` |
| `fig_factor_by_group` | per factor, the predicted probability of each sample along the ordered groups, with the group mean |
| `fig_neighbor_labels` | per group, what its atlas neighbours are: curated growth phase, medium type, lab strain, and study |

A worked example on a growth-to-starvation time course is in
[`examples/GSE67402_growth_starvation_time_course`](../examples/GSE67402_growth_starvation_time_course/README.md).

## Training and reproduction

`train` reproduces the notebook pipeline: metadata harmonisation → study-level TEST split and
3 study folds → z-score on CPM fit on the training pool → factor encoders (optional Optuna,
`--tune`) → out-of-fold operating thresholds → TEST metrics → residual autoencoder (optional
Optuna) → bundle + `manifest.json` + `splits.csv`. With `--hparams manifest.json --splits splits.csv
--no-tune` it refits with the recorded hyper-parameters and split, which takes minutes on a GPU
(`scripts/slurm_train_repro.sh`). `--write-atlas` also embeds the whole count table into
`<out>/atlas/`.

A refit is not bitwise identical to the original Colab run (different torch/CUDA builds), and with
a TEST set of about 120 samples the balanced accuracy of a single fit moves by 0.1 to 0.2 between
seeds. Judge refits by a **seed sweep**, not by one number:

```bash
ecoli_atlas train ... --hparams outputs/models/manifest.json --splits outputs/models/splits.csv --no-tune \
                  --seeds 1,2,3,4,5,6,7,8 --select best --select-on cv --out outputs/models_repro_sweep --write-atlas
```

Every seed is fitted on the same split and saved under `<out>/seed_<s>/`; the selected one
(`best` = highest mean AUROC, `median`, or `first`) is promoted to `<out>/`, and its
`manifest.json` gains a `seed_sweep` block with the per-seed CV and TEST metrics and their min /
max / mean / sd (also in `<out>/seed_sweep.tsv`). Seeds are ranked on the study-level
out-of-fold (CV) AUROC of the training pool (`--select-on cv`, the default), so TEST stays an
unbiased check of the promoted model; `--select-on test` exists only to reproduce older sweeps.
Re-running with `--resume` re-selects without retraining. `verify-models --old <original> --new <out>` then
reports `old_within_new_seed_spread`: the original bundle is considered reproduced when its TEST
metrics fall inside the sweep's range and prediction agreement is above 95 %. Exact agreement (max
deviation < 1e-4) is expected only when re-loading the same weights.

`--write-atlas` writes to `<out>/atlas` unless `--atlas-out` is given; it never touches an existing
`outputs/atlas`.

## Reproducing the paper figures

`figures` rebuilds every figure of the training notebook from a saved bundle and the training
tables, without retraining the model: the recorded split (`splits.csv`) is re-applied, the input is
z-scored with the bundle's saved statistics, and the encoders embed every sample. Only
`threshold_curves` refits the three fold models per factor (about half a minute on 8 CPU threads);
their out-of-fold scores are cached in `<out>/oof_scores.csv`.

```bash
ecoli_atlas figures --list
ecoli_atlas figures --counts atlas_data/merged_count_table.txt --metadata atlas_data/metadata_revised.csv \
    --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt --models outputs/models \
    --atlas outputs/atlas --reference-sweep outputs/models_repro_sweep/manifest.json \
    --out outputs/figures_repro --device cpu --threads 8 --strict
```

| Figure name | Files (pdf + png unless noted) | Notebook |
|---|---|---|
| `corpus_composition` | `fig_corpus_composition` | cell 11 |
| `factor_test_eval` | `fig_factor_test_eval`, `growth_phase_misclassified.tsv` | cells 29, 35 |
| `split_balance` | `fig_split_balance` | cell 14 |
| `pca_batch` | `fig_pca_mg1655_batch`, `fig_pca_bw_batch`, `fig_pca_strain_all_complex`, `pca_logcpm_coords.csv` | cell 16 |
| `batch_genes` | `fig_batch_genes`, `batch_genes.tsv` | cell 17 |
| `factor_umaps` | `fig_factor_umaps`, `fig_factor_umaps_overlay`, `umap_factor_coords.csv` | cell 27 |
| `factor_gene_correlates` | `fig_factor_gene_correlates`, `factor_gene_correlations.txt` (GSEA input), `.tsv` | cells 31, 33 |
| `study_leakage` | `fig_study_leakage` (`--recombat` adds the transductive reComBat bar if the package is installed) | cells 43, 44 |
| `threshold_curves` | `fig_threshold_curves`, `oof_scores.csv` (skipped with `--no-retrain`) | cell 25 |

Every number behind the figures goes to `<out>/figure_metrics.json`, together with checks against
the bundle's `manifest.json`: TEST sample counts must match exactly and TEST AUROC / balanced
accuracy within 0.001 / 0.01 (same weights, so they agree to all digits); the regenerated embedding
must match `--atlas` within 1e-3; the final embedding must mix studies better than the z-scored
input. The out-of-fold refits of `threshold_curves` depend on the seed and hardware, so they are
accepted when they fall inside the CV spread of a seed sweep given with `--reference-sweep`
(otherwise within 0.03 of the manifest). `--strict` turns a failed check into exit status 1. Figures
run independently; a failure is recorded in the sidecar and the others still run. The command
refuses to write into the bundle, `<models>/../figures` or `<models>/../atlas`.

### Case studies

```bash
ecoli_atlas casestudy all --counts atlas_data/merged_count_table.txt --metadata atlas_data/metadata_revised.csv \
    --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt --models outputs/models --out outputs/figures_repro
ecoli_atlas casestudy glu-cas --counts "raw_data/Contreras Paper/Paper1_all_samples_counts.tsv" \
    --case-metadata "raw_data/Contreras Paper/metadata_paper1.csv" --models outputs/models --out outputs/figures_repro
```

- `uti` (GSE128997): UTI vs urine paired within patient on the full embedding; the GSM list and
  titles are built in (`casestudy/uti.py`) because the metadata carries no condition. Writes
  `fig_uti_axis`, `uti_axis_genes.tsv`, `uti_samples.tsv`, `uti_metrics.json`.
- `oxygen` (GSE189154, GSE78756; `--oxygen-studies` to change): one LDA axis across the two withheld
  oxygen studies, per-study AUROC, genes along the axis. Writes `fig_oxygen_lda_projection`,
  `fig_oxygen_axis_genes`, `oxygen_axis_genes.tsv`, `oxygen_samples.tsv`, `oxygen_metrics.json`.
  Oxygen spellings the notebook did not recognise (e.g. the 376 "aeorbic" rows) stay unlabelled and
  are reported.
- `glu-cas`: embeds an inference count table with the bundle and contrasts glucose vs casamino acids,
  paired within (BioProject, timepoint) and with one LDA on all samples. The metadata columns
  `treatment` ("GLU condition. timepoint S2") and `BioProject` are those of `metadata_paper1.csv`.
  Writes `fig_glu_cas_axis`, `fig_glu_cas_lda_projection`, `fig_glu_cas_axis_genes`,
  `glu_cas_gene_correlations.csv`, `glu_cas_lda_scores.csv`, `glu_cas_metrics.json`.

All case studies assert that their samples are not in TRAIN; sequencing runs of one GSM are averaged.
The numbers (leave-one-out hits, sign / Wilcoxon / permutation p-values, AUROCs, top genes) are
collected in `<out>/casestudy_metrics.json`.

## Where do whole studies sit? `neighbors`

```bash
ecoli_atlas neighbors GSE189154 GSE78756 --atlas outputs/atlas --models outputs/models --out work/neighbors_oxygen --umap
```

For every sample of the given GEO studies (which must already be in the atlas) this reports the `-k`
nearest atlas samples in latent space. Same-study neighbours are excluded by default
(`--within-study` keeps them) and the SRR runs of one GSM are averaged into one row
(`--no-collapse-runs` keeps run-level rows). `--models` adds each sample's split bucket:
`test`, `fold1..K`, `train(aug)` (pool samples without a fold), `withheld` (the proof-of-concept
study), `projected_only` (in the table but never fitted on), `no_metadata`.

- `neighbors.tsv`: one row per (query sample, rank) with the neighbour's metadata, split bucket and
  encoder predictions.
- `summary.tsv`: one row per query sample: nearest sample / distance / study, dominant neighbour
  study and `top_studies`, per factor the kNN vote, its agreement with the sample's own metadata
  (`<f>_agree_meta`) and with the encoder prediction (`<f>_agree_pred`), per-block neighbour
  distances, novelty, UMAP coordinates with `--umap`.
- `study_summary.tsv`: one row per GSE (split buckets, top neighbour studies, dominant votes and
  agreement fractions, median nearest distance, mean novelty percentile).
- `neighbors_manifest.json`: parameters and which studies were found; a GSE missing from the atlas
  stops the command (exit 1) unless `--allow-missing`.

## Module map

| Module | Role |
|---|---|
| `io.py`, `prepare.py` | read count tables (all inGEST flavours), metadata, feature table; `prepare` |
| `preprocess.py` | gene filtering/renaming, CPM, z-score, alignment + coverage report |
| `labels.py`, `splits.py`, `data.py` | metadata → labels, study-level splits, `TrainingData` |
| `models.py`, `losses.py` | `FactorEncoder`, `ResidualAE`, class weights, push / pull losses |
| `train_factor.py`, `train_residual.py`, `tune.py`, `train.py` | training, CV objectives, Optuna, orchestration |
| `artifacts.py` | save / load a model bundle (compatible with the original notebook files) |
| `embed.py`, `compare.py`, `compare_plots.py`, `verify.py` | inference, atlas comparison (+ group figures), bundle comparison |
| `figures/` | `context.py` rebuilds the notebook state from a bundle; one module per figure; `checks.py`, `run.py` |
| `casestudy/` | `axis.py` (LDA / leave-one-group-out / permutation), `uti.py`, `oxygen.py`, `glu_cas.py` |
| `neighbors.py` | whole-study nearest-neighbour report on the atlas |
