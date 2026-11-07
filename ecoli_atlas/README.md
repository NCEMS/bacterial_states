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

```bash
# 1. inGEST results -> atlas input format (counts.tsv genes x samples, samples.tsv)
ecoli_atlas prepare --ingest-results my_run/tutorial_results --out prepared [--models outputs/models]

# 2. embed new samples -> final_embedding.csv, sample_predictions.csv, embedding_block_layout.csv, gene_coverage.tsv
ecoli_atlas embed --counts prepared/counts.tsv --models outputs/models --out embedded \
                  --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt

# 3. compare with the atlas -> neighbors.tsv, summary.tsv [, umap_atlas.csv, umap_query.csv, umap.png]
ecoli_atlas compare --embeddings embedded/final_embedding.csv --atlas outputs/atlas --out compared -k 15 [--umap]

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
                  --seeds 1,2,3,4,5,6,7,8 --select best --out outputs/models_repro --write-atlas
```

Every seed is fitted on the same split and saved under `<out>/seed_<s>/`; the selected one
(`best` = highest mean TEST AUROC, `median`, or `first`) is promoted to `<out>/`, and its
`manifest.json` gains a `seed_sweep` block with the per-seed TEST metrics and their min / max /
mean / sd (also in `<out>/seed_sweep.tsv`). `verify-models --old <original> --new <out>` then
reports `old_within_new_seed_spread`: the original bundle is considered reproduced when its TEST
metrics fall inside the sweep's range and prediction agreement is above 95 %. Exact agreement (max
deviation < 1e-4) is expected only when re-loading the same weights.

`--write-atlas` writes to `<out>/atlas` unless `--atlas-out` is given; it never touches an existing
`outputs/atlas`.

## Module map

| Module | Role |
|---|---|
| `io.py`, `prepare.py` | read count tables (all inGEST flavours), metadata, feature table; `prepare` |
| `preprocess.py` | gene filtering/renaming, CPM, z-score, alignment + coverage report |
| `labels.py`, `splits.py`, `data.py` | metadata → labels, study-level splits, `TrainingData` |
| `models.py`, `losses.py` | `FactorEncoder`, `ResidualAE`, class weights, push / pull losses |
| `train_factor.py`, `train_residual.py`, `tune.py`, `train.py` | training, CV objectives, Optuna, orchestration |
| `artifacts.py` | save / load a model bundle (compatible with the original notebook files) |
| `embed.py`, `compare.py`, `verify.py` | inference, atlas comparison, bundle comparison |
| `figures/`, `casestudy/` | reserved for the publication figures and case studies (round 2) |
