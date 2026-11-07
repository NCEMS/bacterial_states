# CLAUDE.md — working in `bacterial_states`

## What this repo is

Two independently installable packages in one repo (GitHub `NCEMS/bacterial_states`):

- **`ecoli_ingest/`** — Ecoli_inGEST. A Snakemake RNA-seq workflow (fastp → FastQC → Centrifuge →
  `vg giraffe` → featureCounts → DESeq2 → MultiQC) shipped as package data inside a small Python
  package (dependency: pyyaml). The `ecoli_ingest` CLI runs the workflow **in the current
  directory**: `init` writes `config.yaml`, `check` validates tools/resources/reads, `run` executes.
  Two execution paths: the `ecoli_ingest` conda env (`environment.yaml`, primary) or the Apptainer
  image (`--container always`); `auto` picks based on whether `snakemake` and `vg` are on PATH.
- **`ecoli_atlas/`** — Ecoli_ATLAS. Pure-Python (torch, sklearn, pandas) encoder–decoder that
  embeds samples and compares them with the atlas. CLI: `prepare`, `train`, `embed`, `compare`,
  `verify-models`, `download`; `figures` and `casestudy` are stubs for round 2.
- `bin/ecoli_ingest`, `bin/ecoli_atlas` run either CLI from the checkout without `pip install`.

## Data lives outside git — hard rules

Never `git add` any of these (all in `.gitignore`): `resources/`, `old_resources/`, `atlas_data/`,
`raw_data/`, `data/`, `outputs/`, `work/`, `notebooks/`, and any `*.pt *.sif *.gbz *.cf *.fastq*
*.bam *.ipynb`. Specifically:

| Local path | What | Status |
|---|---|---|
| `resources/{vg,annotation,centrifuge,container}` | inGEST reference bundle (13 GB) | downloaded; `resources/container/rnaseq_vg.sif` is the image |
| `atlas_data/` | `merged_count_table.txt`, `metadata_revised.csv`, feature table | atlas training inputs |
| `outputs/models/`, `outputs/atlas/` | the Colab-trained bundle (+ `manifest.json`, `splits.csv`) and the atlas built from it | **the originals: never overwrite, commit or redistribute.** Refits go to `outputs/models_repro*/` via `ecoli_atlas train --seeds ...` (see `ecoli_atlas/scripts/slurm_train_repro.sh`); accept a refit when the original's TEST metrics lie within the sweep's spread (`verify-models` → `old_within_new_seed_spread`) |
| `outputs/models_repro/`, `outputs/atlas_repro/` | first GPU refit (seed 77) and its atlas | kept for comparison |
| `outputs/predictions/final_embedding.csv` | baseline atlas embeddings (3342 samples) | source of the future `outputs/atlas/` bundle |
| `work/experiment3/` | the tutorial run (4 samples) used for verification | a run directory; results in `experiment3_results/` |
| `notebooks/` | the two original Colab notebooks | read-only reference; never edit or commit |
| `raw_data/` | original Drive dump incl. unused files and the Contreras paper data | `Contreras Paper/Paper1_all_samples_counts.tsv` is the inference test case |

Download URLs are placeholders (`TBD`) in exactly two files:
`ecoli_ingest/src/ecoli_ingest/resources.py` and `ecoli_atlas/src/ecoli_atlas/download.py`.

## Git

- **Never commit or push without asking the user first.** Prepare, show the file list, wait.
- Use `git mv` for moves so history follows.

## Environments and running things

```bash
# host (login node has no GPU; training goes through SLURM)
conda activate ecoli_ingest        # snakemake + all workflow tools
conda activate ecoli_atlas         # torch (CPU), sklearn, pandas, optuna, umap
export ECOLI_INGEST_APPTAINER=/mnt/gs21/scratch/jbarrick/ncems/build/apptainer_local/bin/apptainer  # unprivileged apptainer

pytest ecoli_ingest/tests ecoli_atlas/tests
cd work/experiment3 && ecoli_ingest check && ecoli_ingest run -n        # tutorial run dir
ecoli_atlas embed --counts "raw_data/Contreras Paper/Paper1_all_samples_counts.tsv" --models outputs/models \
    --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt --out /tmp/x   # parity check vs outputs/predictions_paper1
```

Temporary files go in the session scratchpad or under `work/`, never in the package directories.

## Conventions

- The Snakefile must stay runnable with `--directory <rundir> --configfile <rundir>/config.yaml`
  and `--config resources_dir=…`; resource defaults are in the Snakefile header and mirrored in
  `ecoli_ingest/paths.py` (`RESOURCE_FILES`). Keep them in sync.
- Workflow tool list has one source of truth: `ecoli_ingest/environment.yaml`
  (`build/Singularity.def` installs it). `workflow/envs/rnaseq_vg.yaml` is the legacy image env.
- `annotate_counts` is offline (`workflow/scripts/annotate_counts.py`). `Gene_Summary` holds the
  RefSeq `product`, not the old NCBI gene summary paragraph.
- Gene identifiers: inGEST emits gene **symbols** (GFF `Name`); the training table uses **b-numbers**.
  `ecoli_atlas` maps b-numbers → symbols with the feature table and reports coverage
  (`gene_coverage.tsv`); `embed` fails below `--min-gene-coverage` (0.9).
- Model bundle file layout must stay loadable for the original Colab files (`artifacts.py`).
  New bundles store tensors only (`torch.load(weights_only=True)` works).
- Class calls default to argmax (that is what produced `outputs/predictions*`); the saved operating
  thresholds are opt-in (`--apply-thresholds`).
- Sample ids: inGEST count columns are BAM paths; `io.sample_id_from_column` strips them to the
  config sample name.
- Argparse only (no click/typer); keep both packages free of each other's imports so they can be
  split into separate repos / bioconda recipes later.

## Round-2 backlog

Publication figures (`ecoli_atlas/figures/`), case studies (`casestudy/`: UTI, oxygen, GLU/CAS),
reComBat comparison, GSEA, `atlas_build/` cleanup (`combine_counts.sh` assumes the old column
layout), container rebuild from `environment.yaml` (drop entrez-direct/pytorch), LICENSE file,
tutorial-data bundle, real hosting URLs, a Snakefile rule that emits an atlas-ready counts TSV.
