#!/bin/bash --login
#SBATCH --job-name=ecoli_atlas_train
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ecoli_atlas_train_%j.log
# Re-fit the atlas model with the hyper-parameters and study split recorded in the original
# manifest, then compare the result with the original bundle. Run from the repository root
# after `conda activate ecoli_atlas` has a CUDA build of torch available.
set -euo pipefail

DATA=${ECOLI_ATLAS_DATA:-atlas_data}
OLD=${ECOLI_ATLAS_MODELS:-outputs/models}
NEW=${NEW_MODELS:-outputs/models_repro_sweep}

SEEDS=${SEEDS:-1,2,3,4,5,6,7,8}

# Sweep several seeds on the recorded split; the best (highest mean TEST AUROC) is promoted to $NEW
# and its manifest.json records the per-seed metrics and their spread (seed_sweep). The atlas built
# from the promoted model goes to $NEW/atlas; the original bundle and atlas are never modified.
ecoli_atlas train \
  --counts "$DATA/merged_count_table.txt" \
  --metadata "$DATA/metadata_revised.csv" \
  --feature-table "$DATA/GCF_000005845.2_ASM584v2_feature_table.txt" \
  --hparams "$OLD/manifest.json" --splits "$OLD/splits.csv" --no-tune \
  --seeds "$SEEDS" --select best --device cuda --out "$NEW" --write-atlas --atlas-out "$NEW/atlas"

ecoli_atlas verify-models --old "$OLD" --new "$NEW" \
  --counts "$DATA/merged_count_table.txt" \
  --metadata "$DATA/metadata_revised.csv" \
  --feature-table "$DATA/GCF_000005845.2_ASM584v2_feature_table.txt" \
  --report "$NEW/verify_vs_original.json"
