#!/bin/bash --login
#SBATCH --job-name=ecoli_atlas_train
#SBATCH --ntasks=1
#SBATCH --gpus=v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=ecoli_atlas_train_%j.log
# Train the atlas model on the recounted atlas tables and evaluate it on the held-out studies.
#
#   sbatch ecoli_atlas/scripts/slurm_train.sh                       # one model, seed 77
#   SEEDS=1,2,3,4,5,6,7,8 CV=1 sbatch ecoli_atlas/scripts/slurm_train.sh    # seed sweep ranked on out-of-fold scores
#   FIT=one-pass OUT=outputs/sweeps/models_one_pass SPLITS=outputs/models/splits.csv sbatch ecoli_atlas/scripts/slurm_train.sh
#                                                         # another way of fitting, on the same split
#   MASK=0 OUT=outputs/sweeps/models_unmasked ... sbatch ...  # without the missing-gene calls (zeros of absent genes are fitted)
#
# Run from the repository root. Name an x86 GPU type in --gpus (v100s, v100, a100, l40s): the conda
# environment does not run on ARM GPU nodes. The pip build of torch in the `ecoli_atlas` environment
# brings its own CUDA libraries; no CUDA module is loaded.
set -euo pipefail

DATA=${ECOLI_ATLAS_DATA:-atlas_data}
FEATURES=${FEATURES:-resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv}
OUT=${OUT:-outputs/models}
REPO=${REPO:-$PWD}

set +u; conda activate ecoli_atlas; set -u
export PYTHONPATH="$REPO/ecoli_atlas/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

args=(--counts "$DATA/atlas_v2_counts.tsv" --metadata "$DATA/metadata_v2.tsv" --qc "$DATA/sample_qc.tsv"
      --features "$FEATURES" --out "$OUT" --device cuda --threads "${SLURM_CPUS_PER_TASK:-8}")
[[ -f "$DATA/atlas_v2_antisense_counts.tsv" ]] && args+=(--antisense "$DATA/atlas_v2_antisense_counts.tsv")   # veto for the missing-gene calls
SPLITS=${SPLITS:-$OUT/splits.csv}
[[ -f "$SPLITS" ]] && args+=(--splits "$SPLITS")                     # keep the split the figures were made from
[[ -n "${SEEDS:-}" ]] && args+=(--seeds "$SEEDS")
[[ -n "${CV:-}" ]] && args+=(--cv)
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "$EPOCHS")
[[ -n "${PHASE_HEAD:-}" ]] && args+=(--phase-head "$PHASE_HEAD")     # binary (default) or ordinal
[[ "${DNA_DECODER:-0}" == 1 ]] && args+=(--dna-decoder)               # genomic-DNA term in the expected counts (off by default)
[[ "${DNA_INPUTS:-0}" == 1 ]] && args+=(--dna-inputs)                 # expected DNA reads taken off the encoder inputs (off by default)
[[ "${POOL:-1}" == 0 ]] && args+=(--no-pool)                          # POOL=0: missing-gene calls per sample only
[[ "${MASK:-1}" == 0 ]] && args+=(--no-mask-missing)                  # MASK=0: genes a strain lacks stay measured zeros
[[ -n "${FIT:-}" ]] && args+=(--fit "$FIT")                           # two-stage (default), one-pass or unstructured
python -m ecoli_atlas.cli train "${args[@]}"
