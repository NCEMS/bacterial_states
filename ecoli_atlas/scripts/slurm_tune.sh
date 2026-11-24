#!/bin/bash --login
#SBATCH --job-name=ecoli_atlas_tune
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --gpus=v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=ecoli_atlas_tune_%A_%a.log
# Hyper-parameter search of the atlas model (`ecoli_atlas tune`): an array of workers that share one Optuna study in
# $OUT (study.journal). Each worker runs up to TRIALS trials; all stop once the study holds MAX_TOTAL trials.
#
#   sbatch ecoli_atlas/scripts/slurm_tune.sh                                   # 4 workers, 40 trials each, into work/hpo/study1
#   sbatch --array=0-7 --export=ALL,TRIALS=20 ecoli_atlas/scripts/slurm_tune.sh    # 8 workers
#   sbatch ecoli_atlas/scripts/slurm_tune.sh                                   # again: resumes the study up to MAX_TOTAL
#   OUT=work/hpo/study2 MAX_TOTAL=300 sbatch ecoli_atlas/scripts/slurm_tune.sh     # another study
#
# Run from the repository root. The fixed settings are those of the current model: the antisense veto, the
# time courses and the phase-order term (weight 1); everything else is the default of `train`. A worker stopped by the
# time limit marks its running trial failed. After a hard kill (node failure), give FAIL_RUNNING=1 to one worker, with no
# other worker running, to clear the trials it left running. Name an x86 GPU type in --gpus (not gh200: ARM).
set -euo pipefail

DATA=${ECOLI_ATLAS_DATA:-atlas_data}
FEATURES=${FEATURES:-resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv}
SPLITS=${SPLITS:-outputs/models/splits.csv}
OUT=${OUT:-work/hpo/study1}
TRIALS=${TRIALS:-40}
MAX_TOTAL=${MAX_TOTAL:-160}
REPO=${REPO:-$PWD}

# the training environment (environment-train.yaml, CUDA torch from requirements-cuda.txt); CONDA_ENV= picks another
set +u
if ! conda activate "${CONDA_ENV:-ecoli_atlas_train}" 2>/dev/null; then
  echo "conda env ${CONDA_ENV:-ecoli_atlas_train} not found; using ecoli_atlas" >&2
  conda activate ecoli_atlas
fi
set -u
export PYTHONPATH="$REPO/ecoli_atlas/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, optuna; assert torch.cuda.is_available(), 'CUDA is not available to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'optuna', optuna.__version__)"

args=(--counts "$DATA/atlas_v2_counts.tsv" --antisense "$DATA/atlas_v2_antisense_counts.tsv"
      --metadata "$DATA/metadata_v2.tsv" --qc "$DATA/sample_qc.tsv" --features "$FEATURES"
      --time-series "$DATA/time_courses.tsv" --phase-order-weight "${PHASE_ORDER_WEIGHT:-1}"
      --splits "$SPLITS" --out "$OUT" --trials "$TRIALS" --max-total-trials "$MAX_TOTAL"
      --sampler-seed "${SLURM_ARRAY_TASK_ID:-0}" --device cuda --threads "${SLURM_CPUS_PER_TASK:-8}")
[[ "${FAIL_RUNNING:-0}" == 1 ]] && args+=(--fail-running)
[[ -n "${TRIAL_SEEDS:-}" ]] && args+=(--trial-seeds "$TRIAL_SEEDS")      # e.g. 1,2: each trial is the mean of two seeds
python -m ecoli_atlas.cli tune "${args[@]}"
