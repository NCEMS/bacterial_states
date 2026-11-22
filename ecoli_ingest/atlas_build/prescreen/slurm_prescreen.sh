#!/bin/bash --login
#SBATCH --job-name=ecoli_prescreen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=prescreen_%A_%a.log
# Prescreen one part of the candidate samples per array task. Run from the repository root with the
# ecoli_ingest environment active (fastp, vg, samtools, bedtools, rseqc, centrifuge, sra-tools).
# Each task screens JOBS samples at a time with 2 threads each; finished samples are skipped, so a
# task that ran out of time can simply be submitted again. Without SLURM, run prescreen.py directly.
set -euo pipefail

MANIFEST=${MANIFEST:-work/atlas_manifest}
OUT=${OUT:-work/prescreen}
RESOURCES=${ECOLI_INGEST_RESOURCES:-resources}
JOBS=${JOBS:-$(( ${SLURM_CPUS_PER_TASK:-32} / 2 ))}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}

python ecoli_ingest/atlas_build/prescreen/prescreen.py \
  --manifest "$MANIFEST" --out "$OUT" --resources "$RESOURCES" \
  --scratch "${TMPDIR:-/tmp}/prescreen_$I" --jobs "$JOBS" --threads 2 --shard "$I/$N" "$@"
