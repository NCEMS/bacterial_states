#!/bin/bash --login
#SBATCH --job-name=atlas_ingest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# Run one batch of the atlas recount (see make_batches.py) with inGEST.
#
#   sbatch --array=1-N%30 --output=$OUT/logs/%x_%A_%a.log slurm_batch.sh $OUT std.list     # array task = line of the list
#   sbatch --time=H:00:00 --output=$OUT/logs/%x_%j.log slurm_batch.sh $OUT <batch>          # one batch by name
#   bash slurm_batch.sh $OUT <batch>                                                        # without SLURM
#
# OUT is the directory make_batches.py wrote (absolute path). The run directory is
# $RUN_ROOT/<batch> (reads and intermediates; put it on scratch). When the batch has finished its
# results are copied to $OUT/results/<batch>/, $OUT/done/<batch> is written and the run directory is
# deleted. A batch that failed or ran out of time keeps its run directory; submitting it again
# continues where it stopped. Batches with a done marker are skipped.
set -euo pipefail

OUT=${1:?directory written by make_batches.py}
WHAT=${2:?batch name or list file}
REPO=${REPO:-$(cd "$OUT/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:?directory for run directories (scratch)}
CORES=${SLURM_CPUS_PER_TASK:-${CORES:-24}}
MAX_DOWNLOADS=${MAX_DOWNLOADS:-2}

if [[ -f "$OUT/$WHAT" ]]; then
  BATCH=$(sed -n "${SLURM_ARRAY_TASK_ID:?list given without an array task}p" "$OUT/$WHAT")
else
  BATCH=$WHAT
fi
[[ -n "$BATCH" && -f "$OUT/batches/$BATCH/config.yaml" ]] || { echo "no such batch: $BATCH" >&2; exit 2; }
if [[ -e "$OUT/done/$BATCH" ]]; then echo "$BATCH: already done"; exit 0; fi

if [[ "$(type -t conda)" != function ]]; then   # not a login shell (`bash slurm_batch.sh`)
  source "$(${CONDA_EXE:-conda} info --base)/etc/profile.d/conda.sh"
fi
set +u; conda activate ecoli_ingest; set -u
export ECOLI_INGEST_RESOURCES=${ECOLI_INGEST_RESOURCES:-$REPO/resources}
INGEST="$REPO/bin/ecoli_ingest"
# vg giraffe maps a writable distance index as a shared writable file. On a network file system
# that stalls every alignment as soon as several nodes align at once; a read-only file is mapped
# read-only.
chmod a-w "$ECOLI_INGEST_RESOURCES"/vg/*.dist 2>/dev/null || true
RUN="$RUN_ROOT/$BATCH"
mkdir -p "$RUN/sra/manifest" "$OUT/results" "$OUT/done" "$OUT/failed"
cp "$OUT/batches/$BATCH/config.yaml" "$RUN/config.yaml"
cp "$OUT/batches/$BATCH"/manifest/*.json "$RUN/sra/manifest/"
cd "$RUN"

# A killed job (scancel, time limit, node failure) leaves partly written outputs, which Snakemake
# took for finished files on resume in spite of --rerun-incomplete. Remove them and what was made
# from them before continuing.
python "$REPO/ecoli_ingest/atlas_build/full_run/clear_incomplete.py" .
rm -f results/temp-core-*.tmp   # paired-read temporary files of a killed featureCounts

# array tasks start together: spread the first requests to the SRA
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then sleep $(( RANDOM % 120 )); fi

echo "== $BATCH on $(hostname), $CORES cores, $(date '+%F %T'), commit $(git -C "$REPO" rev-parse --short HEAD)"
"$INGEST" run --container never --skip-checks --cores "$CORES" -- --unlock >/dev/null 2>&1 || true
status=0
"$INGEST" run --container never --skip-checks --cores "$CORES" --max-downloads "$MAX_DOWNLOADS" \
    -- --rerun-incomplete --keep-going > run.log 2>&1 || status=$?

if [[ $status -ne 0 ]]; then
  echo "== $BATCH FAILED (exit $status), run directory kept: $RUN"
  cp run.log "$OUT/failed/$BATCH.log"
  grep -E "^Error in rule|^(MissingOutput|Workflow|Accession|Download)[A-Za-z]*(Error|Exception)" run.log | sort | uniq -c | head -20 || true
  exit $status
fi

DEST="$OUT/results/$BATCH"
mkdir -p "$DEST"
# Reads, alignments and Centrifuge's per-read table are intermediates. A run that was continued
# after a failure can still hold them (Snakemake does not always remove temp() files then); they
# are not results and are left behind.
rsync -a --delete --delete-excluded \
    --exclude='/*/raw/' --exclude='/*/*.fastq.gz' --exclude='/*/centrifuge/initial.tsv' \
    --exclude='/*/centrifuge/tmp/' --exclude='/*/vg/*.gam' --exclude='/*/vg/*.bam' \
    --exclude='/temp-core-*' results/ "$DEST/results/"
rsync -a checkpoints/qc_failures.json config.yaml run.log "$DEST/"
[[ -d benchmarks ]] && rsync -a benchmarks/ "$DEST/benchmarks/"
rm -f "$OUT/failed/$BATCH.log"
date '+%F %T' > "$OUT/done/$BATCH"
cd "$RUN_ROOT" && rm -rf "$RUN"
echo "== $BATCH done $(date '+%F %T'), results in $DEST"
