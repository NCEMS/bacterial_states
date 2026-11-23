#!/bin/bash
# Figures of the training data and of the atlas model (ecoli_atlas figures): data composition, batch structure of the
# raw expression with its GO enrichment, TEST evaluation, head scores, gene correlates, study leakage, thresholds.
#   conda activate ecoli_atlas; bash run.sh            (MODELS=<bundle> bash run.sh for another bundle)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
   --qc $DATA/sample_qc.tsv
   --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
$ROOT/bin/ecoli_atlas figures $T --models "$MODELS" --out figures --strict --threads 16
