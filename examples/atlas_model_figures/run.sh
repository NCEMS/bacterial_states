#!/bin/bash
# Figures of the training data and of the atlas model (ecoli_atlas figures): data composition, TEST evaluation, head
# scores, gene correlates, study leakage, thresholds. The batch structure of the raw expression is the example
# ../atlas_lab_strain_LB_variation.
#   conda activate ecoli_atlas; bash run.sh            (MODELS=<bundle> bash run.sh for another bundle)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
   --qc $DATA/sample_qc.tsv
   --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
$ROOT/bin/ecoli_atlas figures $T --models "$MODELS" --out figures --strict --threads 16 \
  --only split_balance,corpus_composition,head_test_eval,head_scores,head_gene_correlates,study_leakage,threshold_curves
