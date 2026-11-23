#!/bin/bash
# GSE224463, wild type and rpoS deletion through three stress time courses (stationary-phase entry, cold, high
# osmolarity), a study withheld from the atlas model: its atlas neighbours, then its calls and its place on the
# growth-phase dimension along each time course, over the bundle's seeds.
#   conda activate ecoli_atlas; bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
EA=$ROOT/bin/ecoli_atlas
$EA neighbors GSE224463 --models "$MODELS" --metadata $DATA/metadata_v2.tsv --out neighbors --umap
# the study's samples (embedding and calls of the bundle and of every seed) as an `embed` output
$EA extract GSE224463 --models "$MODELS" --seeds --out atlas
# its nearest atlas samples of other studies, pooled per time course and genotype (group_neighbors.tsv)
$EA compare --embeddings atlas --models "$MODELS" --atlas-metadata $DATA/metadata_v2.tsv --exclude-study GSE224463 \
  --metadata samples.csv --group-by panel --out atlas/compared
# on the growth-phase dimension fitted without the study, one panel per time course and genotype
$EA axes --models "$MODELS" --metadata $DATA/metadata_v2.tsv \
  --query atlas --query-study GSE224463 --exclude-study GSE224463 \
  --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
  --holdout lag --holdout late_stationary --holdout long_term_stationary \
  --where treated=false --where standard_rnaseq=true \
  --series samples.csv:replicate:time_min:panel --seeds --out atlas/axes
