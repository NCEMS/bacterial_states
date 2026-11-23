#!/bin/bash
# Case study on the atlas embedding (ecoli_atlas casestudy oxygen): aerobic against microaerobic and anaerobic growth (GSE189154, a TEST study).
#   conda activate ecoli_atlas; bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
   --qc $DATA/sample_qc.tsv --time-series $DATA/time_courses.tsv
   --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
$ROOT/bin/ecoli_atlas casestudy oxygen $T --models "$MODELS" --out casestudy --threads 16
