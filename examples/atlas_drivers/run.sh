#!/bin/bash
# The genes that move each call of the atlas model (ecoli_atlas drivers, all seeds of the bundle), their GO terms and
# the curated modules (regulons, iModulons, KEGG) they belong to.
#   conda activate ecoli_atlas; bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
   --qc $DATA/sample_qc.tsv --time-series $DATA/time_courses.tsv
   --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
# curated modules (regulons, iModulons, KEGG) when their table is there: `ecoli_atlas download modules` makes it
MOD=$DATA/modules/modules.tsv
if [ -f "$MOD" ]; then MODULES="--modules $MOD"; else MODULES=""; echo "no $MOD (ecoli_atlas download modules): drivers without modules" >&2; fi
for head in phase medium lab_strain; do
  $ROOT/bin/ecoli_atlas drivers $T --models "$MODELS" --head $head --gsea $MODULES --out drivers/$head --formats pdf,png
done
