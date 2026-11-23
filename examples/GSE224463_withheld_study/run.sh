#!/bin/bash
# Where a study the model never saw sits in the atlas (ecoli_atlas neighbors): GSE224463 is withheld from training and
# from the split (projected only), so its samples are placed by the model like new data.
#   conda activate ecoli_atlas; bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
$ROOT/bin/ecoli_atlas neighbors GSE224463 --models "$MODELS" --metadata $DATA/metadata_v2.tsv --out neighbors --umap
