#!/bin/bash
# Variation between LB studies of the laboratory strains (ecoli_atlas figures, raw expression): PCA and the most
# study-variable genes of MG1655 and BW25113, gene-set enrichment of the between-study ranking of every lab strain
# (GO, and RegulonDB / KEGG with the module table), and the top regulons against the growth rate of each study.
#   conda activate ecoli_atlas; bash run.sh            (MODELS=<bundle> bash run.sh for another bundle)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
   --qc $DATA/sample_qc.tsv
   --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
MOD=$DATA/modules/modules.tsv
if [ -f "$MOD" ]; then MODULES="--modules $MOD"; else MODULES=""; echo "no $MOD (ecoli_atlas download modules): GO terms only" >&2; fi
$ROOT/bin/ecoli_atlas figures $T --models "$MODELS" $MODULES --only pca_batch,batch_genes,gsea,batch_tracking --out figures --strict --threads 16
