#!/bin/bash
# GSE147611, wild type against greA greB deletion (3 + 3 samples): reads -> counts (inGEST), then counts -> embedding,
# calls and atlas neighbours (atlas). The smallest read-based example: about 2 million read pairs per library.
#   conda activate ecoli_ingest; bash run.sh ingest      # downloads the reads and runs the workflow here
#   conda activate ecoli_atlas;  bash run.sh atlas       # needs the inGEST results of the first step
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
case "${1:-}" in
  ingest)
    $ROOT/bin/ecoli_ingest check
    $ROOT/bin/ecoli_ingest run --cores "${CORES:-8}" ;;
  atlas)
    $ROOT/bin/ecoli_atlas embed --ingest-results . --models "$MODELS" --out atlas/embedded
    $ROOT/bin/ecoli_atlas compare --embeddings atlas/embedded --models "$MODELS" --atlas-metadata $DATA/metadata_v2.tsv \
      --metadata samples.csv --group-by condition --exclude-study GSE147611 --out atlas/compared ;;
  *) echo "usage: bash run.sh ingest|atlas" >&2; exit 2 ;;
esac
