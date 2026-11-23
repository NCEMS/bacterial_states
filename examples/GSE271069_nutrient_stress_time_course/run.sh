#!/bin/bash
# GSE271069 nutrient stress in two media: reads -> counts (inGEST), then counts -> embedding and comparison with the atlas,
# and the glucose against casamino acids case study on the atlas tables (the study is a forced TEST study).
#   conda activate ecoli_ingest; bash run.sh ingest      # 16 libraries, 25 GB of reads
#   conda activate ecoli_atlas;  bash run.sh atlas       # needs the inGEST results of the first step
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
case "${1:-}" in
  ingest)
    $ROOT/bin/ecoli_ingest check
    $ROOT/bin/ecoli_ingest run --cores "${CORES:-96}" --max-downloads 6 ;;
  atlas)
    $ROOT/bin/ecoli_atlas embed --ingest-results . --models "$MODELS" --out atlas/embedded
    $ROOT/bin/ecoli_atlas compare --embeddings atlas/embedded --models "$MODELS" --out atlas/compared \
      --metadata samples.csv --group-by condition --order-by order
    T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
       --qc $DATA/sample_qc.tsv --time-series $DATA/time_courses.tsv
       --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
    $ROOT/bin/ecoli_atlas casestudy glu-cas $T --models "$MODELS" --out atlas/casestudy --threads 16 ;;
  *) echo "usage: bash run.sh ingest|atlas" >&2; exit 2 ;;
esac
