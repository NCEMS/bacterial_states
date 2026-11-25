#!/bin/bash
# GSE147611, wild type against greA greB deletion (3 + 3 samples): reads -> counts (inGEST), then counts -> embedding,
# calls and atlas neighbours (atlas). The smallest read-based example: about 2 million read pairs per library.
#   conda activate ecoli_ingest; bash run.sh ingest      # downloads the reads and runs the workflow here
#   conda activate ecoli_atlas;  bash run.sh atlas       # needs the inGEST results of the first step
#                                bash run.sh view        # only the page atlas/atlas_view.html again
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
view_page() {
  # one HTML page: the atlas in 3-D with these samples in it (open atlas/atlas_view.html in a web browser).
  # Study text, time courses and regulons are added when their tables are there.
  local V=(--models "$MODELS" --metadata $DATA/metadata_v2.tsv --counts $DATA/atlas_v2_counts.tsv --qc $DATA/sample_qc.tsv)
  [ -f $DATA/metadata_v2_studies.tsv ] && V+=(--studies $DATA/metadata_v2_studies.tsv)
  [ -f $DATA/metadata_v2_sample_text.tsv ] && V+=(--sample-text $DATA/metadata_v2_sample_text.tsv)
  [ -f $DATA/time_courses.tsv ] && V+=(--time-series $DATA/time_courses.tsv)
  [ -f $DATA/modules/modules.tsv ] && V+=(--modules $DATA/modules/modules.tsv --overlay-counts results/all_samples_counts.tsv)   # ecoli_atlas download modules
  $ROOT/bin/ecoli_atlas view "${V[@]}" --embeddings atlas/embedded=GSE147611_inGEST --overlay-metadata samples.csv --out atlas/atlas_view.html
}
case "${1:-}" in
  ingest)
    $ROOT/bin/ecoli_ingest check
    $ROOT/bin/ecoli_ingest run --cores "${CORES:-8}" ;;
  atlas)
    $ROOT/bin/ecoli_atlas embed --ingest-results . --models "$MODELS" --out atlas/embedded
    $ROOT/bin/ecoli_atlas compare --embeddings atlas/embedded --models "$MODELS" --atlas-metadata $DATA/metadata_v2.tsv \
      --metadata samples.csv --group-by condition --exclude-study GSE147611 --out atlas/compared
    view_page ;;
  view) view_page ;;
  *) echo "usage: bash run.sh ingest|atlas|view" >&2; exit 2 ;;
esac
