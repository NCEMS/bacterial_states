#!/bin/bash
# GSE67402 growth and starvation time course: reads -> counts (inGEST), then counts -> embedding, calls and axes (atlas).
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
  $ROOT/bin/ecoli_atlas view "${V[@]}" --embeddings atlas=GSE67402_inGEST --overlay-metadata samples.csv --overlay-series replicate:time_h --out atlas/atlas_view.html
}
case "${1:-}" in
  ingest)
    $ROOT/bin/ecoli_ingest check
    $ROOT/bin/ecoli_ingest run --cores "${CORES:-44}" ;;
  atlas)
    $ROOT/bin/ecoli_atlas embed --ingest-results . --models "$MODELS" --fill-missing-gc study --seeds --out atlas
    # the study's samples on dimensions fitted on the atlas without the study (--exclude-study); --seeds: the range over the 8 seeds
    $ROOT/bin/ecoli_atlas axes --models "$MODELS" --metadata $DATA/metadata_v2.tsv \
      --query atlas --query-study GSE67402 --exclude-study GSE67402 \
      --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
      --holdout lag --holdout late_stationary --holdout long_term_stationary \
      --axis medium:medium_class:complex,defined_rich,minimal_plus_hydrolysate,defined_minimal \
      --where treated=false --where standard_rnaseq=true --series samples.csv:replicate:time_h --seeds --out atlas/axes
    view_page ;;
  view) view_page ;;
  *) echo "usage: bash run.sh ingest|atlas|view" >&2; exit 2 ;;
esac
