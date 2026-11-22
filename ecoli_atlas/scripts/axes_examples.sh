#!/bin/bash
# The example axes of the paper (`ecoli_atlas axes`), for any bundle: growth phase, oxygen, carbon source (sugar against
# non-sugar) and glycerol against the other carbon sources. Rerun after every refit of the model.
#
#   MODELS=<bundle directory> bash ecoli_atlas/scripts/axes_examples.sh            # -> outputs/axes_examples
#   MODELS=<bundle directory> OUT=<output directory> bash ecoli_atlas/scripts/axes_examples.sh
#
# Run from the repository root, in the ecoli_atlas environment. One output folder per axis under $OUT (score table,
# summary JSON, distribution figures). All axes use untreated samples of standard RNA-seq; every sample is scored by a
# model that has seen no sample of its study. Categories with too few studies to be fitted are held out and only scored.
set -euo pipefail

MODELS=${MODELS:?set MODELS to the bundle directory}
OUT=${OUT:-outputs/axes_examples}
DATA=${ECOLI_ATLAS_DATA:-atlas_data}
REPO=${REPO:-$PWD}
export PYTHONPATH="$REPO/ecoli_atlas/src${PYTHONPATH:+:$PYTHONPATH}"
axes() { python -m ecoli_atlas.cli axes --models "$MODELS" --metadata "$DATA/metadata_v2.tsv" \
             --where treated=false --where standard_rnaseq=true "$@"; }

# growth phase: fitted from mid exponential to stationary; lag and the late stationary levels only scored (extrapolation)
axes --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
     --holdout lag --holdout late_stationary --holdout long_term_stationary \
     --series "$DATA/time_courses.tsv" --out "$OUT/phase"

# oxygen: fitted on aerobic against anaerobic fermentation; microaerobic (one study, GSE189154, TEST) and anaerobic
# respiration (two studies) are held out and only scored, as an interpolation test
axes --axis oxygen:oxygen:aerobic,microaerobic,anaerobic_respiration,anaerobic_fermentation \
     --holdout microaerobic --holdout anaerobic_respiration --out "$OUT/oxygen"

# carbon source in defined minimal media: sugars against glycerol and organic acids (the value of each sample is kept in
# the score table, so glucose / other sugars / glycerol / organic acids can be compared along the axis)
axes --axis carbon:carbon_source:sugar=glucose+other_sugar,non_sugar=glycerol+organic_acid \
     --where medium_class=defined_minimal --out "$OUT/carbon"

# glycerol against the other stated carbon sources, defined minimal media
axes --axis glycerol:carbon_source:non_glycerol=glucose+other_sugar+organic_acid,glycerol=glycerol \
     --where medium_class=defined_minimal --out "$OUT/glycerol"

echo "wrote the example axes for $MODELS to $OUT"
