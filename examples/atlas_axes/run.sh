#!/bin/bash
# Ordered categories as dimensions of the embedding (ecoli_atlas axes): growth phase, oxygen, carbon source (sugar against
# non-sugar) and glycerol against the other carbon sources, then the genes that move each axis (drivers --axis, all seeds)
# and their GO terms. All axes use untreated samples of standard RNA-seq; every sample is scored by a model that has
# seen no sample of its study. Categories with too few studies to be fitted are held out and only scored.
#   conda activate ecoli_atlas; bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../.. ; DATA=$ROOT/atlas_data
MODELS=${MODELS:-$ROOT/outputs/models}
EA=$ROOT/bin/ecoli_atlas
axes() { $EA axes --models "$MODELS" --metadata $DATA/metadata_v2.tsv --where treated=false --where standard_rnaseq=true "$@"; }

# growth phase: fitted from mid exponential to stationary. Only samples whose phase GEO states (per sample or for the
# whole study), as in the phase call's scoring: phases inferred from sampling times (mostly GSE206047, microplate time
# courses) are left out
axes --axis phase:growth_phase:mid_exponential,transition,stationary \
     --where growth_phase_basis=stated,study_default \
     --series $DATA/time_courses.tsv --out axes/phase
# oxygen: fitted on aerobic against anaerobic fermentation; microaerobic (one study, GSE189154, TEST) and anaerobic
# respiration (two studies) are held out and only scored, as an interpolation test
axes --axis oxygen:oxygen:aerobic,microaerobic,anaerobic_respiration,anaerobic_fermentation \
     --holdout microaerobic --holdout anaerobic_respiration --out axes/oxygen
# carbon source in defined minimal media: sugars against glycerol and organic acids (each sample keeps its own value);
# exponential samples only, so that the stated carbon source is the one the cells use (in stationary phase it is gone)
axes --axis carbon:carbon_source:sugar=glucose+other_sugar,non_sugar=glycerol+organic_acid \
     --where medium_class=defined_minimal --where growth_phase=early_exponential,mid_exponential --out axes/carbon
# glycerol against the other stated carbon sources, defined minimal media
axes --axis glycerol:carbon_source:non_glycerol=glucose+other_sugar+organic_acid,glycerol=glycerol \
     --where medium_class=defined_minimal --out axes/glycerol

# the genes that move each axis between its first and last fitted category, averaged over the seeds of the bundle
T="--counts $DATA/atlas_v2_counts.tsv --antisense $DATA/atlas_v2_antisense_counts.tsv --metadata $DATA/metadata_v2.tsv
   --qc $DATA/sample_qc.tsv --time-series $DATA/time_courses.tsv
   --features $ROOT/resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
contrast() { python3 -c "import json,sys; s=json.load(open(sys.argv[1])); f=[c for c in s['categories'] if c not in s.get('holdout', [])]; print(f[0] + ':' + f[-1])" "$1"; }
# curated modules (regulons, iModulons, KEGG) when their table is there: `ecoli_atlas download modules` makes it
MOD=$DATA/modules/modules.tsv
if [ -f "$MOD" ]; then MODULES="--modules $MOD"; else MODULES=""; echo "no $MOD (ecoli_atlas download modules): drivers without modules" >&2; fi
for name in phase oxygen carbon glycerol; do
  summary=axes/$name/axis_${name}_summary.json
  $EA drivers $T --models "$MODELS" --axis $summary --contrast "$(contrast $summary)" --gsea $MODULES --out drivers/$name --formats pdf,png
done
