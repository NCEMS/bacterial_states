#!/bin/bash
# Genotypes of the LB-exponential lab-strain samples of atlas_lab_strain_LB_variation (fig_batch_tracking) from their
# reads, and how they go with the samples' distance from the regulon-vs-ribosome trend lines.
#
# Quick (minutes): figures from the atlas tables and the tracked per-sample states (sample_states.tsv)
#   conda activate ecoli_atlas; bash run.sh figures            (MODELS=<bundle> for another bundle)
# Full (a day or more, several hundred GB in $SCR): reads -> breseq -> loci -> tables -> figures; rewrites
# sample_states.tsv
#   SCR=<big disk> bash run.sh all [N]           N samples at a time (default 6; each breseq run takes 16 threads; the
#                                                largest libraries need ~100 GB of disk and 5-8 hours)
# or step by step: reference, fetch [N], breseq [N], loci [N], collect, figures.
# The full path needs python with ecoli_atlas (pandas, scipy, matplotlib), sra-tools, fastp, samtools (the
# ecoli_ingest environment), bowtie2, R and breseq 0.40 on PATH.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../..
MODELS=${MODELS:-$ROOT/outputs/models}
export SCR=${SCR:-$PWD/scratch}
PY=${PY:-python}
N=${2:-6}
GSMS() { cut -f1 reads.tsv | tail -n +2; }
need() { for t in "$@"; do command -v "$t" >/dev/null || { echo "run.sh: $t not on PATH" >&2; exit 1; }; done; }

samples() { [ -s samples.tsv ] || $PY scripts/samples.py --models "$MODELS"; $PY scripts/residuals.py; }
figures() {
  samples
  for s in FlhDC FliA; do $PY scripts/fig_regulon.py --set $s; done
  $PY scripts/fig_is1h.py --set FlhDC --set FliA
  $PY scripts/fig_stock.py
  $PY scripts/fig_grid.py
}
reference() { need samtools bowtie2-build; $PY scripts/make_reference.py; }
fetch() { need fasterq-dump fastp; mkdir -p logs; GSMS | xargs -P "$N" -I{} bash -c \
  'bash scripts/fetch.sh {} 4 > logs/fetch_{}.log 2>&1 || echo "fetch FAILED {}"'; }
breseq() { need breseq bowtie2 R samtools; mkdir -p logs; GSMS | xargs -P "$N" -I{} bash -c \
  'bash scripts/breseq.sh {} 16 > logs/breseq_{}.log 2>&1 || echo "breseq FAILED {}"'; }
loci() { need bowtie2 samtools; mkdir -p logs; GSMS | xargs -P "$N" -I{} bash -c \
  'bash scripts/loci_bam.sh {} 8 > logs/loci_{}.log 2>&1 || echo "loci FAILED {}"'; }
collect() { need gdtools samtools bowtie2; samples; $PY scripts/collect.py; }

case "${1:-}" in
  figures) figures ;;
  reference) reference ;;
  fetch) fetch ;;
  breseq) breseq ;;
  loci) loci ;;
  collect) collect ;;
  all) reference; fetch; breseq; loci; collect; figures ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
