#!/bin/bash
# breseq of one sample's trimmed reads ($SCR/trim/<GSM>/) against NC_000913.3: the GenBank file of the inGEST core
# bundle (its CONTIG line makes breseq ignore its sequence) and the sequence from reference/NC_000913.fasta (run.sh
# reference). Output in $SCR/breseq/<GSM>/ (output/,
# data/reference.bam; the intermediate steps are deleted when the run succeeds and kept when it stops, so a rerun
# resumes). Settings: consensus mode, targeted (-t: no coverage-distribution fit, which RNA-seq coverage does not
# follow), no junction score cutoff, brief HTML output. Samples with junction_k50 = True in reads.tsv cap the
# alignment of reads to the junction candidates at 50 hits per read (breseq's default is 2000): in those libraries
# most reads hit many candidates and that alignment passed 1 TB; the run directory gets a JUNCTION_K50 file.
#   bash scripts/breseq.sh GSM [THREADS]       (needs breseq 0.40, bowtie2, R, samtools)
set -euo pipefail
GSM=$1; J=${2:-16}
EX=$(cd "$(dirname "$0")/.." && pwd)
SCR=${SCR:-$EX/scratch}
W=$SCR/breseq/$GSM
[ -s "$W/output/output.gd" ] && { echo "$GSM done"; exit 0; }
mkdir -p "$W"
K50=$(awk -F'\t' -v g="$GSM" 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "junction_k50") c = i} $1 == g {print $c}' \
      "$EX/reads.tsv")
EXTRA=()
if [ "$K50" = True ]; then
  EXTRA=(--bowtie2-junction "--local -i S,1,0.25 --score-min L,1,0.70 -k 50")
  echo "--bowtie2-junction \"--local -i S,1,0.25 --score-min L,1,0.70 -k 50\"" > "$W/JUNCTION_K50"
fi
start=$(date +%s)
breseq -j "$J" -n "$GSM" -o "$W" -r "$EX/../../resources/annotation/GCF_000005845.2_ASM584v2_genomic.gbff" \
  -r "$EX/reference/NC_000913.fasta" -t --junction-score-cutoff 0 --brief-html-output \
  "${EXTRA[@]}" "$SCR"/trim/"$GSM"/*.fq.gz >> "$W/breseq.log" 2>&1 && st=ok || st=FAIL
echo "$GSM $st $(( $(date +%s) - start ))s" | tee -a "$SCR/breseq/times.txt"
[ "$st" = ok ] && rm -rf "$W"/0[1-8]_* "$W"/data/*.sam
[ "$st" = ok ]
