#!/bin/bash
# One sample's trimmed reads aligned (bowtie2 --local, all mates as single reads) to the regions of reference/loci.fasta
# only (scripts/make_reference.py), so the clipped parts of reads across IS ends are kept. Positions are shifted back
# to the genome: contigs flhDC, crl, oppA, fim -> NC_000913; fiminv (the fim region with the switch inverted) ->
# NC_000913_fimS_inverted. Output $SCR/loci/<GSM>.bam (about a minute per sample on 8 threads).
#   bash scripts/loci_bam.sh GSM [THREADS]     (needs bowtie2 and samtools)
set -euo pipefail
GSM=$1; J=${2:-8}
EX=$(cd "$(dirname "$0")/.." && pwd)
SCR=${SCR:-$EX/scratch}
OUT=$SCR/loci/$GSM.bam
[ -s "$OUT.bai" ] && { echo "$GSM done"; exit 0; }
mkdir -p "$SCR/loci"
FQ=$(ls "$SCR"/trim/"$GSM"/*.fq.gz | paste -sd,)
bowtie2 --local -p "$J" --no-unal --no-hd -x "$EX/reference/loci" -U "$FQ" 2> "${OUT%.bam}.bowtie2.log" \
  | awk 'BEGIN{OFS="\t"} {split($3, c, ":"); $3 = (c[1] == "fiminv") ? "NC_000913_fimS_inverted" : "NC_000913";
         $4 = $4 + c[2]; if ($7 != "*" && $7 != "=") $7 = "*"; if ($8 > 0) $8 = $8 + c[2]; print}' \
  | cat <(printf "@SQ\tSN:NC_000913\tLN:4641652\n@SQ\tSN:NC_000913_fimS_inverted\tLN:4641652\n") - \
  | samtools sort -@ 2 -o "$OUT" -
samtools index "$OUT"
