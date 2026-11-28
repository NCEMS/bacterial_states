#!/bin/bash
# Reads of one sample of reads.tsv: up to MAX_READS reads (default 20 M, split over the sample's runs) with sra-tools,
# the ENA FASTQ files as fallback (ecoli_ingest's sra_reads.py), trimmed with fastp (--detect_adapter_for_pe for
# paired libraries; all mates are used as single reads later) into $SCR/trim/<GSM>/.
#   bash scripts/fetch.sh GSM [THREADS]        (needs sra-tools and fastp: the ecoli_ingest environment)
set -euo pipefail
GSM=$1; THREADS=${2:-4}
EX=$(cd "$(dirname "$0")/.." && pwd)
SCR=${SCR:-$EX/scratch}
MAX_READS=${MAX_READS:-20000000}
SRA=$EX/../../ecoli_ingest/src/ecoli_ingest/workflow/scripts/sra_reads.py
line=$(awk -F'\t' -v g="$GSM" '$1 == g' "$EX/reads.tsv")
[ -n "$line" ] || { echo "$GSM not in reads.tsv" >&2; exit 1; }
LAYOUT=$(cut -f3 <<<"$line"); RUNS=$(cut -f4 <<<"$line"); ENA=$(cut -f6 <<<"$line")
T=$SCR/trim/$GSM
[ -e "$T/done" ] && { echo "$GSM fetched"; exit 0; }
R0=$SCR/raw/$GSM; rm -rf "$R0" "$T"; mkdir -p "$R0" "$T"
IFS=, read -ra R <<<"$RUNS"
PER=$(( MAX_READS / ${#R[@]} ))
MATES=1; [ "$LAYOUT" = PAIRED ] && MATES=1,2
for run in "${R[@]}"; do
  ena=$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])[sys.argv[2]]))' "$ENA" "$run")
  python3 "$SRA" fetch --run "$run" --out-dir "$R0" --mates "$MATES" --method auto --ena "$ena" --max-spots "$PER"
  if [ "$LAYOUT" = PAIRED ]; then
    fastp -w "$THREADS" --detect_adapter_for_pe -l 25 -i "$R0/${run}_1.fastq" -I "$R0/${run}_2.fastq" \
      -o "$T/${run}_1.fq.gz" -O "$T/${run}_2.fq.gz" -j "$T/$run.fastp.json" -h /dev/null 2> "$T/$run.fastp.log"
  else
    fastp -w "$THREADS" -l 25 -i "$R0/${run}_1.fastq" -o "$T/${run}_1.fq.gz" -j "$T/$run.fastp.json" -h /dev/null \
      2> "$T/$run.fastp.log"
  fi
  rm -f "$R0/${run}"_*.fastq
done
rmdir "$R0"; echo "$LAYOUT" > "$T/done"; echo "$GSM fetched"
