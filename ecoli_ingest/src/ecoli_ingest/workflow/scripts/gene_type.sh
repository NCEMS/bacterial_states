# Percentage of read-feature overlaps per feature type, for the MultiQC "Feature overlap" plot.
#   gene_type.sh <annotation.bed> <sorted.bam> <output.tsv> <scripts_dir> [forward|reverse|unstranded] [SE|PE]
# The BED name column is <group>:<feature id>; group is CDS, ncRNA, tRNA, rRNA, pseudo or Rfam.
# For a strand-specific library only reads on the feature's own strand are counted (mate 1 decides
# for paired-end reads), which is the rule featureCounts uses.
bed_file="$1"
bam_file="$2"
output_file="$3"
scripts_dir="$4"
strand_call="${5:-unstranded}"
layout="${6:-SE}"

case "$strand_call" in
  forward) strand_flag="-s" ;;
  reverse) strand_flag="-S" ;;
  *)       strand_flag="" ;;
esac
mate_filter=""
if [ "$layout" = "PE" ] && [ -n "$strand_flag" ]; then mate_filter="-f 64"; fi

samtools view -b $mate_filter "$bam_file" \
  | bedtools intersect -c -sorted $strand_flag -a "$bed_file" -b stdin > ${bam_file}_temp
cat "$scripts_dir"/fo_header.txt <(
  awk -F'\t' '
    BEGIN { OFS="\t" }
    {
      split($4, parts, ":")
      id = parts[1]
      count = $NF
      total += count
      counts[id] += count
    }
    END {
      for (id in counts) {
        printf "%s\t%.4f\n", id, ((counts[id] + 1) / (total + 1)) * 100
      }
    }
  ' ${bam_file}_temp
) > "$output_file"
rm ${bam_file}_temp
