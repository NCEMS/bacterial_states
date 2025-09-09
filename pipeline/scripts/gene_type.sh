bed_file="$1"
bam_file="$2"
output_file="$3"
scripts_dir="$4"

bedtools intersect -c -sorted -a "$bed_file" -b "$bam_file" > ${bam_file}_temp
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
