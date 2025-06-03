bam_file="$2"
bed_file="$1"
output_file="$3"

bedtools intersect -c -a "$bed_file" -b "$bam_file" > temp
cat scripts/fo_header.txt <(sed 's/:/\t/g' temp | awk -v OFS='\t' '{print $4,$14}' | awk -F'\t' '{count[$1] += $2; total += $2} END {for (c in count) printf "%s\t%.4f\n", c, (count[c]/total)*100}') > "$output_file"
rm temp

