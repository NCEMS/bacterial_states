#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <featureCounts_output_file>"
    exit 1
fi

INPUT="$1"
TEMP_GENE_IDS="temp_cleaned_gene_ids.txt"
TEMP_LOOKUP="temp_gene_lookup.tsv"

HEADER_LINE=$(grep -v '^#' "$INPUT" | head -n 1)

grep -v '^#' "$INPUT" | tail -n +2 | \
awk -F'\t' '{
    split($1, a, ",");
    for (i in a) {
        if (a[i] ~ /^GeneID:/) {
            split(a[i], b, ":");
            print b[2];
        }
    }
}' | sort -u > "$TEMP_GENE_IDS"

if [ ! -s "$TEMP_GENE_IDS" ]; then
    echo "No valid GeneIDs found."
    rm -f "$TEMP_GENE_IDS"
    exit 0
fi

ID_LIST=$(paste -sd, "$TEMP_GENE_IDS")

esummary -db gene -id "$ID_LIST" 2>/dev/null | \
grep -vE '^(WARNING|ERROR|nquire|SECOND ATTEMPT|LAST ATTEMPT|QUERY FAILURE|<error>)' | \
xtract -pattern DocumentSummary -element Id Name Summary | \
awk -F'\t' 'BEGIN { OFS="\t" } {
    gsub(/[\r\n\t]+/, " ", $3);
    sub(/\[More information.*$/, "", $3);
    gsub(/^ +| +$/, "", $3);
    print $1, $2, $3;
}' > "$TEMP_LOOKUP"

grep -v '^#' "$INPUT" | awk -F'\t' -v OFS='\t' -v lookup="$TEMP_LOOKUP" -v header="$HEADER_LINE" '
BEGIN {
    print "Gene_Symbol", "Gene_IDs_Field", "Gene_Summary", substr(header, index(header, "\t") + 1);
    while ((getline < lookup) > 0) {
        meta[$1] = $2 "\t" $3;
    }
}
FNR == 1 { next }
{
    gene_ids_field = $1;
    n = split(gene_ids_field, parts, ",");
    gene_id = "NA";
    for (i = 1; i <= n; i++) {
        if (parts[i] ~ /^GeneID:/) {
            split(parts[i], x, ":");
            gene_id = x[2];
            break;
        }
    }

    if (gene_id in meta) {
        split(meta[gene_id], info, "\t");
        gene_symbol = info[1];
        summary = info[2];
    } else {
        gene_symbol = gene_ids_field;
        summary = "NA";
    }

    line = "";
    for (j = 2; j <= NF; j++) {
        line = line OFS $j;
    }

    print gene_symbol, gene_ids_field, summary, substr(line, 2);
}
'

rm -f "$TEMP_GENE_IDS" "$TEMP_LOOKUP"
