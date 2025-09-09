#Script used to combine all the counts files from all the different experiments

TMP_DIR="tmp_counts"
OUTPUT_FILE="all_gse_counts.tsv"

mkdir -p "$TMP_DIR"

#Finding directories
for GSE_DIR in GSE*/; do
    GSE_NAME=$(basename "$GSE_DIR")
    COUNTS_FILE="${GSE_DIR}${GSE_NAME}_results/all_samples_counts_extended.tsv"

    if [[ ! -f "$COUNTS_FILE" ]]; then
        echo "Skipping $GSE_NAME (no counts file found)"
        continue
    fi

    echo "Processing $GSE_NAME..."

    #Getting header
    HEADER=$(head -1 "$COUNTS_FILE" | awk -F'\t' '{
        out=$1
        for(i=9;i<=NF;i++){
            split($i,a,"/")
            split(a[length(a)],b,"_sort")
            out=out "\t" b[1]
        }
        print out
    }')

    #Getting counts
    awk -F'\t' -v OFS='\t' 'NR>1{
        out=$1
        for(i=9;i<=NF;i++){
            out=out "\t" $i
        }
        print out
    }' "$COUNTS_FILE" > "$TMP_DIR/${GSE_NAME}.tsv"

    #Concatenating the header
    sed -i "1i$HEADER" "$TMP_DIR/${GSE_NAME}.tsv"
done

#Loop over files and join them after sorting on gene name
cp "${FILES[0]}" "$OUTPUT_FILE"
for F in "${FILES[@]}"; do
    [[ "$F" == "${FILES[0]}" ]] && continue
    sort -k1,1 "$OUTPUT_FILE" > "$OUTPUT_FILE.sorted"
    sort -k1,1 "$F" > "$TMP_DIR/$(basename "$F").sorted"
    join -t $'\t' -a1 -a2 -e '0' -o auto "$OUTPUT_FILE.sorted" "$TMP_DIR/$(basename "$F").sorted" > "${OUTPUT_FILE}.tmp"
    mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"
done

echo "Complete"

#cut -f 1,6,8 ../../postprocessing_scripts/labeled.txt | awk -v OFS='\t' '{print $1"_"$2,$3}' | grep -v "growth"
