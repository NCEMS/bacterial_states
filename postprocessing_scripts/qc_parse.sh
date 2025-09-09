#Script used to extract QC information from GSEs after they run

OUTPUT="sample_metrics.tsv"
#QC metrics we are looking for
echo -e "sample\tgrowth_phase\tstrandedness_t_ratio\ttRNA_fraction\trRNA_fraction\tcontamination_percent\tpaired_single_end\tGC_bias\tTotal_reads" > "$OUTPUT"

while read SAMPLE GROWTH; do
    echo "Processing $SAMPLE..."

    GSM_PREFIX=$(echo "$SAMPLE" | cut -d'_' -f1)

    #Look for sample directory (have list of total GSE)
    SAMPLE_DIR=$(find . -type d -path "*/GSE*/*_results/${GSM_PREFIX}_*" 2>/dev/null | head -1 || true)

    if [[ -z "$SAMPLE_DIR" ]]; then
        echo "Warning: $SAMPLE directory not found, filling with NAs"
        echo -e "$SAMPLE\t$GROWTH\tNA\tNA\tNA\tNA\tNA\tNA\tNA" >> "$OUTPUT"
        continue
    fi

    #Strandedness ratio
    INFER_FILE=$(find "$SAMPLE_DIR/rseqc" -name "*_infer_experiment.txt" 2>/dev/null | head -1 || true)
    if [[ -f "$INFER_FILE" ]]; then
        STRANDED=$(awk '
            BEGIN{val="NA"}
            /^This is/ {if($3=="SingleEnd") type="Single"; else type="Paired"}
            /Fraction of reads explained/ {
                match($0, /: ([0-9.]+)/, arr)
                if(arr[1]!="") val=arr[1]
            }
            END {printf "%s", val}
        ' "$INFER_FILE")
    else
        STRANDED="NA"
    fi

    #tRNA and rRNA fractions
    FEATURE_FILE=$(find "$SAMPLE_DIR/feature_overlap" -name "*_feature_overlap_mqc.tsv" 2>/dev/null | head -1 || true)
    if [[ -f "$FEATURE_FILE" ]]; then
        tRNA=$(awk '$1=="tRNA"{print $2}' "$FEATURE_FILE")
        rRNA=$(awk '$1=="rRNA"{print $2}' "$FEATURE_FILE")
    else
        tRNA="NA"
        rRNA="NA"
    fi

    #Contamination (non e coli reads)
    CENT_FILE=$(find "$SAMPLE_DIR/centrifuge" -name "*_report.txt" 2>/dev/null | head -1 || true)
    if [[ -f "$CENT_FILE" ]]; then
        E_COLI=$(awk '$6=="Escherichia" && $7=="coli"{print $1}' "$CENT_FILE")
        if [[ -n "$E_COLI" ]]; then
            CONTAM=$(awk -v x="$E_COLI" 'BEGIN{print 100 - x}')
        else
            CONTAM="NA"
        fi
    else
        CONTAM="NA"
    fi

    #Paired versus single end
    if ls "$SAMPLE_DIR/fastqc/"*R2*fastqc.html &> /dev/null; then
        PAIRED="Paired"
    else
        PAIRED="Single"
    fi

    #GC bias and total number of reads
    FASTQC_ZIP=$(find "$SAMPLE_DIR/fastqc" -name "*_clean_R1_fastqc.zip" 2>/dev/null | head -1 || true)
    FASTQC_DIR=$(find "$SAMPLE_DIR/fastqc" -name "*_clean_R1_fastqc" -type d 2>/dev/null | head -1 || true)

    if [[ -z "$FASTQC_DIR" && -n "$FASTQC_ZIP" ]]; then
        unzip -q "$FASTQC_ZIP" -d "$SAMPLE_DIR/fastqc"
        FASTQC_DIR=$(find "$SAMPLE_DIR/fastqc" -name "*_clean_R1_fastqc" -type d 2>/dev/null | head -1 || true)
    fi

    FASTQC_FILE="$FASTQC_DIR/fastqc_data.txt"

    if [[ -f "$FASTQC_FILE" ]]; then
        read GC TOTAL <<< $(awk '
            BEGIN {gc_sum=0; gc_count=0; total_reads="NA"; in_gc=0; in_basic=0}
            /^>>Basic Statistics/ {in_basic=1}
            /^>>Per base sequence content/ {in_gc=1; next}
            /^>>END_MODULE/ {in_gc=0; in_basic=0}
            in_gc && !/^#/ {gc_sum+=$2; gc_count++}
            in_basic && /Total Sequences/ {total_reads=$3}
            END {if(gc_count>0) printf "%.2f ", gc_sum/gc_count; else printf "NA "; print total_reads}
        ' "$FASTQC_FILE")
    else
        GC="NA"
        TOTAL="NA"
    fi

    echo -e "$SAMPLE\t$GROWTH\t$STRANDED\t$tRNA\t$rRNA\t$CONTAM\t$PAIRED\t$GC\t$TOTAL" >> "$OUTPUT"

done < labels.txt #List of GSEs analyzed

echo "Complete. Results saved to $OUTPUT"

