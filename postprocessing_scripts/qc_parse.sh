#Script used to extract QC information from GSEs after they run

OUTPUT="sample_metrics.tsv"
INDEX="sample_dirs.tsv"

#Getting directories
> "$INDEX"

for d in GSE*/*_results/GSM*; do
    [[ -d "$d" ]] || continue
    sample=$(basename "$d")
    echo -e "$sample\t$d" >> "$INDEX"
done

#QC metrics we are looking for
echo -e "sample\tstrandedness_t_ratio\ttRNA_fraction\trRNA_fraction\tcontamination_percent\tpaired_single_end\tGC_bias\tfraction_aligned\tfraction_perfect_aligned" > "$OUTPUT"

while IFS=$'\t' read -r SAMPLE SAMPLE_DIR; do
    echo "Processing $SAMPLE..."

    GSM_PREFIX=${SAMPLE%%_*}

    #Strandedness
    shopt -s nullglob
    INFER_FILES=("$SAMPLE_DIR"/rseqc/"${GSM_PREFIX}"*_infer_experiment.txt)
    shopt -u nullglob

    if (( ${#INFER_FILES[@]} )); then
        STRANDED=$(awk '
            /Fraction of reads explained/{
                match($0, /: ([0-9.]+)/, a); print a[1]; exit
            }
        ' "${INFER_FILES[0]}")
    else
        STRANDED="NA"
    fi

    #tRNA / rRNA fractions
    shopt -s nullglob
    FEATURE_FILES=("$SAMPLE_DIR"/feature_overlap/"${GSM_PREFIX}"*_feature_overlap_mqc.tsv)
    shopt -u nullglob
    
    if (( ${#FEATURE_FILES[@]} )); then
        read tRNA rRNA <<< $(awk '
            BEGIN{t="NA"; r="NA"}
            $1=="tRNA" && $2 ~ /^[0-9.]+$/ {t=$2/100}
            $1=="rRNA" && $2 ~ /^[0-9.]+$/ {r=$2/100}
            END{
                printf "%s %s",
                    (t=="NA" ? "NA" : sprintf("%.4f", t)),
                    (r=="NA" ? "NA" : sprintf("%.4f", r))
            }
        ' "${FEATURE_FILES[0]}")
    else
        tRNA="NA"
        rRNA="NA"
    fi

    #Contamination percentage
    shopt -s nullglob
    CENT_FILES=("$SAMPLE_DIR"/centrifuge/"${GSM_PREFIX}"*_report.txt)
    shopt -u nullglob

    if (( ${#CENT_FILES[@]} )); then
        E_COLI=$(awk '$6=="Escherichia" && $7=="coli"{print $1; exit}' "${CENT_FILES[0]}")
        [[ -n "$E_COLI" ]] && CONTAM=$(awk -v x="$E_COLI" 'BEGIN{print 100-x}') || CONTAM="NA"
    else
        CONTAM="NA"
    fi

    #Sequencing type
    shopt -s nullglob
    R2_FILES=("$SAMPLE_DIR"/fastqc/*R2*fastqc.html)
    (( ${#R2_FILES[@]} )) && PAIRED="Paired" || PAIRED="Single"
    shopt -u nullglob

    #GC bias
    shopt -s nullglob
    FASTQC_ZIPS=("$SAMPLE_DIR"/fastqc/*_clean_R1_fastqc.zip)
    shopt -u nullglob

    if (( ${#FASTQC_ZIPS[@]} )); then
        GC=$(unzip -p "${FASTQC_ZIPS[0]}" */fastqc_data.txt | awk '
            BEGIN{gc=0;n=0;g=0}
            /^>>Per base sequence content/{g=1;next}
            /^>>END_MODULE/{g=0}
            g && !/^#/{gc+=$2;n++}
            END{if(n>0) printf "%.2f\n", gc/n; else print "NA"}
        ')
    else
        GC="NA"
    fi

    #Alignment stats
    shopt -s nullglob
    VG_FILES=("$SAMPLE_DIR"/vg/"${SAMPLE}"*_giraffe.stats.txt)
    shopt -u nullglob

    if (( ${#VG_FILES[@]} )); then
        read TOTAL_ALN ALIGNED PERFECT <<< $(awk '
            /Total alignments:/ {ta=$3}
            /Total aligned:/    {al=$3}
            /Total perfect:/    {pe=$3}
            END{print ta, al, pe}
        ' "${VG_FILES[0]}")

        if [[ -n "${TOTAL_ALN:-}" && "$TOTAL_ALN" -gt 0 ]]; then
            FRACTION_ALIGNED=$(awk -v a="$ALIGNED" -v t="$TOTAL_ALN" 'BEGIN{printf "%.4f", a/t}')
            FRACTION_PERFECT=$(awk -v p="$PERFECT" -v t="$TOTAL_ALN" 'BEGIN{printf "%.4f", p/t}')
        else
            FRACTION_ALIGNED="NA"
            FRACTION_PERFECT="NA"
        fi
    else
        FRACTION_ALIGNED="NA"
        FRACTION_PERFECT="NA"
    fi

    echo -e "$SAMPLE\t$STRANDED\t$tRNA\t$rRNA\t$CONTAM\t$PAIRED\t$GC\t$FRACTION_ALIGNED\t$FRACTION_PERFECT" >> "$OUTPUT"

done < "$INDEX"

