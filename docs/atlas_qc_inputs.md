# QC to save during the atlas recount

Status: implemented in the inGEST workflow on `ingest-update` (2026-11-15), except the technical
twins at the end. It lists what a full inGEST run keeps per sample so that the atlas model
(`docs/atlas_model_v2.md`, on the `update-atlas` branch) can use library and sequencing properties
as inputs. "Today" in the table below describes the workflow before this change; the section
"What the workflow writes" describes it now. Alignments (`.gam`, `_sort.bam`) are deleted after the recount, so anything
that needs them has to be computed during the run.

Principle: save distributions, not only summary values. Quantiles and ratios can then be redefined
without running the samples again.

File fields were checked against `work/ingest_update/GSE271069/results` (paired-end, fastp 1.4.0)
and `work/experiment3/experiment3_results` (single-end, fastp 0.26.0).

## Items

"Needs alignments" marks what cannot be recovered once BAM, GAM and temporary files are gone.

| # | Item | Source | Needs alignments | Today |
|---|---|---|---|---|
| 1 | Trimmed read-length histogram per mate | FastQC `fastqc_data.txt`, module `Sequence Length Distribution` (in `fastqc/<s>_clean_R{1,2}_fastqc.zip`) | no | in the zip, not parsed; the QC table has an integer mean only |
| 2 | Fragment-size histogram (paired-end) | `TLEN` of properly paired reads in `_sort.bam` | yes | only `insert_size.peak` from fastp |
| 3 | Reads per RNA family and per feature type | count table and `features.tsv` | no | `pct_*` are shares of read–feature overlaps by type, not reads, with no family split |
| 4 | Antisense count per feature | second featureCounts pass with the opposite `-s` | yes | not produced |
| 5 | featureCounts assignment summary | `all_samples_raw_counts.txt.summary` | no | kept, not in the QC table |
| 6 | Stranded coverage track | `_sort.bam` | yes | not produced |
| 7 | Reads aligned to the graph but not placed on MG1655 | `.gam` and `_sort.bam` | yes | not computed |
| 8 | Full `vg stats` block | `vg/<s>_giraffe.stats.txt` | no | 4 of its values are in the QC table |
| 9 | Gene-body coverage profile | `rseqc/<s>_genebody_cov.geneBodyCoverage.txt` (100 percentiles) | no | kept, not parsed |
| 10 | Species metrics for full runs | per-read Centrifuge tables (`initial.tsv`, `<s>_output.tsv`) | yes (both are `temp()`) | prescreen only; the `full_*` columns stay empty |
| 11 | Read quality and trimming detail | fastp JSON | no | partly |
| 12 | Library and run metadata | `work/atlas_manifest/runs.tsv` | no | `qc_table.py` copies layout and instrument model only |
| 13 | Strand fractions and call | `strandedness.tsv` | no | saved |
| 14 | Tool versions | `multiqc_report_data/multiqc_software_versions.txt`, `summary.fastp_version` | no | not in the QC table |
| 15 | DNA contamination estimate | derived from items 4, 5 and 6 | yes, through those items | not computed |

### Details

**1. Read length.** Store the histogram as `[length, count]` pairs for each mate. FastQC groups
lengths into ranges for long reads; keep the range bounds as given. Derived values: quantiles at
1, 5, 25, 50, 75, 95, 99%. From fastp also keep the cycles before trimming
(`read{1,2}_before_filtering.total_cycles`, `summary.sequencing`) and the `filtering_result`
counts (`too_short_reads`, `low_quality_reads`, `too_many_N_reads`, `adapter_dimer_reads`).
The fastp JSON has no length histogram.

**2. Fragment size.** Histogram of absolute `TLEN` in 1 bp bins up to 1,000, one overflow bin.
Also keep fastp's `insert_size.histogram` (512 bins) and `insert_size.unknown`; fastp only
measures pairs whose mates overlap, so it is a fallback. Single-end samples have neither.

**3. RNA families.** Read counts, summed from the count table, for: 16S (`rrs*`), 23S (`rrl*`),
5S (`rrf*`), tRNA, ssrA, rnpB, ssrS, ffs, and for each `Type` (CDS, pseudo, ncRNA, Rfam). Because
counting uses `-O`, a read on two features is counted for both; also store the featureCounts
`Assigned` number so shares of assigned reads can be formed.

**4. Antisense counts.** Same annotation and options as the main count, with the strand setting
reversed. One table per run (`all_samples_antisense_counts.tsv`). Uses: strand leakage per gene,
DNA contamination (reads on both strands and between genes), evidence for missing genes.

**5. featureCounts summary.** All rows per sample: `Assigned`, `Unassigned_NoFeatures`,
`Unassigned_Unmapped` and the rest. `Unassigned_Ambiguity` and `Unassigned_MultiMapping` are 0 by
construction (`-O`; no NH tag).

**6. Coverage track.** Summed read depth in 100 bp bins along NC_000913.3 (46,417 bins), one array
per strand, strand meaning the transcript strand after applying the library's strand call. Two
unsigned 32-bit arrays are 371 kB before compression. One compressed file per sample
(`coverage/<s>.cov100.npz`). This is the catch-all for questions that would otherwise need the
BAM: deleted regions, coverage between genes, antisense signal, uneven coverage along operons.

**7. Reads off the MG1655 path.** Number of reads aligned in the GAM (`Total aligned` of
`vg stats`) minus the number of mapped reads in the surjected BAM, as a share of aligned reads.
It measures how much of the library comes from sequence MG1655 does not have. How `vg surject`
reports reads that are not on the path must be confirmed on one non-K-12 sample before this
definition is fixed.

**8. `vg stats`.** Keep every line: aligned, perfect, gapless, paired, properly paired; alignment
score and mapping quality (mean, median, sd); insertions, deletions, substitutions and softclips
(bases and events). Substitutions per aligned base is a measure of divergence from MG1655. The
file currently ends with placeholder lines (`Total time: 123 seconds`, `Speed: 123 reads/second`)
after the real ones; ignore the last two lines.

**9. Gene-body coverage.** The 100 values as written by RSeQC. Derived value: slope from 5′ to 3′.

**10. Species.** `species_reads`, `pct_escherichia_shigella`, `top_other_genus`,
`pct_top_other_genus` and the `_nostable` variants, as the prescreen computes them
(`qc_metrics.species_metrics`), plus `pct_unclassified` and `pct_synthetic` from an all-reads
report. Compute them in the workflow before the per-read tables are removed.

**11. fastp.** `before_filtering` and `after_filtering`: `total_reads`, `total_bases`, `q20_rate`,
`q30_rate`, `gc_content`, mean lengths. `duplication.rate`. `adapter_cutting`:
`adapter_trimmed_reads`, `adapter_trimmed_bases`, `read{1,2}_adapter_sequence`. From FastQC: the
ten most frequent entries of `Overrepresented sequences` and `Total Deduplicated Percentage`.

**12. Manifest.** Per run: `library_strategy`, `library_source`, `library_selection`,
`library_layout`, `instrument_platform`, `instrument_model`, `read_count`, `base_count`,
`first_public`. Per sample: the list of runs with these values (a sample can combine runs of
different length or layout) and the run count.

**15. DNA contamination.** Genomic DNA gives reads evenly along the genome on both strands. Values
to store (definitions and use in `docs/atlas_model_v2.md`, section 5a):

| Value | From |
|---|---|
| `intergenic_share` = `Unassigned_NoFeatures` / (`Assigned` + `Unassigned_NoFeatures`) | item 5 |
| `antisense_share` = antisense assigned / (sense + antisense assigned) | items 4, 5 |
| `floor_a`, `leakage_b`: intercept and slope of a robust fit over genes of antisense reads per base on sense reads per base | item 4 and the main count table |
| `dna_fraction_est` = 2 · `floor_a` · genome length / aligned reads | above and item 8 |
| `floor_cov_plus`, `floor_cov_minus`: 10% quantile of bin depth per strand | item 6 |
| `intergenic_cov_plus`, `intergenic_cov_minus`: median depth of bins that overlap no feature on either strand | item 6 and the counting BED |

rRNA and tRNA genes are left out of the fit. The antisense counts and the coverage track are the
primary data; these values can be recomputed from them, so the definitions may still change.

## Files

Per sample, under the sample's results directory:

| File | Content | Size |
|---|---|---|
| `qc/<s>.qc.json` | items 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15 | tens of kB |
| `coverage/<s>.cov100.npz` | item 6 | under 1 MB |

Per run: `all_samples_antisense_counts.tsv` (item 4).

Layout of `qc/<s>.qc.json`:

```json
{
  "sample": "GSM0000000",
  "pipeline": {"commit": "f9497a6", "versions": {"fastp": "1.4.0", "vg": "..."}},
  "runs": [{"run": "SRR0000000", "layout": "PAIRED", "instrument_model": "...",
            "library_selection": "...", "read_count": 0, "base_count": 0}],
  "reads": {
    "cycles": [43, 43],
    "before": {"total_reads": 0, "total_bases": 0, "q20_rate": 0, "q30_rate": 0, "gc_content": 0},
    "after":  {"total_reads": 0, "total_bases": 0, "q20_rate": 0, "q30_rate": 0, "gc_content": 0},
    "filtering": {"too_short_reads": 0, "low_quality_reads": 0, "too_many_N_reads": 0},
    "duplication_rate": 0,
    "adapter": {"trimmed_reads": 0, "trimmed_bases": 0, "read1_sequence": "", "read2_sequence": ""},
    "length_hist": {"r1": [[15, 44], [16, 50]], "r2": []},
    "overrepresented": {"r1": [["SEQUENCE", 0]], "r2": []}
  },
  "fragment": {"tlen_hist": [], "fastp_insert_hist": [], "fastp_insert_unknown": 0},
  "alignment": {"vg_stats": {"total": 0, "aligned": 0, "perfect": 0, "properly_paired": 0,
                             "score_mean": 0, "mapq_mean": 0, "substitution_bp": 0,
                             "insertion_bp": 0, "deletion_bp": 0, "softclip_bp": 0},
                "bam_mapped": 0, "off_reference_fraction": 0},
  "strand": {"layout": "PE", "fraction_forward": 0, "fraction_reverse": 0,
             "fraction_undetermined": 0, "call": "reverse"},
  "assignment": {"Assigned": 0, "Unassigned_NoFeatures": 0, "Unassigned_Unmapped": 0},
  "composition": {"by_type": {"CDS": 0, "pseudo": 0, "ncRNA": 0, "Rfam": 0, "rRNA": 0, "tRNA": 0},
                  "by_family": {"16S": 0, "23S": 0, "5S": 0, "tRNA": 0,
                                "ssrA": 0, "rnpB": 0, "ssrS": 0, "ffs": 0},
                  "antisense_assigned": 0},
  "dna": {"intergenic_share": 0, "antisense_share": 0, "floor_a": 0, "leakage_b": 0,
          "dna_fraction_est": 0, "floor_cov_plus": 0, "floor_cov_minus": 0,
          "intergenic_cov_plus": 0, "intergenic_cov_minus": 0},
  "gene_body": [0, 0, 0],
  "species": {"species_reads": 0, "pct_escherichia_shigella": 0, "top_other_genus": "",
              "pct_top_other_genus": 0, "pct_unclassified": 0, "pct_synthetic": 0}
}
```

Files that must not be deleted because the JSON is built from them and they are small: fastp
JSON, `fastqc_data.txt` (or the FastQC zip), `infer_experiment.txt`, `strandedness.tsv`, the
feature-overlap table, the gene-body coverage text, `giraffe.stats.txt`.

## What the workflow writes

Rules in `ecoli_ingest/src/ecoli_ingest/workflow/Snakefile`, scripts in `workflow/scripts/`:

| Rule | Script | Output |
|---|---|---|
| `featurecounts_combined` | second `featureCounts` call with every sample's strand reversed | `all_samples_antisense_counts.txt` and `.summary` (item 4) |
| `bam_qc` | `sample_qc.py bam` | `<s>/coverage/<s>.cov100.npz` (item 6), `<s>/qc/<s>_bam_qc.json` (fragment sizes, mapped reads, coverage background) |
| `species_qc` | `species_metrics.py` | `<s>/qc/<s>_species.json`, `<s>/centrifuge/<s>_report_all.txt` (item 10); only with Centrifuge on |
| `tool_versions` | `sample_qc.py versions` | `tool_versions.json` (item 14) |
| `sample_qc` | `sample_qc.py collect` | `<s>/qc/<s>.qc.json`, everything about the sample |

Differences from the specification above:

- **Coverage file.** Arrays are named `<contig>__plus` and `<contig>__minus` (transcript strand
  after applying the strand call), `uint32`, summed per-base depth per 100 bp bin, with `bin_size`
  and `strand_call`. Read with `numpy.load`.
- **Species (item 10)** is computed from the classification of every trimmed read (Centrifuge's
  first pass, `initial.tsv`), for all reads and again without reads on rRNA / tRNA genes
  (`*_nostable`), exactly as the prescreen does (`species_metrics.py` is shared). It is reported,
  not used as a gate; the workflow's `centrifuge_min_percentage` gate still uses the older
  species-level value, which fails real *E. coli* libraries (see `docs/prescreen.md`) and should be
  left unset for prescreened samples.
- **DNA values (item 15).** The fit is a Theil-Sen line (median slope over 200,000 random gene
  pairs, then median intercept), genes of type rRNA and tRNA left out. `dna_fraction_est` uses
  `Assigned + Unassigned_NoFeatures` of featureCounts as the number of aligned units, so that it
  is in the same unit as the counts (fragments for paired-end runs). Coverage background values
  are mean depth per base: `floor_cov_*` is the 10 % quantile over bins, `intergenic_cov_*` the
  median over bins that touch no feature on either strand. `antisense_per_base` (antisense reads
  per base over all genes outside rRNA / tRNA) is added because the median-based fit and the
  coverage quantiles are 0 for shallow samples (a 200,000-pair test sample gave 0 for all of them).
- **Off-reference share (item 7)** is (`Total aligned` of `vg stats` - mapped reads in the BAM) /
  `Total aligned`. Not yet checked on a non-K-12 sample.
- **Tool versions (item 14)** come from the tools themselves (`--version`), not from MultiQC.
- **Manifest (item 12).** Set `run_metadata: <runs.tsv>` in `config.yaml`; library and instrument
  fields of the sample's runs are copied from it. Without it, the record lists the runs and their
  read numbers for accession samples only.
- **Added:** `ladder` (reads in, kept, aligned, on the reference, assigned, assigned outside rRNA
  and tRNA), `antisense_assignment`, `gene_body_slope`, `reads.length_quantiles`.
- Unstranded samples are not counted and get no record.

`qc_table.py add-full` takes the species values of a full run from the record. The other new
values are not yet columns of `sample_qc.tsv`.

## Technical twins

Purpose: samples whose biology is identical by construction and whose read length, layout or
depth differ. They test whether those properties move a sample in the embedding, and can be used
in training as pairs that must coincide.

- **Selection:** about 200 paired-end samples with at least 75 cycles and at least 5 million read
  pairs, spread over studies and instruments, passing the prescreen.
- **Variants,** each made from the raw reads before fastp and then run through the unchanged
  workflow:

| Variant | Change |
|---|---|
| `t36`, `t50`, `t75` | both mates truncated to 36, 50, 75 bases (only lengths below the original) |
| `se` | mate 2 dropped |
| `se_t50` | mate 2 dropped and mate 1 truncated to 50 bases |
| `d1M`, `d250k` | random subsample of 1,000,000 and 250,000 read pairs |
| `g1`, `g5`, `g20` | simulated MG1655 genomic reads added (even coverage, both strands, the sample's read length and layout) to make up 1, 5 and 20% of the reads; for 50 of the selected samples |

- **Naming:** `<GSM>__<variant>`; the QC table gets a `twin_of` column with the original GSM.
- **Use:** twins are excluded from the atlas that `compare` and `neighbors` search.
- **Cost:** up to 7 extra runs per selected sample, about 1,400 runs, plus 150 with added genomic
  reads; the subsampled ones are short.
- **Limit:** truncation imitates fewer cycles, not a different fragment-size distribution or a
  different depletion kit. Those differences are only covered by the real variation between studies.
