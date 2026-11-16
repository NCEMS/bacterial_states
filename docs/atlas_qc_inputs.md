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
  (`*_nostable`), exactly as the prescreen does (`species_metrics.py` is shared). It is also the
  workflow's species gate (`species_gate`, on by default with Centrifuge): `verdict` in the
  record's `species` block is `ok`, `no_species`, `phage`, `not_ecoli` or `mixed_organism`, and
  only `ok` and `no_species` samples are counted. Samples that are clearly not *E. coli* are
  stopped before alignment and have no record.
- **DNA values (item 15).** `dna_fraction_est` is no longer the antisense fit. It is made from the
  coverage track: for every 100 bp bin the lower of the two strands' depths (DNA covers both
  strands, RNA mostly one), then the median over bins (`minstrand_cov`), and
  `dna_fraction_est = 2 x minstrand_cov x genome length / aligned bases`. It is computed when the
  record is assembled, from the saved track, so the definition can change without alignments.
  The antisense fit is kept as `floor_a`, `leakage_b`, `dna_fraction_fit` (Theil-Sen line over
  genes, rRNA and tRNA left out). Also kept: `floor_cov_*` (10 % quantile of bin depth),
  `q25_cov_*`, `intergenic_cov_*` (median over bins that touch no feature), `antisense_per_base`,
  `aligned_bases`, `dna_fraction_limit` (the DNA share below which the estimate reads 0 or too low; see the calibration). Depths are mean
  depth per base. See "Calibration of the DNA estimate" below.
- **Off-reference share (item 7)** is (`Total aligned` of `vg stats` - mapped reads in the BAM) /
  `Total aligned`. Not yet checked on a non-K-12 sample.
- **Tool versions (item 14)** come from the tools themselves (`--version`), not from MultiQC.
- **Manifest (item 12).** Set `run_metadata: <runs.tsv>` in `config.yaml`; library and instrument
  fields of the sample's runs are copied from it. Without it, the record lists the runs and their
  read numbers for accession samples only.
- **Added:** `ladder` (reads in, kept, aligned, on the reference, assigned, assigned outside rRNA
  and tRNA), `antisense_assignment`, `gene_body_slope`, `reads.length_quantiles`,
  `composition.short_features` (reads on features under 200 nt, rRNA and tRNA genes left out).
- Unstranded samples are not counted and get no record.
- **Deleting alignments.** `<s>.gam` is always removed once `vg surject` and `vg stats` have
  read it. With `keep_alignments: false` in `config.yaml`, `<s>_sort.bam` and its index are
  removed too, as soon as counting, the coverage track and the QC steps have read them. A two-sample test left 7 MB of results and no alignment file.

### In the QC table

`qc_table.py add-full` reads each sample's record and fills, next to the `full_*` metrics:

- the species values of the full run;
- `qc_metrics.RECORD_COLUMNS` as `full_<name>`: read-length quantiles (`length_q05_r1`,
  `length_q50_r1`, `length_q95_r1`, `length_q50_r2`), `cycles_r1`, `gc_content`, fragment-size
  quartiles, `assigned`, `counting_unit`, `pct_assigned_not_stable`, the RNA family shares
  (`pct_16S`, `pct_23S`, `pct_5S`, `pct_tRNA_family`, `pct_ssrA`, `pct_rnpB`, `pct_ssrS`,
  `pct_ffs`), `pct_short_features`, `off_reference_fraction`, `substitution_rate`,
  `gene_body_slope`, the DNA-background values, and `n_runs`, `library_selection`,
  `instrument_model`, `pipeline_commit` as recorded by the run. Family shares are % of all
  read-feature assignments.

`merge-prescreen` copies the manifest's `n_runs`, `reads`, `bases`, `library_strategy`,
`library_source`, `library_selection`, `instrument_platform`, `instrument_model` and
`first_public` into the identity columns. A technical twin run under the name `<GSM>__<variant>`
gets its own row with `twin_of` set to the GSM.

## Calibration of the DNA estimate

Run of 2026-11-15, `work/dna_calibration/`. Three libraries (K-12 76 bp, K-12 150 bp, Nissle 1917
150 bp; 5.1-6.0 M read pairs) were run as they are and with simulated MG1655 genomic reads added to
make up 1, 5 and 20 % of the reads (`make_twins.py`, variants `g1`, `g5`, `g20`). Each estimator
was converted to a share of the aligned library and its increase over the unspiked sample divided
by the share actually added (1.00 = exact):

| Estimator | Mean | Range over the 9 spiked samples | Unspiked samples read |
|---|---|---|---|
| median over bins of the lower strand's depth (**used**) | 1.02 | 0.94-1.24 | 0.2, 1.7, 1.0 % |
| 25 % quantile of bin depth per strand | 1.05 | 0.92-1.36 | 0.1, 1.2, 0.1 % |
| median depth between genes | 1.12 | 1.00-1.28 | 0.9, 2.2, 1.4 % |
| 10 % quantile of bin depth per strand (the first definition's cross-check) | 0.70 | 0.31-0.86 | 0, 0, 0 % |
| antisense fit over genes (the first definition) | 1.27 | 1.10-1.53 | 1.0, 1.7, 1.0 % |
| `intergenic_share`, `antisense_share` | 0.45 | 0.43-0.49 | 1.1-6.2 % |

- The chosen estimate follows added DNA within 6 % at 5 and 20 % and within 24 % at 1 %.
- Low quantiles underestimate because DNA coverage itself varies from bin to bin; the fit
  overestimates, and its slope `leakage_b` rises with DNA (0.0004 to 0.012 in one library), so the
  two are not independent.
- `intergenic_share` and `antisense_share` each rise by a little under half of the added share,
  as expected for reads spread over both strands and over genes and gaps alike.
- What the estimate reads without added DNA (0.2-1.7 %) is real DNA plus transcription of both
  strands of the same region, which this test cannot separate. It is an upper bound.

Second run (`work/se_depth_test/`, `work/dna_calibration2/`), same spikes under other conditions:

| Library | Increase / added at 1, 5, 20 % | Note |
|---|---|---|
| K-12, single-end (mate 2 dropped), 5.9 M reads | 1.25, 1.02, 0.98 | as for paired-end |
| K-12 subsampled to 1 M pairs | 0.75, 1.12, 0.94 | limit 2 % |
| K-12 subsampled to 250,000 pairs | 0, 0, 0.70 | limit 8 %; reads 0 below it |
| Nissle 1917 + simulated Nissle 1917 DNA | 0.88, 0.82, 0.79 | DNA of sequence MG1655 lacks is not seen |

- **Depth.** The median is 0 until DNA reaches roughly half a base of depth per strand.
  `dna_fraction_limit` (= genome length / aligned bases) is the DNA share at which that happens:
  about 0.4-0.5 % at 5 M pairs, 2 % at 1 M, 8 % at 250,000. Below the limit the estimate is 0 or
  too low, so an estimate of 0 means "below about the limit". From about twice the limit it was
  within 25 % of the added share.
- **Other strains.** With the strain's own DNA the estimate recovers 79-88 %: reads from sequence
  that MG1655 lacks do not align to the reference path (`off_reference_fraction` rose from 2.5 to
  4.8 %) and are not in the coverage track. About 13 % of the 100 bp bins have no coverage on
  either strand in this Nissle library at any spike level, which is the part of MG1655 that Nissle
  lacks.
- An estimator from the share of empty bins was tried for shallow libraries and rejected: it
  recovered only 0.3-0.45 of the added share.
- **Gene-body slope depends on layout.** The same library gives -0.08 as paired-end and +0.54 with
  mate 2 dropped: read 1 of a reverse-stranded library comes from the 3' end of the fragment. The
  slope must be used together with the layout.

## Technical twins

`ecoli_ingest/atlas_build/twins/make_twins.py` makes every variant below from raw reads. Only the
genomic-DNA variants have been run (the calibration above); the selection of about 200 samples and
the other variants have not.


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
