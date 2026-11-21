# Prescreening atlas candidates and the sample QC table

Branch `ingest-update`. The counting on this branch accepts strand-specific libraries only
([counting.md](counting.md)), and strandedness is not in the metadata. The prescreen runs the
inGEST steps on a small read subsample of every candidate sample so that only usable samples are
processed in full.

## 1. Candidate manifest

```bash
python ecoli_ingest/atlas_build/preprocessing/build_manifest.py --out-dir work/atlas_manifest \
    --atlas-samples atlas_data/merged_count_table.txt --old-qc raw_data/quality_control.tsv
```

One ENA search (*E. coli* and all strains, RNA-Seq) gives `runs.tsv`, `samples.tsv` (one row per
GSM) and `studies.tsv`, with a status and exclusion reason for every run. ENA holds some runs
without an organism; those in a GEO study the search already found are added and marked
`taxon_missing_in_ena`. A study in which every run lacks the organism is not found.

## 2. Prescreen

```bash
conda activate ecoli_ingest          # needs fastp, vg, samtools, bedtools, rseqc, centrifuge; sra-tools if available
python ecoli_ingest/atlas_build/prescreen/prescreen.py --manifest work/atlas_manifest \
    --out work/prescreen --resources resources --jobs 16 --threads 2
```

For each candidate it takes `--spots` reads (pairs; default 200,000, the 2026-11 screen used
100,000) of the sample's first run without downloading the run, and writes
`work/prescreen/samples/<GSM>.json`:

| Step | Metrics |
|---|---|
| fastp (paired-end with `--detect_adapter_for_pe`) | % reads kept, mean read length after trimming, % adapter-trimmed, insert-size peak, duplication rate |
| vg giraffe + surject | % aligned, % perfect, % properly paired, mean MAPQ |
| RSeQC `infer_experiment` | forward and reverse fractions, strand call |
| `gene_type.sh` on the counting annotation | % of read overlaps on CDS, rRNA, tRNA, ncRNA, Rfam, pseudogene features |
| Centrifuge on 50,000 trimmed reads | species test (below); % unclassified, % Enterobacteriaceae, % synthetic constructs; % *E. coli* as the workflow's gate computes it |

- **Reads** come from `fastq-dump -N/-X` when sra-tools is installed (skipping the first 50,000
  spots), otherwise from the start of the ENA FASTQ files (the connection is closed once enough
  has arrived). `--fetch-interval` (default 2 s) is the minimum time between the starts of two
  downloads over all jobs, to stay well inside NCBI's and ENA's request limits.
- **Time limit**: each step is stopped after 25 minutes. `vg giraffe` can take very long on reads
  that are not *E. coli*; such a sample gets an `error` record.
- **Centrifuge index**: copied once to `/dev/shm/ecoli_ingest_centrifuge` (8.6 GB of memory) and
  memory-mapped from there; each sample then opens it in about a second. Do not memory-map it
  from a network file system: that took minutes per sample. `--index-cache DIR` chooses another
  local directory, `--index-cache none` reads the index normally from the resources directory,
  `--remove-index-cache` deletes the copy at the end.
- **Resuming**: samples with a JSON are skipped. `--redo-errors` repeats the failed ones,
  `--redo` everything. `--shard i/n` screens one of n parts (`slurm_prescreen.sh` runs one part
  per array task); `--gsm`, `--gse`, `--limit` and `--list` select samples.
- A failed sample gets a JSON with `error` (step and message) and its log stays in the scratch
  directory.

Cost depends on read length. For 100,000 pairs of 150 bp reads with 30 samples at a time on 48
cores: about 90 s giraffe, 30 s surject, 15 s fastp, 5 s download (sra-tools), 5 s Centrifuge per
sample, about 800 samples per hour. `vg` uses more than one core per job even with `--threads 1`,
so run fewer jobs than cores.

### Is the sample *E. coli*?

The workflow's Centrifuge gate (`pct_ecoli`: share of the species *E. coli* among reads with a
single match) is not used for the decision. It splits *E. coli* reads between *Escherichia* and
*Shigella* genomes and counts synthetic constructs against the sample, so real *E. coli*
libraries can score under 20 % (all of GSE67402 does).

The species test (`qc_metrics.species_metrics`) uses the same Centrifuge output: among reads with
a single match that resolves to a genus, synthetic constructs left out, `pct_escherichia_shigella`
is the share in *Escherichia* or *Shigella* and `top_other_genus` / `pct_top_other_genus` the
largest other genus. On test samples, two *Salmonella enterica* libraries scored 0.7 and 1.4 %
with *Salmonella* at 95-99 %; eight *E. coli* libraries (K-12, BL21, O157:H7, CFT073; including
GSE67402) scored 53-99 % with no other genus above 7 %. It cannot separate *Shigella* from
*E. coli*, and a mixed culture is caught only when the other organism dominates.

The prescreen records the test twice: from all classified reads (this needs only Centrifuge
output, so the workflow could compute it before alignment) and again without the reads on rRNA /
tRNA genes (`*_nostable`). The decision uses the second. Ten test samples separated either way,
but over the 10,672 samples of the 2026-11 screen the share is below 50 % in 1,473 samples with
those reads and in 860 without: in libraries that are mostly stable RNA, rRNA reads are assigned
to *Salmonella* and other enteric genera. A workflow version of the test therefore has to run
after alignment.

## 3. Sample QC table

`atlas_data/sample_qc.tsv` (not tracked in git), one row per GSM:

| Columns | Content |
|---|---|
| `GSM` … `in_atlas_v1` | identity and manifest status |
| `pre_*` | prescreen metrics, with the run, read number, date and pipeline commit |
| `pre_decision`, `pre_reasons` | `pass`, `fail`, `incomplete` or `error`, and why |
| `full_*` | the same metrics from a complete inGEST run, with the results directory and date |
| `in_model` | the atlas model the sample is part of |

```bash
T=atlas_data/sample_qc.tsv
Q=ecoli_ingest/atlas_build/prescreen/qc_table.py
python $Q merge-prescreen --manifest work/atlas_manifest --samples-dir work/prescreen/samples --table $T
python $Q add-full --results <run>/results --config <run>/config.yaml --table $T   # after a full inGEST run
python $Q add-full --batches work/atlas_full/results --table $T                    # every batch of the atlas recount
python $Q decide --table $T --min-pct-escherichia-shigella 50 --min-pct-aligned 50 # change thresholds, no re-run
python $Q mark-model --samples model_samples.txt --model atlas_v2 --table $T
python $Q mark-model --metadata atlas_data/metadata_v2.tsv --min-cds-reads 200000 --max-dna-fraction 0.10 --model atlas_v2 --table $T   # by rule
python $Q summary --table $T
```

`pre_*` and `full_*` are read by the same functions (`qc_metrics.py`) from the same kinds of
files, so they are directly comparable. `add-full` maps inGEST sample names to GSMs through the
`accession:` entries of the run's `config.yaml`.

Default thresholds (`qc_metrics.DEFAULT_THRESHOLDS`). A sample fails as:

| Reason | When |
|---|---|
| `unstranded` | dominant strand fraction below 0.8 |
| `phage` | the largest other genus is a bacteriophage (`qc_metrics.PHAGE_GENERA`) and at least 0.5 % of all reads given to Centrifuge. Phage infection and induction experiments are excluded. The PhiX spike-in and plant or animal viruses do not count. |
| `not_ecoli` | *Escherichia* + *Shigella* share below 50 % and the largest other genus at least 2 % of all reads. The second condition matters: libraries of short reads leave few classifiable reads, spread over many genera, and are judged by alignment instead. |
| `mixed_organism` | the largest other genus is at least 10 % of all reads although *E. coli* is the majority |
| `trimming` | under 75 % of reads kept by fastp |
| `short_reads` | mean read length after trimming below 30 bp |
| `low_alignment` | under 50 % of reads aligned |

The species rows are the workflow's species gate (`species_metrics.species_verdict`,
`SPECIES_THRESHOLDS`), so the prescreen and a full inGEST run decide in the same way. In a full
run the gate also stops samples before alignment when they are clearly something else (a phage,
or under 20 % *Escherichia* / *Shigella* with another genus at 5 % or more of all reads, on the
all-reads values): on this screen that early check stops 362 of the 547 samples the full rule
refuses and 1 of the 10,125 it accepts.

With fewer than 100 genus-resolved reads there is no species decision (`incomplete`). The strand
and trimming rules are the workflow's own gates; the others were set on the distribution of the
2026-11 screen and can be changed with `qc_table.py decide`. Full inGEST runs have no species
test yet, so the `full_*` species columns stay empty.

`study_candidates` and `large_study` mark studies with at least 500 candidate samples (GSE206047,
1,077 samples). One such study is about an eighth of all passing samples, so statistics over the
atlas should be reported with and without it; `qc_table.py summary` prints both.

## 4. Setting up a node

1. Clone the repository and check out `ingest-update`.
2. `mamba env create -f ecoli_ingest/environment.yaml` (includes sra-tools and pigz), then
   `conda activate ecoli_ingest`.
3. Copy `resources/vg`, `resources/centrifuge` and the four
   `resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.*` files. The v2 annotation is not
   in the `resources-v1` release; alternatively rebuild it with `build/build_annotation.py` from
   `work/ingest_update/ncbi/`.
4. Copy `work/atlas_manifest/` or rebuild it (step 1 above; the comparison options need
   `atlas_data/merged_count_table.txt` and `raw_data/quality_control.tsv`).
5. Needs: outbound HTTPS to NCBI and ENA; about 9 GB of `/dev/shm` for the Centrifuge index plus
   under 1 GB of memory per running sample; local scratch of about 200 MB per running sample.

Run order: a pilot on samples with known answers (`--gse GSE271069 GSE67402`, whose full-run
values are in the QC table after `add-full` on `work/ingest_update/*/results`), compare `pre_*`
with `full_*`, then the full list.
