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
GSM) and `studies.tsv`, with a status and exclusion reason for every run.

## 2. Prescreen

```bash
conda activate ecoli_ingest          # needs fastp, vg, samtools, bedtools, rseqc, centrifuge; sra-tools if available
python ecoli_ingest/atlas_build/prescreen/prescreen.py --manifest work/atlas_manifest \
    --out work/prescreen --resources resources --jobs 16 --threads 2
```

For each candidate it takes 200,000 reads (pairs) of the sample's first run without downloading
the run, and writes `work/prescreen/samples/<GSM>.json`:

| Step | Metrics |
|---|---|
| fastp (paired-end with `--detect_adapter_for_pe`) | % reads kept, mean read length after trimming, % adapter-trimmed, insert-size peak, duplication rate |
| vg giraffe + surject | % aligned, % perfect, % properly paired, mean MAPQ |
| RSeQC `infer_experiment` | forward and reverse fractions, strand call |
| `gene_type.sh` on the counting annotation | % of read overlaps on CDS, rRNA, tRNA, ncRNA, Rfam, pseudogene features |
| Centrifuge on 20,000 trimmed reads | % *E. coli* (the number the workflow's gate uses), largest other species |

- **Reads** come from `fastq-dump -N/-X` when sra-tools is installed (skipping the first 50,000
  spots), otherwise from the start of the ENA FASTQ files (the connection is closed once enough
  has arrived).
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

Measured on five samples (2 threads each, reads from ENA): 36-79 seconds per sample, of which
14-24 s download, 10-37 s giraffe, 7-14 s surject, 1.4 s Centrifuge.

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
python $Q decide --table $T --min-pct-ecoli 30 --min-pct-aligned 50                # change thresholds, no re-run
python $Q mark-model --samples model_samples.txt --model atlas_v2 --table $T
python $Q summary --table $T
```

`pre_*` and `full_*` are read by the same functions (`qc_metrics.py`) from the same kinds of
files, so they are directly comparable. `add-full` maps inGEST sample names to GSMs through the
`accession:` entries of the run's `config.yaml`.

Default thresholds (`qc_metrics.DEFAULT_THRESHOLDS`): a sample fails when its dominant strand
fraction is below 0.8 (`unstranded`), % *E. coli* below 30 (`contamination`), % reads kept below
75 (`trimming`), mean trimmed read length below 30 bp (`short_reads`), or % aligned below 50
(`low_alignment`). The first three are the workflow's own gates; the last two are provisional.

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
