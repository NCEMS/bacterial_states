# Example: growth to long-term starvation (GEO GSE67402)

From GEO accessions to atlas neighbours, with every command and what came out of it.

**Data.** *E. coli* B REL606 in glucose-limited minimal medium, sampled at 3, 4, 5, 6, 8, 24, 48,
168 and 336 hours: three biological replicates of rRNA-depleted RNA, and replicate 3 again
without rRNA depletion (36 paired-end libraries, 77 SRA runs, 38 GB of FASTQ). Houser et al. 2015,
PLoS Comput Biol 11:e1004400. None of the samples, and no REL606 sample at all, is in the atlas
training data, so this is a held-out test.

**Files here.** `config.yaml` (the inGEST run configuration) and `samples.csv` (time, replicate and
library type per sample) are tracked. Everything the commands below write (`sra/`, `results/`,
`checkpoints/`, `benchmarks/`, `atlas/`, logs) stays in this folder and is ignored by git.

The numbers quoted are from the run of 2026-11-13 (96 cores, 72 GB, one 6-hour job).

## 1. inGEST: reads to counts

```bash
cd examples/GSE67402_growth_starvation_time_course
conda activate ecoli_ingest
ecoli_ingest check                 # tools, resources, 36 accessions -> 77 runs, 154 files, 38.24 GB
ecoli_ingest run -n                # what would run
ecoli_ingest run --cores 72 --max-downloads 12
```

Samples are listed by GSM accession in `config.yaml`; the workflow resolves them to runs, fetches
the reads into `sra/`, merges the runs of a sample, and goes through fastp, (Centrifuge,) `vg
giraffe`, featureCounts, DESeq2 and MultiQC. Results are in `results/`:

| Look at | What it tells you |
|---|---|
| `results/multiqc_report.html` | all QC in one report |
| `checkpoints/qc_failures.json` | samples excluded by a QC gate (empty here) |
| `results/all_samples_raw_counts.txt` | gene x sample counts (36 sample columns) |
| `results/deseq2/pca_plot.png`, `t003h_vs_t*_deseq_results.tsv` | PCA and one table per contrast |

What happened in this run, including the parts that did not go smoothly:

- **Downloads.** The reads were downloaded from ENA at about 1 MB/s per connection. One file
  (`SRR1945242_2.fastq.gz`) is an empty directory on the ENA server and was taken from NCBI with
  `fasterq-dump` instead; its mate 1 from NCBI is identical to the ENA file. This is why the
  workflow now fetches with sra-tools first and uses ENA as the backup (`sra_method`, see the
  [inGEST README](../../ecoli_ingest/README.md)); the same run takes 20 seconds that way.
- **Adapters.** These libraries contain many very short inserts. The first pass, before fastp was
  given `--detect_adapter_for_pe`, left adapter read-through in about half of the read pairs;
  Centrifuge classified those as "synthetic construct" (83-99 % of the classified reads) and every
  sample failed the 30 % *E. coli* gate. With the option, fastp trims both adapters; the median
  sample keeps 78 % of its reads (mean length about 50 of 101 bp). The option is off by default
  (it was not used for the atlas training counts); this config turns it on with
  `fastp_detect_adapter_for_pe: true`.
- **Two weak samples.** `t005h_rep1` and `t168h_rep1` keep only 25 % and 32 % of their reads (mean
  length 26 and 31 bp) and align worst (17 % and 40 %; the median sample aligns 69 %). They are
  included because `fastp_min_kept_percentage` is set to 20 in this config; with the default 75 they
  and 13 other samples would be excluded. Read their results with that in mind.
- **Centrifuge is off in this config** (`centrifuge: false`) to fit one 6-hour job: the screen takes
  up to 75 minutes for the largest samples. Set it to `true` to run it.
- **DESeq2**, every later time point against 3 h (padj < 0.05 and at least two-fold): 9, 16 and 14
  genes at 4, 5 and 6 h, then 1529 at 8 h, 2524 at 24 h, 2308 at 48 h, 2274 at 168 h and 1962 at
  336 h. The culture changes little during growth and switches between 6 and 8 hours.

## 2. Ecoli_ATLAS: counts to a position in the atlas

```bash
conda activate ecoli_atlas
FT=../../atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt
M=../../outputs/models ; A=../../outputs/atlas

ecoli_atlas prepare --ingest-results results --metadata samples.csv --models $M --feature-table $FT --out atlas/prepared
ecoli_atlas embed   --counts atlas/prepared/counts.tsv --models $M --feature-table $FT --out atlas/embedded
ecoli_atlas compare --embeddings atlas/embedded/final_embedding.csv --atlas $A --out atlas/compared -k 15 --umap \
    --metadata atlas/prepared/samples.tsv --group-by time_h --order-by time_h
```

- `prepare` turns the inGEST counts into the atlas input (`counts.tsv`, 4493 genes x 36 samples) and
  attaches `samples.csv` (`samples.tsv`).
- `embed` gives each sample its 46-dimensional embedding and the three encoder calls. Gene coverage
  is 4367 of 4385 model genes (99.6 %), although REL606 is a B strain (`gene_coverage.tsv`).
- `compare` finds the 15 nearest atlas samples of every sample and, with `--group-by time_h`,
  summarises per time point.

| Look at | What it tells you |
|---|---|
| `atlas/embedded/sample_predictions.csv` | growth phase, medium type and lab-strain call with probabilities |
| `atlas/compared/summary.tsv` | per sample: nearest atlas sample and study, neighbour votes, novelty |
| `atlas/compared/group_summary.tsv` | the same per time point, plus replicate agreement and step size |
| `atlas/compared/fig_factor_by_group.png` | encoder probabilities along the time course |
| `atlas/compared/fig_umap_groups.png` | the samples on the atlas UMAP, coloured by time, centroids joined in order |
| `atlas/compared/fig_query_distances.png` | distances between the 36 samples |
| `atlas/compared/fig_neighbor_labels.png` | what the neighbours of each time point are |

What the model says about a strain and a design it has never seen:

- **Growth phase.** All 12 samples from 3 to 5 h and 3 of 4 at both 6 h and 8 h are called exponential. At 24 h
  three of four are called stationary (mean probability 0.81). The call does not stay there: one of
  four at 48 h and at 168 h and two of four at 336 h are called stationary. The two-class encoder
  finds the entry into stationary phase but does not recognise long-term starvation as stationary.
- **Medium.** 31 of 36 samples are called defined, which is correct; the five "complex" calls are
  at 4 h, 24 h and 48 h.
- **Lab strain.** 30 of 36 are called a lab strain. REL606 is one, but a B strain, which the
  training data does not contain.
- **Relationships between the samples** (`fig_query_distances`). The 3-6 h samples form one tight
  group (cosine distance about 0.002 between and within time points). The 8 h samples move away
  (centroid step 0.19) and disagree with each other (within-group distance 0.09), 24 h and 48 h sit
  close to the growth group again, and 168 h and 336 h move away once more (steps 0.27 and 0.09).
  Replicates of one time point are closer than samples of different time points (median 0.005
  against 0.057 for the rRNA-depleted libraries).
- **Novelty.** The mean novelty percentile per time point is 0.30-0.64, highest at 8 h and 168 h: no
  time point is unlike everything in the atlas.
- **Neighbour studies.** Growth-phase samples sit next to GSE107093 and GSE21341 (K-12 in M9),
  24-48 h next to GSE224463 (the study withheld from training, LB), 8 h and the starved samples next
  to GSE111874 (MG1655 in MOPS through exponential, early stationary and stationary phase) and
  GSE235465.

Two variations on the same command:

```bash
# growth-phase block only (6 of the 46 dimensions)
ecoli_atlas compare --embeddings atlas/embedded/final_embedding.csv --atlas $A --out atlas/compared_growth_phase -k 15 --umap \
    --blocks growth_phase --metadata atlas/prepared/samples.tsv --group-by time_h --order-by time_h
# rRNA-depleted against total RNA
ecoli_atlas compare --embeddings atlas/embedded/final_embedding.csv --atlas $A --out atlas/compared_library -k 15 --umap \
    --metadata atlas/prepared/samples.tsv --group-by library
```

Library type hardly matters: the rRNA-depleted and the total-RNA library of the same RNA
(replicate 3) are 0.000-0.017 apart at the nine time points, less than replicates are from each
other, and both groups have the same neighbour studies.

## 3. Where the neighbouring studies sit

`compare` is for new samples. `neighbors` asks the same question for studies that are already in
the atlas; here the four that the time course lands next to:

```bash
ecoli_atlas neighbors GSE107093 GSE224463 GSE111874 GSE21341 --atlas $A --models $M --out atlas/neighbors_top_studies --umap
```

`atlas/neighbors_top_studies/study_summary.tsv` has one row per study (which split it was in, its
own nearest studies, how often the neighbour votes agree with its metadata), `summary.tsv` one row
per sample, `umap.png` the studies on the atlas UMAP. GSE107093 and GSE224463 are each other's
nearest studies, and GSE111874's neighbours agree with its curated growth phase for only 42 % of its
samples, so the region where the starved samples land is one where growth phase is mixed.

## Not covered here

Reproducing the paper figures and case studies (`ecoli_atlas figures`, `casestudy`) and retraining
(`train`, `verify-models`) work on the atlas training data, not on new samples: see the
[Ecoli_ATLAS README](../../ecoli_atlas/README.md).
