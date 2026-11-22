# Example: growth to long-term starvation (GEO GSE67402)

From GEO accessions to counts and to a growth-phase call for every sample, with every command and
what came out of it.

**Data.** *E. coli* B REL606 in glucose-limited minimal medium (DM25), sampled at 3, 4, 5, 6, 8,
24, 48, 168 and 336 hours, three biological replicates of rRNA-depleted RNA (Houser et al. 2015,
PLoS Comput Biol 11:e1004400). The study also sequenced replicate 3 a second time without rRNA
depletion; those nine libraries are not part of this example (see *Choices* below). GSE67402 is
held out of the atlas model's training (a forced TEST study), so everything the model says about
these samples is out of sample.

**Files here.** `config.yaml` (the inGEST run configuration) and `samples.csv` (time and replicate
per sample) are tracked. Everything the commands below write (`sra/`, `results/`, `checkpoints/`,
`benchmarks/`, `atlas/`, logs) stays in this folder and is ignored by git.

The numbers below are from the run of 2026-11-22 (inGEST counting of branch `ingest-update`; the
atlas model of branch `update-atlas`, default settings).

## 1. inGEST: reads to counts

```bash
cd examples/GSE67402_growth_starvation_time_course
conda activate ecoli_ingest
ecoli_ingest check                 # tools, resources, accessions -> runs and files to download
ecoli_ingest run --cores 44
```

Samples are listed by GSM accession in `config.yaml`; the workflow resolves them to SRA runs,
fetches the reads into `sra/` (sra-tools, with the ENA files as the backup), merges the runs of a
sample, and goes through fastp, `vg giraffe`, the strandedness call, featureCounts, the QC record,
DESeq2 and MultiQC. The 36 libraries of the study took 1 h 55 min on 44 cores, downloads included.

| Look at | What it tells you |
|---|---|
| `results/multiqc_report.html` | all QC in one report |
| `checkpoints/qc_failures.json` | samples stopped by a QC gate (`t005h_rep1`, see below) |
| `results/strandedness.tsv` | every library is forward-stranded (0.97-0.99 of reads on the sense strand) |
| `results/all_samples_raw_counts.txt` | feature x sample counts (26 samples) |
| `results/<sample>/qc/<sample>.qc.json` | the QC record of a sample (read lengths, fragment sizes, GC bias, DNA estimate, ...) |
| `results/deseq2/pca_plot.png`, `t003h_vs_t*_deseq_results.tsv` | PCA and one table per contrast |

**DESeq2**, every later time point against 3 h (padj < 0.05 and at least two-fold): 0, 5 and 33
genes at 4, 5 and 6 h, then 1,129 at 8 h, 1,972 at 24 h, 1,906 at 48 h, 2,365 at 168 h and 2,035
at 336 h. The culture changes little during growth and switches between 6 and 8 hours.

## Choices made for this study

- **Read filter.** These libraries hold many very short inserts: after adapter trimming the median
  read is about 35 of 101 bp and the median sample keeps 75 % of its reads (the default gate,
  `fastp_min_kept_percentage: 75`, would stop half of them). The gate is set to **30**, which stops
  one sample, `t005h_rep1` (25 % kept). That library is unlike anything in the atlas: median insert
  27 bp and trimmed read 18 bp (shorter than in any atlas sample), 17 % of its reads align, and its
  GC bias has the opposite sign to its replicates'. Its expression matches the other exponential
  samples, but the atlas model calls it stationary, so it is a technical failure, not biology.
- **No total-RNA libraries.** The nine replicate-3 libraries made without rRNA depletion are left
  out of `config.yaml`: about 95 % of their reads are rRNA, which leaves them with about 100,000
  reads on genes, far below the 200,000 the atlas requires of its own samples, and the model calls
  even their exponential samples stationary.
- **Centrifuge is off** (`centrifuge: false`) to keep the run short; the species gate then does not
  apply. Set it to `true` to screen the reads.

## 2. Ecoli_ATLAS: counts to a growth-phase call

```bash
conda activate ecoli_atlas
M=../../outputs/models                       # the atlas model bundle
ecoli_atlas embed --ingest-results . --models $M --fill-missing-gc study --out atlas
```

`embed` reads the counts, antisense counts and QC records of the run directory and writes, per
sample, the embedding (`atlas/embedding.tsv`: phase, medium and lab-strain scores and the 24-d
residual), the three calls (`atlas/predictions.tsv`) and the sample table (`atlas/samples.tsv`).
It printed two warnings, both worth reading:

- **`--fill-missing-gc study`.** Four libraries (`t024h_rep1`, `t024h_rep3`, `t048h_rep1`,
  `t048h_rep3`, 21,000-67,000 reads on CDS) are too shallow for the GC-bias fit of the QC record,
  and the model needs it. With the option they get the median GC slope and curvature of the other
  samples of the study (`gc_filled` in `samples.tsv`); without it `embed` refuses them.
- **Low depth.** Seven samples have fewer than 200,000 reads on CDS features (`low_depth` in
  `samples.tsv`): `t008h_rep3`, the four above, `t048h_rep2` and `t168h_rep1`. They are outside the
  depths the model was trained on; their calls agree with their replicates here, but read them with
  care.

**Growth phase.** Probability of stationary phase (`p_stationary` in `predictions.tsv`); `*` low
depth, `+` GC bias filled:

| Hours | rep1 | rep2 | rep3 |
|---|---|---|---|
| 3 | 0.02 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 |
| 5 | (stopped by QC) | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 |
| 8 | 1.00 | 1.00 | 1.00 * |
| 24 | 1.00 *+ | 1.00 | 1.00 *+ |
| 48 | 1.00 *+ | 1.00 * | 1.00 *+ |
| 168 | 1.00 * | 1.00 | 1.00 |
| 336 | 1.00 | 1.00 | 1.00 |

Every sample from 3 to 6 h is called exponential and every sample from 8 h on stationary. The
model places the switch between 6 and 8 hours, as DESeq2 does, although the GEO text labels the
8 h samples mid-exponential. The phase score (`z1` in `embedding.tsv`) tells the same: 0.41-0.53
for the growing samples (`t003h_rep1`, -0.22, is the least clear) and -2.36 to -2.39 from 8 h on.

**Medium and strain.** All 26 samples are called lab strain (REL606 is a lab strain). The medium
call is minimal through 24 h; from 48 h on it moves toward rich and is firmly rich by one week
(`p_minimal` 0.005-0.06 at 168 and 336 h). This follows the time spent starving, not the entry into
stationary phase (the 8 h samples are still clearly minimal), and it is not caused by depth or by
the filled GC bias: the deep one-week libraries stay rich when thinned to 50,000 reads or given the
study-median GC. The atlas has no minimal-medium samples starved for days; in long starvation the
biosynthetic genes that mark minimal medium shut down, and the cells may also live on what dead
cells release. The two 48 h replicates still called minimal are the two shallowest libraries of the
example (46,000 and 21,000 reads on CDS).

## 3. Both dimensions along the time course

`ecoli_atlas axes` places samples on ordered dimensions of the embedding. With `--query` it scores
the new samples on dimensions fitted on the atlas; `--exclude-study GSE67402` leaves the study's own
atlas samples out of the fits, so the dimensions have seen nothing of it:

```bash
ecoli_atlas axes --models $M --metadata ../../atlas_data/metadata_v2.tsv \
    --query atlas --query-study GSE67402 --exclude-study GSE67402 \
    --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
    --holdout lag --holdout late_stationary --holdout long_term_stationary \
    --axis medium:medium_class:complex,defined_rich,minimal_plus_hydrolysate,defined_minimal \
    --where treated=false --where standard_rnaseq=true --series samples.csv:replicate:time_h --out atlas/axes
```

Each dimension is scaled so that its first fitted category of the atlas has median 0 and its last
median 1: growth phase from mid exponential (0) to stationary (1), medium from complex (0) to
defined minimal (1).

| Hours | Growth phase | Medium |
|---|---|---|
| 3-6 | 0.21-0.40 | 0.94-1.16 |
| 8 | 0.82-0.90 | 0.77-1.10 |
| 24 | 0.86-0.91 | 0.73-0.75 |
| 48 | 0.91-0.98 | 0.45-0.68 |
| 168 | 0.97-0.99 | 0.44-0.59 |
| 336 | 0.94-0.99 | 0.46-0.52 |

| Look at | What it shows |
|---|---|
| `atlas/axes/fig_axes_phase_by_medium.png` | the atlas on both dimensions, the 26 samples on top, joined in time order per replicate |
| `atlas/axes/fig_axis_phase_series.png`, `fig_axis_medium_series.png` | each dimension against time, one line per replicate |
| `atlas/axes/axis_*_scores.tsv` | the scores of every sample (`role` = query for the new ones) |

The samples start among the atlas's exponential samples in minimal media, move to the stationary
level between 6 and 8 h, and then drift down the medium dimension with time in starvation. On the
continuous dimension the drift stops about halfway (0.45-0.6 at one and two weeks, between the
atlas's defined rich media, 0.32, and minimal media, 0.97): the starved cells look less minimal
than growing ones, not like cells in a complex medium.

## Not covered here

Training the model, the paper figures and the case studies work on the atlas training data, not on
new samples: see the [Ecoli_ATLAS README](../../ecoli_atlas/README.md).
