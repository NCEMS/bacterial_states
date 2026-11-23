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

The numbers below are from the inGEST run of 2026-11-22 and the atlas step of 2026-11-23 with the current model
(`outputs/models`). `run.sh ingest` and `run.sh atlas` run the commands of sections 1-3.

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
ecoli_atlas embed --ingest-results . --models $M --fill-missing-gc study --seeds --out atlas
```

`embed` reads the counts, antisense counts and QC records of the run directory and writes, per
sample, the embedding (`atlas/embedding.tsv`: phase, medium and lab-strain scores and the 24-d
residual), the three calls (`atlas/predictions.tsv`) and the sample table (`atlas/samples.tsv`).
With `--seeds` it also embeds the samples with each of the bundle's eight seed models (`atlas/seeds/seed_<n>/`)
and writes `atlas/predictions_seeds.tsv`: for every call the median, minimum and maximum probability over the seeds
and the number of seeds that make it (`p_stationary_n_called`, ...). It printed two warnings, both worth reading:

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
| 3 | 1.00 | 0.31 | 0.01 |
| 4 | 0.00 | 0.00 | 0.00 |
| 5 | (stopped by QC) | 0.00 | 0.51 |
| 6 | 0.01 | 0.02 | 0.00 |
| 8 | 1.00 | 1.00 | 1.00 * |
| 24 | 1.00 *+ | 1.00 | 1.00 *+ |
| 48 | 1.00 *+ | 1.00 * | 1.00 *+ |
| 168 | 1.00 * | 1.00 | 1.00 |
| 336 | 1.00 | 1.00 | 1.00 |

Every sample from 8 h on is called stationary, and 9 of the 11 samples from 3 to 6 h exponential; the model places
the switch between 6 and 8 hours, as DESeq2 does, although the GEO text labels the 8 h samples mid-exponential. The
exceptions are `t003h_rep1` (called stationary) and `t005h_rep3` (0.51, on the boundary). They are seed-dependent: the
bundle's model is the seed chosen on out-of-fold scores (seed 1), and over the eight seeds (`predictions_seeds.tsv`)
`t003h_rep1` is called stationary by 2 and `t005h_rep3` by 1, every other 3-6 h sample by none; from 8 h on every
sample is called stationary by all eight seeds except `t008h_rep2` (7). The phase score (`z1` in `embedding.tsv`) tells
the same: -0.8 to 0.4 for the growing samples, -1.84 for `t003h_rep1`, and -2.12 to -1.86 from 8 h on.

**Medium and strain.** All 26 samples are called lab strain (REL606 is a lab strain). The medium
call is minimal through 48 h and moves to rich with starvation: two of three one-week samples
(`p_minimal` 0.08 and 0.09) and all two-week samples (0.001-0.003) are called rich. This follows the time spent starving, not the entry into
stationary phase (the 8 h samples are still clearly minimal), and it is not caused by depth or by
the filled GC bias: the deep one-week libraries stay rich when thinned to 50,000 reads or given the
study-median GC. The atlas has no minimal-medium samples starved for days; in long starvation the
biosynthetic genes that mark minimal medium shut down, and the cells may also live on what dead
cells release.

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
    --where treated=false --where standard_rnaseq=true --series samples.csv:replicate:time_h --seeds --out atlas/axes
```

Each dimension is scaled so that its first fitted category of the atlas has median 0 and its last
median 1: growth phase from mid exponential (0) to stationary (1), medium from complex (0) to
defined minimal (1). With `--seeds` every axis is also fitted on the embedding of each of the eight seeds, the same
way and on the same scale; the score tables get `ordinal_seed_median` / `_min` / `_max`, `axis_<name>_seed_scores.tsv`
has every seed's score, and the series figures draw each seed's score as a small dot beside the sample. The table
gives the range over the replicates of the bundle's model (seed 1) and of the median over the seeds:

| Hours | Growth phase, seed 1 | Growth phase, seed median | Medium, seed 1 | Medium, seed median |
|---|---|---|---|---|
| 3-6 | 0.03-0.45 | 0.13-0.38 | 0.72-0.97 | 0.94-1.15 |
| 8 | 0.45-0.49 | 0.63-0.76 | 0.46-0.65 | 0.76-1.00 |
| 24 | 0.49-0.63 | 0.80-0.83 | 0.44-0.71 | 0.70-0.76 |
| 48 | 0.67-0.72 | 0.87-0.94 | 0.63-0.78 | 0.69-0.81 |
| 168 | 0.70-0.72 | 0.91-0.93 | 0.38-0.72 | 0.47-0.77 |
| 336 | 0.68-0.73 | 0.86-0.95 | 0.31-0.38 | 0.41-0.42 |

| Look at | What it shows |
|---|---|
| `atlas/axes/fig_axes_phase_by_medium.png` | the atlas on both dimensions, the 26 samples on top, joined in time order per replicate |
| `atlas/axes/fig_axis_phase_series.png`, `fig_axis_medium_series.png` | each dimension against time (the time points as categories, the replicates side by side, one colour each), each seed's score as a small dot |
| `atlas/axes/axis_*_scores.tsv` | the scores of every sample (`role` = query for the new ones), with the median, min and max over the seeds |
| `atlas/axes/axis_*_seed_scores.tsv` | every seed's score of every sample |
| `atlas/axes/fig_heads_series.png` | the three calls (probabilities from `atlas/predictions.tsv`, 0-1) against time, the replicates side by side at each time point, each seed's call as a small dot |
| `atlas/axes/fig_heads_seed_votes.png` | how many of the eight seeds make each call: one bar per sample, red = share of seeds calling stationary / minimal / lab, blue = the rest |
| `atlas/axes/fig_heads_phase_by_medium.png` | P(stationary) against P(minimal): the atlas (GSE67402 left out) coloured by its labels, the 26 samples on top |

The samples start among the atlas's exponential samples in minimal media, move up the growth-phase dimension between
6 and 8 h and further over the first two days, and then drift down the medium dimension with time in starvation: the
long-starved cells look much less minimal than growing ones (with seed 1, 0.31-0.38 at two weeks, close to the atlas's
defined rich media at 0.24; defined minimal 0.97, complex -0.02).

**How far they go depends on the seed.** Seed 1 is the lowest of the eight seeds on the growth-phase dimension from
8 h on: it puts the samples of 48 h and later at about 0.7, below the atlas's stationary median of 1, while the other
seeds put them at 0.73-1.05 and the seed median is 0.86-0.95. So the plateau below 1 is mostly a property of seed 1,
not of the samples. On the medium dimension seeds 1 and 3 are the lowest for the growing samples (medians per time
point 0.82-0.93 and 0.89-1.11, the other seeds 0.96-1.19), and the drop with starvation is in every seed but seed 5
(two weeks: 0.20-0.52 for the others, 0.84 for seed 5). The range over the seeds of one sample is 0.22-0.47 on the
growth-phase dimension and 0.22-0.64 on the medium dimension, widest from 8 h on, where the samples are least like
the training data.

**The calls on their 0-1 scale.** `axes` also draws the calls of section 2 (`fig_heads_series`,
`fig_heads_phase_by_medium`) when `--query` is an `embed` folder with its `predictions.tsv`. They are the
probabilities the heads give, not dimensions fitted on ordered categories:

| Hours | P(stationary) | P(minimal) |
|---|---|---|
| 3-6 | 0.00-0.02 (`t003h_rep1` 1.00, `t003h_rep2` 0.31, `t005h_rep3` 0.51) | 1.00 |
| 8-48 | 1.00 | 1.00 |
| 168 | 1.00 | 1.00, 0.08, 0.09 |
| 336 | 1.00 | 0.001-0.003 |

Both calls saturate, and so do most seeds (the small dots of `fig_heads_series`): the spread is in the number of seeds
on each side of 0.5 more than in the probabilities. Every sample from 8 h on has P(stationary) 1.00, so the probabilities do not show the further
movement over the first two days that the phase dimension shows (seed median 0.63-0.76 at 8 h to 0.87-0.94 at 48 h). The medium
call stays at 1.00 through 48 h and then drops to rich (the two rich one-week samples by 6 of 8 seeds, the two-week
samples by 7 of 8), while the medium dimension drifts down gradually. The atlas
looks the same on this map: almost all of its samples sit in the four corners. Use the calls for the class and the
dimensions for how far a sample has gone.

## Not covered here

Training the model, the paper figures and the case studies work on the atlas training data, not on
new samples: see the [Ecoli_ATLAS README](../../ecoli_atlas/README.md).
