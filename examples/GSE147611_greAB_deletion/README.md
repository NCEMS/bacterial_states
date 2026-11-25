# Example: wild type against a greA greB deletion (GSE147611)

The smallest read-based example: six libraries, three replicates of each of two conditions, from GEO accessions to
counts, a DESeq2 contrast and the atlas calls, with every command and what came out of it.

**Data.** *E. coli* K-12 W3110 wild type and its ΔgreA ΔgreB mutant (no Gre factors, which rescue backtracked RNA
polymerase) in LB, mid-exponential phase; three biological replicates each, paired-end, rRNA-depleted, about 2 million
read pairs per library (GEO GSE147611, "Nascent RNA sequencing identifies a widespread sigma70-dependent pausing
regulated by Gre factors in bacteria"; the series also holds nascent-RNA libraries, which are not used). GSE147611 is a
TEST study of the atlas model: the model was not fitted on it.

**Files here.** `config.yaml` (the inGEST run configuration) and `samples.csv` (condition and replicate per sample) are
tracked. Everything the commands below write (`sra/`, `results/`, `checkpoints/`, `benchmarks/`, `atlas/`, logs) stays
in this folder and is ignored by git. `run.sh ingest` and `run.sh atlas` run the commands of sections 1 to 3.

The numbers below are from the run of 2026-11-23 with the current model (`outputs/models`).

## 1. inGEST: reads to counts

```bash
cd examples/GSE147611_greAB_deletion
conda activate ecoli_ingest
ecoli_ingest check                 # tools, resources, accessions -> 6 runs, 2.6 GB to download
ecoli_ingest run --cores 8
```

The workflow resolves the GSM accessions to SRA runs, fetches the reads into `sra/`, and goes through fastp,
`vg giraffe`, the strandedness call, featureCounts, the QC record, DESeq2 and MultiQC. Centrifuge is off
(`centrifuge: false`) to keep the run short; set it to `true` to screen the reads. The run took 51 minutes on 8 cores
of a node shared with another inGEST run; alignment (`vg giraffe`, about 4 minutes per library) was the longest step.

| Look at | What it tells you |
|---|---|
| `results/multiqc_report.html` | all QC in one report |
| `checkpoints/qc_failures.json` | samples stopped by a QC gate: none |
| `results/strandedness.tsv` | every library is reverse-stranded (0.98-0.99 of reads on the antisense strand) and counted |
| `results/all_samples_raw_counts.txt` | feature x sample counts: 1.9-2.5 million read pairs on features per library |
| `results/<sample>/qc/<sample>.qc.json` | the QC record of a sample |
| `results/deseq2/wt_vs_greAB_deseq_results.tsv`, `pca_plot.png` | the contrast (log2 fold change = ΔgreAB against wild type) and the PCA |

**DESeq2.** 691 genes differ at padj < 0.05 and at least two-fold, 324 higher and 367 lower in the mutant. The
strongest: `tnaA` (80-fold lower), `ppsA` (11-fold higher), `nanM` and `nanC` (21- and 41-fold lower), `gatBC`,
`gntT`, `dsdA` and the `sgrS`/`sgrT` pair (9- to 14-fold lower), `recE` (82-fold higher).

## 2. Ecoli_ATLAS: counts to calls and atlas neighbours

```bash
conda activate ecoli_atlas
M=../../outputs/models
ecoli_atlas embed --ingest-results . --models $M --out atlas/embedded
ecoli_atlas compare --embeddings atlas/embedded --models $M --atlas-metadata ../../atlas_data/metadata_v2.tsv \
    --metadata samples.csv --group-by condition --exclude-study GSE147611 --out atlas/compared
```

`embed` writes the embedding, the three calls (`atlas/embedded/predictions.tsv`) and the sample table; it printed no
warnings (1.8-2.5 million reads on CDS features per library, no genomic DNA, every GC fit present). `compare` finds
each sample's 15 nearest atlas samples and summarises them per sample (`atlas/compared/summary.tsv`, `neighbors.tsv`)
and per condition (`group_summary.tsv`, `group_neighbors.tsv`, `fig_head_by_group`, `fig_neighbor_labels`,
`fig_query_distances`).

**`--exclude-study GSE147611`.** The atlas holds these six libraries already (projected, not fitted: a TEST study),
so without the option each sample's nearest atlas sample is its own atlas copy, at distance 0 (the recount here
reproduces the counts the atlas was built from), and the study's own copies take three of every sample's 15
neighbours. With it the study's 6 atlas samples are left out, and every neighbour, vote and novelty value comes from
the other 7,725.

| | Wild type | ΔgreAB |
|---|---|---|
| growth phase | exponential (P(stationary) < 1e-6) | exponential (2e-6) |
| medium | rich (P(minimal) < 1e-5) | rich (< 1e-5) |
| lab strain | lab (P = 0.99999) | lab (0.99999) |
| neighbours' votes (phase, medium, lab strain) | exponential, rich, lab (100% of the labelled neighbours) | exponential, rich, lab (100%) |
| distance to the nearest atlas sample (mean) | 0.09 | 0.13 |
| novelty percentile (mean) | 0.23 | 0.47 |

All six calls agree with the metadata (LB, mid-exponential, K-12), and the deletion changes none of them: Gre factors
are not part of what the three calls are about. Few of the wild type's neighbours have a stated growth phase (`wt_rep1`
has none among its 15, so it has no phase vote); the medium and lab-strain votes rest on 12-15 labelled neighbours per
sample. The novelty percentile compares a sample's distance to its 15th nearest atlas sample with that of atlas samples
to their 15th nearest sample of another study; 0.23 and 0.47 are typical of atlas samples (the median is 0.5), and the
mutant sits further from the atlas than the wild type.

**The neighbours by condition** (`group_neighbors.tsv`: one row per condition and atlas study, strain, genotype class,
treatment, growth phase and base medium, with the number and share of the condition's 45 neighbours, how many of its
three samples have one there, and their mean distance):

| ΔgreAB: neighbours | n | Wild type: neighbours | n |
|---|---|---|---|
| GSE248682, W3110 *rpoC*-His (tagged RNA polymerase), LB | 9 | GSE76916, MG1655, overexpression of a native gene (DicF), LB | 7 |
| GSE135906, MG1655, LB | 9 | GSE136101, BW25113, LB | 7 |
| GSE262812, MG1655 Δ*hfq*, LB (5 untreated, 5 phage) | 10 | GSE234254, BW25113, translation antibiotic, LB | 7 |
| GSE221825, C-5691 deletion mutant | 5 | GSE76916, MG1655 vector control, LB | 5 |
| GSE59377, MG1655 point mutant, LB | 4 | GSE143855, MG1655 deletion mutant, LB | 4 |
| GSE109025 / GSE215300 / GSE220559, antibiotics against translation or transcription | 8 | GSE123048, MG1655, LB | 4 |

Every one of these is exponential growth in a rich medium. The wild type's neighbours are mostly untreated K-12 in LB,
and closer (mean distance 0.10-0.12 for its main studies against 0.15-0.24 for the mutant's). The mutant's include
samples with a perturbed transcription and RNA machinery: a tagged RNA polymerase subunit, an *hfq* deletion and an
antibiotic against transcription. That fits a strain without Gre factors, but it rests on 15 neighbours per sample;
read it as a lead, not as a result.

## 3. Your samples in the atlas, as a page

```bash
D=../../atlas_data
ecoli_atlas view --models $M --metadata $D/metadata_v2.tsv --counts $D/atlas_v2_counts.tsv --qc $D/sample_qc.tsv \
    --studies $D/metadata_v2_studies.tsv --sample-text $D/metadata_v2_sample_text.tsv --time-series $D/time_courses.tsv \
    --modules $D/modules/modules.tsv --overlay-counts results/all_samples_counts.tsv \
    --embeddings atlas/embedded=GSE147611_inGEST --overlay-metadata samples.csv --out atlas/atlas_view.html
```

`run.sh atlas` ends with this command and `run.sh view` runs it alone (about two minutes, most of it the UMAP layouts;
`run.sh` leaves out the tables that are not there). It writes one HTML file of 15 MB that opens in a web browser without
a server or a network connection: the 7,731 atlas samples as a 3-D map, and the six samples of this inGEST run in it as
diamonds. Nothing is retrained or rebuilt for this: the samples were placed by `embed` in step 2, and the page projects
them onto the same layouts as the atlas.

- **Your samples** tab: the six samples with their calls. `samples.csv` is optional; with it, "colour by" offers its
  columns (here `condition`: wild type against the deletion), and "Turn the camera to them" brings them to the middle.
- **Click a diamond**: its calls, its QC values as percentiles of the atlas, the row of the sample sheet and its
  nearest atlas samples. Each sample's nearest neighbour is its own atlas copy at distance 0.000 (the atlas holds these
  libraries); the first neighbours from other studies follow at 0.07 to 0.13.
- **Example** menu: "Worked example: wild type and a greA greB deletion (GSE147611)" draws the atlas copies of the
  study; the other ten examples are single studies of the atlas (growth-phase series, a biofilm, media, strains, the
  held-out studies).
- **Regulon** box (needs the module table of `ecoli_atlas download modules`): type a regulator or a gene to colour the
  map by the expression of a RegulonDB set. Because the page was given the run's count table (`--overlay-counts`), the
  diamonds are coloured too. In this run, among the sets of ten or more genes in which the three mutant libraries all
  lie on one side of the three wild-type ones, the largest differences are lower NanR-, GntR- and AgaR-repressed genes
  in the mutant (by 1.7, 1.1 and 0.9 SD) and higher CdaR-, Nac- and FadR-activated and LexA-repressed genes (by 0.3 to
  0.4 SD). These are means of standardised expression over a set, from three libraries a side: use them to find what
  to look at in the DESeq2 table, not as a test.

## Not covered here

A time course and the ordered dimensions of the embedding: see `../GSE67402_growth_starvation_time_course`. Training
the model and the case studies: see the [Ecoli_ATLAS README](../../ecoli_atlas/README.md).
