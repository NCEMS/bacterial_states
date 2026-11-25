# Example: nutrient stress time course in two media (GEO GSE271069)

A second worked example, shorter than
[GSE67402](../GSE67402_growth_starvation_time_course/README.md), which explains each command and
output file. Here the design has two factors (medium and time point), the strain is the one most of
the atlas was built from, and the libraries are clean.

**Data.** *E. coli* K-12 MG1655 grown through exponential and stationary phase in minimal medium
with glucose (`GLU`: time points L, S1, S2) or casamino acids (`CAS`: P1, P2, P3, S1, S2) as the
carbon source, two replicates each: 16 paired-end, reverse-stranded RNA-seq libraries (25 GB of
FASTQ). The GEO series also holds ChIP-seq and IPOD libraries, which are not listed in
`config.yaml`. From "Multiscale regulation of nutrient stress responses in *Escherichia coli* from
chromatin structure to small regulatory RNAs". The study is a forced TEST study of the atlas model:
the model is not fitted on it. `samples.csv` gives medium, time point, replicate, `condition`
(medium_timepoint) and `order` (GLU 1-3, CAS 4-8) per sample.

**Files here.** `config.yaml` and `samples.csv` are tracked. Everything the commands below write
(`sra/`, `results/`, `checkpoints/`, `atlas/`, logs) stays in this folder and is ignored by git.

The numbers below are from the run of 2026-11-23 (inGEST from the reads, then the atlas step with the current model,
`outputs/models`).

`run.sh ingest` (in the `ecoli_ingest` environment) and then `run.sh atlas` (in `ecoli_atlas`) run the commands below.

## 1. inGEST

```bash
cd examples/GSE271069_nutrient_stress_time_course
conda activate ecoli_ingest
ecoli_ingest check                 # 16 accessions -> 16 runs, 32 files, 24.83 GB
ecoli_ingest run --cores 96 --max-downloads 6
```

The run took 3 h 36 min on 32 cores of a shared node, downloads and Centrifuge included (Centrifuge is on in this
`config.yaml` and was the longest step). Every library passed QC (`checkpoints/qc_failures.json` is empty), passed the
species check (Centrifuge), is reverse-stranded (`results/strandedness.tsv`) and was counted: 12-30 million read pairs
on features per library.

**DESeq2** (`results/deseq2/`, padj < 0.05 and at least two-fold):

| Contrast | Genes | Higher in the second |
|---|---|---|
| GLU_L vs GLU_S1 | 465 | 229 |
| GLU_L vs GLU_S2 | 2,493 | 1,169 |
| CAS_P1 vs CAS_P2 | 802 | 488 |
| CAS_P1 vs CAS_P3 | 975 | 525 |
| CAS_P1 vs CAS_S1 | 2,199 | 1,074 |
| CAS_P1 vs CAS_S2 | 2,295 | 1,128 |
| GLU_L vs CAS_P1 | 880 | 403 |
| GLU_S2 vs CAS_S2 | 1,307 | 767 |

## 2. Ecoli_ATLAS

```bash
conda activate ecoli_atlas
M=../../outputs/models
ecoli_atlas embed   --ingest-results . --models $M --out atlas/embedded
ecoli_atlas compare --embeddings atlas/embedded --models $M --out atlas/compared \
    --atlas-metadata ../../atlas_data/metadata_v2.tsv --exclude-study GSE271069 \
    --metadata samples.csv --group-by condition --order-by order
```

`--exclude-study GSE271069` leaves the study's own atlas samples (a TEST study, projected, not fitted) out of the
neighbours; `atlas/compared/group_neighbors.tsv` lists the neighbours of each condition by atlas study, strain,
genotype and treatment.

`embed` printed no warnings: 4.4-26.7 million reads on CDS features, genomic DNA at most 1.1%, every GC fit present.

**Calls** (`atlas/embedded/predictions.tsv`, the same for both replicates of each time point) against the curated
labels of the study (all inferred from the GEO text, not stated):

| Condition | Curated phase | P(stationary) | Phase call | Curated medium | P(minimal) | Medium call |
|---|---|---|---|---|---|---|
| GLU_L | mid exponential | 0.00 | exponential | defined minimal | 1.00 | minimal |
| GLU_S1 | stationary | 0.00 | exponential | defined minimal | 1.00 | minimal |
| GLU_S2 | late stationary | 1.00 | stationary | defined minimal | 1.00 | minimal |
| CAS_P1 | (none) | 0.00 | exponential | minimal + casamino acids | 0.00 | rich |
| CAS_P2 | mid exponential | 0.00 | exponential | minimal + casamino acids | 0.00 | rich |
| CAS_P3 | transition | 0.00 | exponential | minimal + casamino acids | 0.01 | rich |
| CAS_S1 | stationary | 0.41 | exponential (near the boundary) | minimal + casamino acids | 1.00 | minimal |
| CAS_S2 | late stationary | 1.00 | stationary | minimal + casamino acids | 0.00-0.16 | rich |

All 16 are called lab strain. In glucose the model calls the first stationary time point (S1) exponential and only S2
stationary; in casamino acids S1 sits near the boundary (0.41) and S2 is stationary, so in both media the call switches
one time point later than the curated labels, which were inferred from the time-point names. The medium call has no
class for a minimal medium with casamino acids (it is trained on rich against minimal only): the growing CAS samples look
rich (amino acids supplied), CAS_S1 looks minimal, and CAS_S2 rich again.

**Neighbours** (`atlas/compared/`, with the study's own atlas samples left out): the glucose samples' neighbours are
minimal-medium samples throughout, exponential for L and S1 and stationary for S2 (73% of S2's neighbours from
GSE206047); the casamino-acid samples' neighbours are exponential, in minimal media for P1 and S1 and in rich media for
P2, P3 and S2 (GSE295494 for P2, GSE224463 for P3, GSE225096 for S1 and S2). `group_neighbors.tsv` lists them by study, strain, genotype and treatment.

**Glucose against casamino acids** (`ecoli_atlas casestudy glu-cas`, `atlas/casestudy/`): one discriminant on the
embedding separates the 16 samples by medium (AUROC 1.00 in sample). Paired by growth phase (the three phases with both
media), glucose is above casamino acids in 3 of 3 phases scored without that phase, the most three pairs can give
(sign test p = 0.125, permutation p = 0.25). See the [Ecoli_ATLAS README](../../ecoli_atlas/README.md).

**The samples in the atlas, as a page.** `run.sh atlas` ends by writing `atlas/atlas_view.html` (`run.sh view` writes it
alone): one HTML file with a 3-D map of the atlas and the 16 samples of this run in it, coloured by any column of
`samples.csv` (medium, time point, condition) in the "Your samples" tab. See section 3 of
[GSE147611_greAB_deletion](../GSE147611_greAB_deletion/README.md) for what the page offers.
