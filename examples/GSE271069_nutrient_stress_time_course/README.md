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

> **Results pending.** This example is being re-run from the reads with the current counting and
> the atlas model trained with this study in TEST; the numbers will be filled in from that run.

## 1. inGEST

```bash
cd examples/GSE271069_nutrient_stress_time_course
conda activate ecoli_ingest
ecoli_ingest check                 # 16 accessions -> 16 runs, 32 files, 24.83 GB
ecoli_ingest run --cores 96 --max-downloads 6
```

## 2. Ecoli_ATLAS

```bash
conda activate ecoli_atlas
M=../../outputs/models
ecoli_atlas embed   --ingest-results . --models $M --out atlas/embedded
ecoli_atlas compare --embeddings atlas/embedded --models $M --out atlas/compared \
    --metadata samples.csv --group-by condition --order-by order
```

The glucose against casamino acids contrast of this study on the atlas embedding is
`ecoli_atlas casestudy glu-cas` (see the [Ecoli_ATLAS README](../../ecoli_atlas/README.md)).
