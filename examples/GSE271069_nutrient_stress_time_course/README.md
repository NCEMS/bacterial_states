# Example: nutrient stress time course in two media (GEO GSE271069)

> **`ingest-update` branch.** The text and numbers below describe the run made with the previous
> counting method (branch `main`). On this branch inGEST counts strand-specific libraries only,
> with stranded counting, a larger feature set and new feature ids
> ([docs/counting.md](../../docs/counting.md)), always trims paired-end adapters, and its counts do
> not match the published atlas model, so the Ecoli_ATLAS sections do not apply to counts made
> here. These libraries are reverse-stranded. The example has not been re-run from the reads on this branch.

A second worked example, shorter than
[GSE67402](../GSE67402_growth_starvation_time_course/README.md), which explains each command and
output file. Here the design has two factors (medium and time point), the strain is the one most of
the atlas was built from, and the libraries are clean.

**Data.** *E. coli* K-12 MG1655 grown through exponential and stationary phase in minimal medium
with glucose (`GLU`: time points L, S1, S2) or casamino acids (`CAS`: P1, P2, P3, S1, S2) as the
carbon source, two replicates each: 16 paired-end RNA-seq libraries (25 GB of FASTQ). The GEO series
also holds ChIP-seq and IPOD libraries, which are not listed in `config.yaml`. From "Multiscale
regulation of nutrient stress responses in *Escherichia coli* from chromatin structure to small
regulatory RNAs". Not in the atlas training data. `samples.csv` gives medium, time point,
replicate, `condition` (medium_timepoint) and `order` (GLU 1-3, CAS 4-8) per sample.

The numbers are from the run of 2026-11-14 on `main`, made without paired-end adapter detection
(`fastp_detect_adapter_for_pe` off), which is how the atlas training counts were processed.

## 1. inGEST

```bash
cd examples/GSE271069_nutrient_stress_time_course
conda activate ecoli_ingest
ecoli_ingest check                 # 16 accessions -> 16 runs, 32 files, 24.83 GB
ecoli_ingest run --cores 96 --max-downloads 6
```

- All 16 samples pass both QC gates (`checkpoints/qc_failures.json` is empty). fastp keeps 91-99 %
  of the reads (median 96 %); Centrifuge assigns 42-82 % of the classified reads to *E. coli*
  (median 79 %); 96.3-99.9 % of the reads align.
- One library (`CAS_S2_rep2`) shows the adapter problem described in the GSE67402 example: 31 % of
  its classified reads are "synthetic construct" and 42 % *E. coli*, and 96.3 % of its reads
  align; the other 15 libraries have 0.4-3.1 % synthetic construct. It still passes the 30 % gate.
  In an earlier run with `fastp_detect_adapter_for_pe: true` the same library had 0.6 % synthetic
  construct and 64 % *E. coli*. That option changes little else here: the Ecoli_ATLAS class calls
  below are the same for all 16 samples, and the DESeq2 gene counts differ by at most 11 genes.
- DESeq2 (`results/deseq2/`, padj < 0.05 and at least two-fold): in glucose 377 genes differ
  between L and S1 and 2186 between L and S2; in casamino acids 674, 840, 1997 and 2098 genes differ
  between P1 and P2, P3, S1, S2. Between the media, 748 genes differ at the first time point and
  1087 at the last.

## 2. Ecoli_ATLAS

```bash
conda activate ecoli_atlas
FT=../../atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt
M=../../outputs/models ; A=../../outputs/atlas

ecoli_atlas prepare --ingest-results results --metadata samples.csv --models $M --feature-table $FT --out atlas/prepared
ecoli_atlas embed   --counts atlas/prepared/counts.tsv --models $M --feature-table $FT --out atlas/embedded
ecoli_atlas compare --embeddings atlas/embedded/final_embedding.csv --atlas $A --out atlas/compared -k 15 --umap \
    --metadata atlas/prepared/samples.tsv --group-by condition --order-by order
ecoli_atlas compare --embeddings atlas/embedded/final_embedding.csv --atlas $A --out atlas/compared_medium -k 15 --umap \
    --metadata atlas/prepared/samples.tsv --group-by medium
```

`--group-by condition --order-by order` summarises per medium and time point, in the order of the
`order` column of `samples.csv` (glucose first, then casamino acids; the line in
`fig_umap_groups` and `distance_to_previous` therefore jump from `GLU_S2` to `CAS_P1`).

- **Gene coverage** 4367 of 4385 model genes (99.6 %).
- **Growth phase.** Every L, S1 (glucose) and P1-P3 (casamino acids) sample is called exponential
  with probability below 0.001 of being stationary; both `GLU_S2` and both `CAS_S2` samples are called
  stationary (0.96-1.00). `CAS_S1` is split: one replicate exponential, one at 0.561. So the
  encoder places the transition at the last time point in both media; the samples the authors
  label S1 still look exponential to it.
- **Medium.** 15 of 16 are called defined, which is correct for both media; `CAS_S2_rep2` is
  called complex.
- **Lab strain.** All 16.
- **Relationships** (`atlas/compared/fig_query_distances.png`). Replicates are 0.001 apart
  (median cosine distance), samples of the same medium at different time points 0.026, samples of
  different media 0.035. The largest step is `GLU_S1` to `GLU_S2` (0.62 between centroids); the
  casamino-acid samples move gradually (0.01, 0.00, 0.04, 0.03).
- **Neighbours.** Exponential glucose samples sit next to GSE64848 and GSE66481 (MG1655 in M9 with
  glucose; novelty percentile 0.01-0.16, i.e. well inside the atlas). `CAS_P1` sits next to
  GSE103421 (MG1655 in MOPS with glucose). The later casamino-acid samples and `GLU_S2` sit next to
  GSE111874, GSE224463 and GSE206609, with higher novelty (up to 0.69 for `GLU_S2`).
- **Same calls as before.** These 16 samples were embedded earlier from a count table made with
  the original pipeline (`outputs/predictions_paper1`); the growth-phase and medium calls from this
  run agree with those for 16 of 16 samples.

## 3. The neighbouring studies

```bash
ecoli_atlas neighbors GSE64848 GSE111874 GSE224463 GSE206609 --atlas $A --models $M --out atlas/neighbors_top_studies --umap
```

GSE64848 is a training study whose neighbours agree with its curated growth phase and medium for
every sample; for GSE111874 and GSE224463 the agreement is 42-44 % for growth phase, so the
neighbourhood of the stationary samples is less clear-cut than that of the exponential ones
(`atlas/neighbors_top_studies/study_summary.tsv`).

A glucose-versus-casamino-acids axis for these samples is one of the built-in case studies:
`ecoli_atlas casestudy glu-cas` (see the [Ecoli_ATLAS README](../../ecoli_atlas/README.md)).
