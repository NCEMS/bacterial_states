# Genotypes behind the flagellar on / off split of LB lab-strain cultures

The [atlas_lab_strain_LB_variation](../atlas_lab_strain_LB_variation/README.md) example (fig_batch_tracking) shows that
LB exponential cultures of the lab strains follow the growth rate for most regulons, but the flagellar regulons (FlhDC,
FliA) are on in some studies and off in others. This example takes the reads of those 116 samples, calls their
genotypes, and tests which genotypes go with each sample's distance from the regulon-vs-ribosome trend lines.

## Run

```bash
conda activate ecoli_atlas
bash run.sh figures                 # quick: minutes, from the atlas tables and sample_states.tsv
SCR=<big disk> bash run.sh all 6    # full: reads -> breseq -> loci -> tables -> figures (rewrites sample_states.tsv)
```

The quick path needs the atlas tables (`../../atlas_data/`), the module table (`ecoli_atlas download modules`) and the
bundle (`MODELS=<bundle>` for another one): `scripts/samples.py` takes each sample's mean log2 expression of the
ribosomal proteins and of nine RegulonDB regulons from the atlas, `scripts/residuals.py` its residual from the line
through the study means (`results/residuals.tsv`), and the figure scripts join them with the tracked per-sample states
in `sample_states.tsv`.

The full path rebuilds `sample_states.tsv` from the reads. Steps (`bash run.sh <step> [N]`, N samples at a time):
`reference` (bowtie2 indexes of NC_000913.3 from the inGEST core bundle, which also holds the GenBank file breseq
uses), `fetch` (up to 20 M reads per sample with sra-tools, ENA as
fallback, fastp), `breseq`, `loci`, `collect`, `figures`. Reads, breseq runs and BAMs go to `$SCR` (default
`scratch/`): several hundred GB. breseq takes 16 threads per run; most runs take under an hour, the largest
libraries 5-8 hours. Size N by the job's memory, not the node's (20 runs at once did not fit in 72 GB). It needs sra-tools, fastp and samtools (the ecoli_ingest
environment), bowtie2, R and breseq 0.40 on PATH.

## Method

- **Samples** (`samples.tsv`, `reads.tsv`): the LB, mid-exponential, untreated wild-type lab-strain samples of
  fig_batch_tracking, plus two controls from the read-based examples (GSE271069 MG1655, GSE147611 W3110).
- **breseq** (`scripts/breseq.sh`): consensus mode, `-t` (RNA-seq coverage does not follow breseq's coverage model),
  `--junction-score-cutoff 0`. Chosen on a test of 11 samples: polymorphism mode added thousands of noise calls per
  sample, and consensus mode agreed best between replicates. Five large libraries (`junction_k50` in `reads.tsv`) cap
  the alignment to junction candidates at 50 hits per read (default 2000), where that step passed 1 TB.
- **Foreign reads**: calls with another call within 1 kb are clusters of reads of a second E. coli in the library
  (where the K-12 genes are silent); they are left out of the genotypes, and samples with >= 20 are `mixed` and left
  out of the tests (shown hollow in the figures).
- **Lineage** (`scripts/stratify.py`): 13 BW25113-lineage markers (in >= 80% of BW25113 samples, <= 10% of MG1655).
  Three GSE267531 samples labelled MG1655 are BW25113-lineage.
- **Loci** (`scripts/loci_bam.sh`, `scripts/loci_states.py`): all reads aligned with bowtie2 `--local` to four short
  regions, so reads across IS ends keep their clipped parts, which are then placed on the genome.
  - IS1H, 300 bp upstream of flhD: present (reads through an IS end) or absent (reads joining the two flanks with
    one copy of the 9-bp target site: never inserted).
  - IS1I inside crl (crl is a pseudogene in the reference because of it) and IS5U between ychE and oppA.
  - The fim switch (4542682-4542995, not annotated in the reference), aligned in both orientations; the reference
    orientation is OFF.
  An IS state needs >= 2 reads; RNA-seq gives no reads across an IS end where the flanks are not transcribed.
- **MG1655 stocks** (Freddolino et al. 2012, J Bacteriol 194:303): CGSC 7740-type (ATCC 700926 / CGSC 7740, and
  the reference NC_000913.3): IS1I, IS5U, glpR and gatC frameshifts; ATCC 47076-type (ATCC 47076 / CGSC 6300, the
  isolate that was sequenced): none of them. BW25113 and MG1693 share the ATCC 47076 states, so those are ancestral.
  IS1H is not a stock marker; it is in the reference and absent from BW25113 and MG1693 (Barker et al. 2004, J
  Bacteriol 186:7529: a poorly motile CGSC MG1655 stock without the IS1 upstream of flhD).
- **rpoS**: codon 33 alleles and deletions (depth inside rpoS against nlpD).
- **Excluded studies** (`scripts/common.py`, kept and flagged in `sample_states.tsv`): GSE224463 (its genotype labels
  are swapped: the LSP_A 0 and 90 min samples titled wild-type lack rpoS, those titled ΔrpoS have it) and GSE64021
  (an abnormal MG1655 clone).
- **Tests** (`scripts/figlib.py`): samples are nested in studies and IS1H hardly varies within a study, so the
  numbers to quote are the Mann-Whitney test over study means and a permutation of the labels between whole studies;
  sample-level and mixed-model p-values are in the tables too.

## Results

| figure | what it shows |
|---|---|
| `fig_regulon_FlhDC`, `fig_regulon_FliA` | each sample's regulon against the ribosomal proteins; fill = genotype group, black ring = IS1H present |
| `fig_is1h_FlhDC`, `fig_is1h_FliA` | residual by IS1H state (`stats_is1h.tsv`: all nine regulons, K-12 and MG1655 only) |
| `fig_mg1655_stock` | MG1655: CGSC 7740-type minus ATCC 47076-type for the nine regulons (`stats_mg1655_stock.tsv`) |
| `fig_genotype_grid` | one row per study: IS1H, IS1I, IS5U, glpR, gatC and the FlhDC residual (`genotype_grid.tsv`) |

- IS1H goes with the flagellar regulons: present in 11 samples of 6 studies (median FlhDC residual +1.94), absent in
  16 of 10 (-0.58). FlhDC: study-level p = 0.016, permutation p = 0.006, R² = 0.47; FliA: p = 0.031 and 0.005,
  R² = 0.58. Within MG1655 alone FlhDC p = 0.041 and 0.015. One exception: GSE328962 has IS1H and low FlhDC.
- The two MG1655 stocks differ in Crp (CGSC 7740-type lower, q = 0.040) and IHF (q = 0.042), probably RpoS and ArcA
  (q < 0.1), but not clearly in the flagellar regulons (q = 0.44): GSE126710 is ATCC 47076-type with IS1H and high
  FlhDC; GSE328962 and GSE262812 are CGSC 7740-type with low FlhDC.
- No variant in crp or cyaA; flagellar-gene variants are rare and private to single studies.

`results/` also has the mutation calls (`mutations.tsv`, `genotypes.tsv`), the target genes (flagella, flhDC
regulators, cAMP-Crp, stress and growth regulators, lab-stock loci), lineage, loci and the associations of every
variable mutation with the residuals, over all K-12 samples and within each lineage (full path only).
