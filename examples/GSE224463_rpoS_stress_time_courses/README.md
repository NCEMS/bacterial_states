# Example: RpoS and three stress time courses, a study the model never saw (GSE224463)

**Question.** Does the growth-phase call follow the entry into stationary phase in a study withheld from the model,
does it need the stationary-phase sigma factor RpoS to do so, and do two other stresses that induce RpoS (cold, high
osmolarity) move the samples towards stationary phase?

**Data.** GEO GSE224463 ("The timing of transcription of RpoS-dependent genes varies across multiple stresses"):
*E. coli* K-12 BW27786, wild type and Δ*rpoS*, in LB, sampled along three time courses. The study found that RpoS
peaks about 50 minutes after a shift to high osmolarity but takes about 240 minutes after a shift to low temperature.

| Time course | Times (min) | Replicate series per genotype | Samples (wild type + Δ*rpoS*) |
|---|---|---|---|
| entry into stationary phase (`stationary_entry`) | 0-300 | 3 short (`SP_*`, 0-150 min) and 3 long (`LSP_*`, 0-300 min) | 29 + 33 |
| shift to 15 °C (`cold_15C`) | 0-300 | 7 (`short_*`, `long_*`) | 41 + 41 |
| shift to high osmolarity (`high_osmolarity`) | 0-180 | 3 (`A`, `C`, `D`) | 17 + 18 |

179 libraries are in the atlas data. The study is **withheld**: it is in no split, so the model was not fitted on it,
and no seed was chosen with it. `samples.csv` (tracked) gives each sample's genotype, time course, replicate series
and time, read from the GEO sample titles (the genotype in the GEO characteristics field contradicts the titles for
about half of the samples; the titles agree with the curated metadata).

```bash
conda activate ecoli_atlas
bash run.sh        # neighbors, extract, compare, axes; about 3 minutes
```

## 1. Where the study sits in the atlas (`neighbors/`)

```bash
ecoli_atlas neighbors GSE224463 --models ../../outputs/models --metadata ../../atlas_data/metadata_v2.tsv --out neighbors --umap
```

The 179 samples' nearest atlas samples come from 121 studies (most from GSE134667, GSE141694, GSE135706); the
neighbours vote exponential phase (95% of the votes) and rich medium (100%), matching the study's metadata
(`neighbors.tsv`, `summary.tsv`, `study_summary.tsv`, `umap.png`).

## 2. The study's samples as new samples (`atlas/`)

```bash
ecoli_atlas extract GSE224463 --models ../../outputs/models --seeds --out atlas
```

The bundle already holds the embedding and the calls of every atlas sample. `extract` writes those of a study in the
layout of `embed` (`atlas/embedding.tsv`, `predictions.tsv`, and with `--seeds` every seed's in `atlas/seeds/` and
the summary `predictions_seeds.tsv`), so the tools for new samples work on it. They then need `--exclude-study
GSE224463`, or the study's own atlas rows are its nearest neighbours.

## 3. Calls and growth phase along the time courses (`atlas/axes/`)

```bash
ecoli_atlas axes --models ../../outputs/models --metadata ../../atlas_data/metadata_v2.tsv \
    --query atlas --query-study GSE224463 --exclude-study GSE224463 \
    --axis phase:growth_phase:lag,early_exponential,mid_exponential,transition,stationary,late_stationary,long_term_stationary \
    --holdout lag --holdout late_stationary --holdout long_term_stationary \
    --where treated=false --where standard_rnaseq=true \
    --series samples.csv:replicate:time_min:panel --seeds --out atlas/axes
```

The growth-phase dimension is fitted on the atlas without the study (mid exponential 0, stationary 1), on the bundle
and on each of its eight seeds. The fourth field of `--series` (`panel`) draws one panel per time course and genotype.

| Look at | What it shows |
|---|---|
| `fig_axis_phase_series.png` | the growth-phase dimension against time, one panel per time course and genotype, one colour per replicate series, every seed's score as a small dot |
| `fig_heads_seed_votes.png` | how many of the 8 seeds call each sample stationary / minimal / lab (one bar per sample, labelled by replicate series) |
| `fig_heads_series.png` | the three call probabilities against time, each seed as a small dot |
| `axis_phase_scores.tsv`, `axis_phase_seed_scores.tsv` | the scores of every sample, on the bundle and on every seed |

**Entry into stationary phase.** Number of the 8 seeds that call a sample stationary, one number per replicate
series, and the median over samples of the growth-phase dimension (seed median):

| Minutes | Wild type: seeds calling stationary | Δ*rpoS* | Wild type: phase dimension | Δ*rpoS* |
|---|---|---|---|---|
| 0-90 | 0 | 0 | -0.04-0.16 | 0.00-0.11 |
| 120 | 0 (5 samples) | 3, 5, 2 (LSP), 0, 0, 0 (SP) | 0.25 | 0.34 |
| 150 | 0, 0, 0 | 0, 4, 5 | 0.31 | 0.49 |
| 180 | 3 | 5, 4, 5 | 0.42 | 0.54 |
| 240 | 8, 8 | 8 | 0.77 | 0.69 |
| 300 | 8 | 8, 8 | 0.89 | 0.87 |

Both genotypes cross from exponential to stationary between 150 and 240 minutes, and all eight seeds agree from
240 minutes on. The growth-phase dimension rises steadily from 90 minutes and reaches about 0.9, close to the
atlas's stationary median. Δ*rpoS* does not lag: in both kinds of replicate series it is a little ahead of the wild type
at 120-180 minutes (seeds start to call it stationary at 120 minutes in the long series and at 150 in the short
series; the wild type at 180). So the call reads the change of the cells' state on entering stationary phase, not
the activity of RpoS. With the bundle's model alone (seed 1) three samples of each genotype are called stationary,
all at 240 minutes or later.

**Cold and high osmolarity.** No sample of either stress is called stationary by any seed (117 samples), and none is
called minimal or non-lab. On the growth-phase dimension the cold-shifted cells move a little (0.01-0.05 at 0 min,
0.12-0.20 at 30-60 min, back to 0.06-0.09 at 300 min) in both genotypes. Under high osmolarity the wild type moves
briefly towards stationary at 30 minutes (0.30, 0.37, 0.42 in its three series, against 0.08-0.15 at 0 min) and is
back at 60 minutes; Δ*rpoS* moves less (0.16-0.23). The timing matches the study's finding that RpoS peaks about 50
minutes after the osmotic shift, and the smaller move without RpoS suggests that part of the wild type's transient is
RpoS-dependent. It rests on three series per genotype.

## 4. Which atlas samples are close to each time course (`atlas/compared/`)

```bash
ecoli_atlas compare --embeddings atlas --models ../../outputs/models --atlas-metadata ../../atlas_data/metadata_v2.tsv \
    --exclude-study GSE224463 --metadata samples.csv --group-by panel --out atlas/compared
```

Each sample's 15 nearest atlas samples (`neighbors.tsv`), pooled per time course and genotype in
`group_neighbors.tsv` (study, strain, genotype, treatment, growth phase, medium; count, share, mean distance). Read
along the time course (`neighbors.tsv` joined with `samples.csv`), the neighbours follow the treatment:

- **Before each shift (0-90 min of the stationary and cold time courses, 0 min of high osmolarity)**: untreated K-12
  growing exponentially in LB, mostly GSE134667 (nucleoid / HU study, its untreated arms), GSE202549 (Δ*cyaA* with
  cAMP) and GSE141694.
- **Cold, from 120 minutes on**: cold-treated samples of other studies, GSE222248 (its cold arm), GSE267531 (the
  transcriptional response to low temperature) and GSE141694 (its cold-stress arm), in both genotypes. These are
  three of the five atlas studies with cold-treated samples: the embedding recognises a cold shock it was never
  trained to call.
- **Entry into stationary phase, 120-180 minutes**: the wild type's neighbours are mostly GSE123048 (MG1655 sampled
  along a growth curve, phase not labelled); Δ*rpoS*'s are GSE201666 (MG1655 stressed with antidepressants and
  antibiotics) and GSE345300 (MG1655 isolates evolved towards antibiotic persistence), both without a stated phase.
- **Entry into stationary phase, 240-300 minutes**: the wild type's neighbours include samples labelled stationary in
  three studies (GSE165634, GSE271069, GSE111874); Δ*rpoS*'s stay with GSE345300 and GSE201666. Both are called
  stationary, but the mutant reaches a different place among the atlas's samples than the wild type, which fits a
  stationary phase without its sigma factor.
- **High osmolarity**: the atlas holds only two small studies with a hyperosmotic treatment (10 samples), and none of
  them is among the neighbours. At 30-60 minutes the wild type is closest to acid-stressed samples (GSE134667, its
  acid arm) and antibiotic-treated samples, Δ*rpoS* to samples treated with membrane-active antimicrobials (GSE160082)
  or silver nanoparticles (GSE151954). These are other stresses, not the same treatment: the atlas cannot say more about
  an osmotic shock than that it looks like a stress response.

The windows hold 3-20 samples per genotype, and many neighbour studies have no stated phase; read these as
leads.

## Not covered here

New samples from reads: `../GSE147611_greAB_deletion` (smallest), `../GSE67402_growth_starvation_time_course` (a time
course explained step by step).
