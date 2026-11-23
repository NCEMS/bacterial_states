# Example: ordered categories as dimensions of the embedding, and the genes that move them

**Question.** Can growth phase, oxygen and carbon source be read off the embedding as ordered dimensions (axes), even
those the model was not trained on, and which genes move each axis?

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas axes (4 axes), then drivers --axis --gsea on each (all 8 seeds)
```

Every sample is scored by an axis fitted without its study; each axis is scaled so that its first fitted category has
median 0 and its last median 1. Held-out categories are only scored (an interpolation or extrapolation test).
Writes `axes/<axis>/` (score table, summary, figures) and `drivers/<axis>/` (as in `atlas_drivers`, plus GO terms).

| Axis | Fitted on | Within-study pairs in order | Held-out categories (median) | Units that move it most |
|---|---|---|---|---|
| growth phase (phase stated in GEO) | mid exponential, transition, stationary | 0.98 (16 studies) | — | `psp`, `iraD`, `ast`, `paa` up; flagella (`fli`, `flg`), `nuo`, ribosomal proteins down |
| oxygen | aerobic, anaerobic fermentation | 0.89 (4 studies) | microaerobic 0.82, anaerobic respiration 0.57 (between the ends) | hydrogenases `hya`, `hyc`, `hydN` up; `sdh`, `cyo`, `fdo` down |
| carbon source (defined minimal, exponential) | sugar, non-sugar | 0.89 (9 studies) | — | `mgl`-`galS`, `glpKF`, `sdh`, `glc`, `lsr`, `prp` towards non-sugar; capsule (`cps`), `glnGL`, `ptsG`, `crr` towards sugar |
| glycerol (defined minimal) | other carbon sources, glycerol | 0.77 (4 studies) | — | `glpKF`, `glpABC`, `glpQT`, `glpD` |

Axis drivers rest on fewer studies than the head drivers (oxygen: 55 samples in 10 studies), so read them with
`study_in_top`; an axis that leans on the residual block also picks up what the residual carries besides it (culture
environment, strain, flagella).

The phase axis uses only samples whose growth phase GEO states (for the sample or for the whole study,
`growth_phase_basis` `stated` or `study_default`), as the model's phase call is scored. Phases inferred from sampling
times are left out: nearly all inferred transition and stationary samples (767 of 800) are GSE206047 (MG1655 with
induced circuits in microplates, 1,077 samples), whose "stationary" samples scored well beyond every other study's
(median 1.8 against 0.5-1.0) and split the stationary category in two. Lag, early exponential and the late stationary levels are not on the axis (early exponential, 58 samples, is too
close to mid exponential to anchor it).
