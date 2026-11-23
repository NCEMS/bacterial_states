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
| growth phase | early exponential … stationary | 0.88 (21 studies) | late stationary 0.65, long-term stationary 0.61 (not beyond stationary) | `psp`, `iraD`, `hya` up; ribosomal proteins, `nuo` down |
| oxygen | aerobic, anaerobic fermentation | 0.89 (4 studies) | microaerobic 0.82, anaerobic respiration 0.57 (between the ends) | hydrogenases `hya`, `hyc`, `hydN` up; `sdh`, `cyo`, `fdo` down |
| carbon source (defined minimal) | sugar, non-sugar | 0.84 (9 studies) | — | `glpKF`, `sdh`, `glc`, `gar` towards non-sugar; `ptsG`, `crr`, capsule towards sugar |
| glycerol (defined minimal) | other carbon sources, glycerol | 0.77 (4 studies) | — | `glpKF`, `glpABC`, `glpQT`, `glpD` |

Axis drivers rest on fewer studies than the head drivers (oxygen: 55 samples in 10 studies), so read them with
`study_in_top`; an axis that leans on the residual block also picks up what the residual carries besides it (culture
environment, strain, flagella).
