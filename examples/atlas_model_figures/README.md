# Example: figures of the training data and of the atlas model

**Question.** What is in the training data, how well does the model call growth phase, medium and lab strain on
held-out studies, and how much study (batch) structure is left?

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas figures ... --models ../../outputs/models --out figures --strict
```

Writes `figures/` (one PDF and PNG per figure, the tables beside them, `figure_metrics.json`). `--strict` fails the run
when the embedding made from the tables differs from the bundle's `embedding.tsv` or the TEST metrics differ from the
manifest; both checks pass (embedding within 5e-5 over 7,731 samples).

| Figure | What came out (bundle `outputs/models`, run 2026-11-23) |
|---|---|
| `fig_split_balance`, `fig_corpus_composition` | class shares of the three calls per fold and TEST; composition of the corpus |
| `fig_pca_*_batch`, `fig_batch_genes` | raw expression, no model. Within MG1655 (68 samples, 26 studies) and BW25113 (19 samples, 7 studies), each mid-exponential, untreated wild type in LB (one medium, so that study differences are not differences between rich media), the genes that differ most between studies: `tnaA`, `ffs`, `tnaC`, `tdcB`, `cspG` (MG1655); `ffs`, tRNA, `chiX`, `arcZ`, `tdcB` (BW25113) |
| `fig_gsea_batch_*` | GO terms of that between-study ranking (one-sided: the most study-variable processes). In both strains: flagellum and chemotaxis, ribosome and translation (and anaerobic respiration in BW25113); 71 terms at FDR < 0.05 in MG1655, 74 in BW25113; NES of the two strains correlate (r 0.86). The scatter labels the 10 terms with the largest NES summed over both strains. Off the diagonal (`gsea_batch_<strain>.tsv`): phenylacetate catabolism (`paa`), D-ribose transport, the TCA cycle and succinate dehydrogenase, thiosulfate transport vary between MG1655 studies only; putrescine catabolism, molybdopterin binding and anaerobic respiration between BW25113 studies only. |
| `fig_gsea_batch_scatter_regulondb`, `_kegg_pathway`, `_kegg_module` | the same one-sided test of the module table (`--modules`, `gsea_batch_modules_<strain>.tsv`), one scatter per database. Study-variable in both strains: the Fnr-activated, ArcA- and NarL-repressed genes (oxygen: the LB cultures of different labs differ in how anaerobic they are, by up to ~5 log2 units in the anaerobic-respiration genes), FlhDC and FliA (flagella), RpoS, flagellar assembly, ribosome, chemotaxis; MG1655 only: Crp-activated genes, the TCA cycle, phenylacetate degradation (PaaX); BW25113 only: PuuR (putrescine), CsgD (curli) |
| `fig_head_test_eval` | confusion matrix of each call on TEST |
| `fig_head_scores` | each head's score by class: AUROC on TEST phase 0.92, medium 1.00, lab strain 1.00 |
| `fig_head_gene_correlates` | genes most correlated with each head's probability |
| `fig_study_leakage` | share of a TEST sample's 25 neighbours from its own study: 0.18 in the expression, 0.07 in the call blocks, 0.09 in the embedding |
| `fig_threshold_curves` | balanced accuracy against the threshold, out-of-fold |

**Why growth-phase and stress regulons vary between "mid-exponential" LB studies.** Every sample of these subsets is
labelled mid-exponential, but LB supports steady exponential growth only to about OD 0.3 (Sezonov et al. 2007,
*J. Bacteriol.*): its few sugars run out early, the cells turn to amino acids, cAMP-Crp rises, RpoS accumulates and,
in denser cultures, oxygen falls. Labs harvest "mid-log" anywhere along that slowdown, and the study means follow it:
RpoS-activated genes rise with Crp-activated (r 0.76) and ArcA-repressed genes (0.67) and fall with the ribosomal
proteins (-0.81), and the model's phase score tracks the same axis (0.68 with RpoS, -0.81 with the ribosome) while
almost every sample is still called exponential. The ODs stated in GEO (0.25 to 0.8) line up only partly (spectrophotometers
differ, and cells diluted from a stationary overnight culture carry RpoS into early growth). So RpoS, Crp, Fnr/ArcA and
the anaerobic-respiration genes top the study-variable lists because "exponential in LB" spans a range of growth rates,
not because the labels are wrong.

The batch-gene and GO panels are Figure 2 C–G of the paper. The GO enrichment is one-sided
(fgsea `scoreType = "pos"`): NES near 1 for a term with no enrichment, near 0 for one whose genes are among the least
study-variable; `--gsea-sides two` gives the two-sided test instead.
