# Example: variation between LB studies of the laboratory strains

**Question.** Cultures of the same lab strain, grown "mid-exponential" in LB, untreated, in different labs: how much do
they differ, in which genes and gene sets, and why? No model is involved: this is the raw expression (log2 normalised
counts of the atlas tables).

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas figures ... --only pca_batch,batch_genes,gsea,batch_tracking --modules ... --out figures
```

The samples are every LB, exponential, untreated wild-type sample of a laboratory strain in the atlas (116 samples):
MG1655 (68 samples, 26 studies), BW25113 (19, 7) and the other K-12, B, C and W strains (29 samples, 6 studies: MG1693,
DH1, an unnamed K-12, B DSM 613, W3110, BW27786). The PCA and the batch-gene panels take MG1655 and BW25113 apart; the
gene-set enrichment and the tracking figure take all of them. Genes the model calls missing from a strain (B DSM 613
lacks about 320 K-12 genes, the K-12 strains at most a few) are left out of that study. The module table
(`ecoli_atlas download modules`, `../../atlas_data/modules/modules.tsv`) gives the RegulonDB and KEGG panels; without
it only the GO terms are tested.

Writes `figures/` (one PDF and PNG per figure, the tables beside them, `figure_metrics.json`).

| Figure | What came out (run 2026-11-23) |
|---|---|
| `fig_pca_*_batch`, `fig_batch_genes` | raw expression, no model. Within MG1655 (68 samples, 26 studies) and BW25113 (19 samples, 7 studies), each mid-exponential, untreated wild type in LB (one medium, so that study differences are not differences between rich media), the genes that differ most between studies: `tnaA`, `ffs`, `tnaC`, `tdcB`, `cspG` (MG1655); `ffs`, tRNA, `chiX`, `arcZ`, `tdcB` (BW25113) |
| `fig_gsea_batch_go` | GO terms of one pooled between-study ranking (`gsea_batch.tsv`; one-sided: the most study-variable processes): all 116 LB exponential, untreated wild-type samples of the laboratory strains, 40 studies × strains (MG1655 26, BW25113 7, and MG1693, DH1, an unnamed K-12, B DSM 613, W3110, BW27786), genes ranked by the SD of their per-study means; a gene is left out of the studies whose strain lacks it (B DSM 613 lacks about 320 K-12 genes, the K-12 strains at most a few). 74 of 839 terms at FDR < 0.05; on top flagellum and chemotaxis (NES 8.2), anaerobic respiration (6.0), ribosome and translation (5.3), then the type III (flagellar) secretion apparatus, aerobic electron transport, siderophore-iron import and the TCA cycle |
| `fig_gsea_batch_regulondb`, `_kegg_pathway`, `_kegg_module` | the same test of the module table (`--modules`, `gsea_batch_modules.tsv`), one bar figure per database (the 20 sets with the largest NES). RegulonDB is tested as whole regulons (activated and repressed targets together; the ranking has no direction): 64 of 143 at FDR < 0.05, on top Crp, FlhDC, RpoS, ArcA, Fnr, FliA, IHF, NarL, Fur, Fis, then NarP, FliZ, H-NS, TdcA/TdcR, CsgD (curli): growth slowdown and oxygen (the LB cultures of different labs differ in how anaerobic they are, by up to ~5 log2 units in the anaerobic-respiration genes), flagella and iron. KEGG pathways (23 of 98): flagellar assembly, chemotaxis, ribosome, TCA cycle, nitrogen cycle; KEGG modules (13 of 68): TCA and methylcitrate cycles, nitrate reduction, lysine and arginine catabolism, GABA shunt, phenylacetate degradation |
| `fig_batch_tracking` | one point per LB study (MG1655, BW25113 and, in green, the 6 studies of other lab strains in the same conditions: MG1693, DH1, an unnamed K-12, B DSM 613, W3110, BW27786; 39 studies) for each of the 9 RegulonDB regulons at the top of `fig_gsea_batch_regulondb`, in that order: the regulon's mean log2 expression against that of the ribosomal proteins (a growth-rate marker), every panel on the same log2 span, each with the R² of one line for all studies and the residual SD around it (`batch_tracking.tsv`, `batch_tracking_sets.tsv`). Growth explains 37-67% of the between-study variance of RpoS, IHF, Crp and ArcA (r -0.61 to -0.82, residual SD 0.4-0.5 log2), 19-21% of Fnr and NarL (0.55 and 1.0 log2 left), 10% of Fur, and none of the flagellar regulons FlhDC and FliA (R² 0.00, about 1.4 log2 left): two bands, on or off in a study whatever its growth rate, a difference between lab stocks (see below). The other lab strains fall among MG1655 and BW25113. `figures --tracking-x phase` puts the study's mean phase ordinal score on x instead; within exponential cultures it explains less. NarL here is mostly its repressed targets, the anaerobic reductases (`frd`, `dms`, `tor`, `ynf`), hydrogenases 1 and 2, `pflB`-`focA` and `adhE`, many of them also Fnr-activated (LB has no nitrate, so this is not NarL at work): the most oxygen-specific readout, nearly off in well-aerated cultures, so it spreads the most and depends on the culture set-up as well as on density |

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

**Why the flagellar genes vary.** They do not follow the growth rate (`fig_batch_tracking`): in each strain the studies
fall into two groups (the late flagellar genes about 7 log2 units apart, the whole FliA regulon about 3.5), on or off
from `flhDC` (the master regulator) down, and one study that
grew both strains (GSE126081) has flagella on in MG1655 and off in BW25113. That points to the lab stocks rather than the
cultures: K-12 stocks differ in insertion elements upstream of `flhDC`. In the two example runs that kept their
alignments, the MG1655 of GSE271069 lacks the IS1 that the reference MG1655 carries there and the W3110 of GSE147611 has
an IS5 at the same site; the atlas recount did not keep alignments, so the atlas studies themselves are not checked.

The batch-gene panels are those of Figure 2 of the paper; the GO panel here pools all the laboratory strains. The enrichment is one-sided
(fgsea `scoreType = "pos"`): NES near 1 for a term with no enrichment, near 0 for one whose genes are among the least
study-variable; `--gsea-sides two` gives the two-sided test instead.
