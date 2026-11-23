# Example: low oxygen on the atlas embedding (GSE189154)

**Question.** Does one direction of the embedding separate low-oxygen from aerobic samples of a study the model was not
trained on? GSE189154 (MG1655, aerobic against microaerobic) is a forced TEST study.

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas casestudy oxygen ... --gsea --modules ../../atlas_data/modules/modules.tsv --out casestudy
```

Writes `casestudy/` (`fig_oxygen_lda_projection`, the sample and gene tables,
`oxygen_metrics.json`, and the pathway analysis below). Result (run 2026-11-23): 14 samples, 5 aerobic and 9 microaerobic; AUROC 1.00 in sample and
1.00 with every sample scored by an axis fitted without it. Genes along the axis towards low oxygen include `torS`,
`torC`, `hybA` and `glrK`; towards aerobic `aroG`, `puuP`, `thiH`.

**Pathway analysis.** Every gene is ranked by its correlation with the axis (mean within-study Spearman rho,
`oxygen_axis_genes.tsv`), and the ranking is tested by preranked GSEA (two-sided, 10,000 permutations; positive NES =
towards low oxygen) against the package's GO terms (`--gsea`) and against the curated module table of
`ecoli_atlas download modules` (`--modules`; without the table `run.sh` runs GO only). The module table holds four
databases: PRECISE-1K iModulons, KEGG pathway maps and KEGG modules (each with its KEGG class, e.g. `[Carbohydrate
metabolism]`), and RegulonDB regulons split by direction (the genes a regulator activates, `X (activated)`, and those
it represses, `X (repressed)`; links of unknown or dual effect left out). Writes `oxygen_gsea_go.tsv` and
`oxygen_gsea_modules.tsv` (`database` and `category` columns) and one figure per database, `fig_oxygen_gsea_go`,
`fig_oxygen_gsea_imodulon`, `fig_oxygen_gsea_kegg_pathway`, `fig_oxygen_gsea_kegg_module`, `fig_oxygen_gsea_regulondb`
(the strongest sets towards low oxygen above, towards aerobic below; FDR >= 0.05 pale).

| Database | Tested | FDR < 0.05 towards low oxygen | towards aerobic | Strongest towards low oxygen | Strongest towards aerobic |
|---|---|---|---|---|---|
| GO terms | 839 | 36 | 201 | anaerobic respiration, anaerobic electron transport chain, glycolytic process, histidine biosynthesis, cell adhesion / pilus | translation, ribosome, rRNA binding |
| iModulon | 162 | 38 | 35 | RpoS, UC-4, NtrC-3, GadXW, Cold Shock, Microaerobic | Translation, Fur-1 and Fur-2, ppGpp, ArcA, Isc System, Pyrimidine |
| KEGG pathway | 98 | 9 | 34 | Histidine metabolism, Glycolysis / Gluconeogenesis, Fructose and mannose metabolism | Ribosome, branched-chain amino-acid biosynthesis, Aminoacyl-tRNA biosynthesis, Lipopolysaccharide biosynthesis, Protein export |
| KEGG module | 68 | 9 | 29 | Glycolysis (glucose => pyruvate), histidine biosynthesis, cytochrome bd ubiquinol oxidase | NADH:quinone oxidoreductase (*nuo*), thiamine and de novo pyrimidine biosynthesis, branched-chain amino-acid biosynthesis, TCA cycle |
| RegulonDB | 165 | 37 | 34 | YdeO, RpoS, ArcA, NtrC, AppY (activated); NarL (repressed) | Fur, ArcA, PurR, Fnr (repressed); IscR (activated) |

The sets agree with what low oxygen does to *E. coli*: the genes ArcA activates and the Microaerobic iModulon rise,
with the high-affinity cytochrome bd oxidase, AppY and glycolysis, while the genes ArcA and Fnr repress, NADH
dehydrogenase I, the TCA cycle and translation fall. The ArcA iModulon falls too: most of its 81 genes are the
TCA-cycle and aerobic-respiration genes ArcA represses (`sdhC`, `cyoA`, `acnAB`, `icd`, `gltA`, `fumAC`, `mqo`). Many
sets pass FDR 0.05 because the axis separates the two conditions completely and most genes correlate with it; the
ranking rests on 14 samples of one study (rho takes few distinct values, so gseapy warns about tied genes). Read the
strongest sets, not the counts. Two iModulons near the top have no regulator: `UC-4` (uncharacterized; PRECISE-1K
notes it may be related to RcsC, and it holds `osmB`, `osmY`, `rcsA`) and `SSW SNPs` (a "genetic alterations"
component that tracks mutations of evolved strains in the PRECISE-1K compendium, mostly *y*-genes and insertion
elements); neither says much about oxygen.

The two other oxygen studies of the GEO search (GSE78756, GSE80451) are not strand-specific and are not in the atlas.
