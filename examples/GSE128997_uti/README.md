# Example: urinary tract infection against urine culture (GSE128997)

**Question.** For the same patient isolate, does the embedding tell the sample taken from the patient (UTI) from the
isolate grown in urine (UR)? GSE128997 is a forced TEST study.

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas casestudy uti ... --gsea --modules ../../atlas_data/modules/modules.tsv --out casestudy
```

Writes `casestudy/`: the sample table, `uti_axis_genes.tsv` (every gene's correlation with the axis), `uti_metrics.json`
and the pathway analysis below. Samples are centred per patient and each patient is scored on a discriminant fitted
without it. Result (run 2026-11-23): 13 patients, all 13 pairs in the right order (sign test and Wilcoxon p = 1.2e-4,
permutation p = 0.002). Towards UTI: growth and translation genes (`rpsB`, `tufB`, `dnaE`, `accB`, `speDE`); towards
urine culture: glycerol-3-phosphate uptake and use (`glpQ`, `glpT`, `glpK`), `lac`, and stationary-phase genes (`raiA`,
`cspD`, `iraP`).

**Pathway analysis.** The genes ranked by their correlation with the out-of-fold axis (`uti_axis_genes.tsv`) are tested
by preranked GSEA (two-sided, 10,000 permutations; positive NES = towards UTI) against the GO terms (`--gsea`) and the
curated module table (`--modules`: iModulons, KEGG pathways and modules with their KEGG class, RegulonDB regulons by
direction), as in the [oxygen example](../GSE189154_oxygen/README.md). Writes `uti_gsea_go.tsv`, `uti_gsea_modules.tsv`
and one figure per database: `fig_uti_gsea_go`, `fig_uti_gsea_imodulon`, `fig_uti_gsea_kegg_pathway`,
`fig_uti_gsea_kegg_module`, `fig_uti_gsea_regulondb` (the 20 strongest sets each way; in the RegulonDB figure ↑ / ↓ mark
activated / repressed, red for the set's own direction, grey for the regulator's other direction when it has one).

| Database | Tested | FDR < 0.05 towards UTI | towards urine culture | Strongest towards UTI | Strongest towards urine culture |
|---|---|---|---|---|---|
| GO terms | 839 | 218 | 81 | translation, ribosome, rRNA binding | ABC transporters, oxidoreductases, carbohydrate metabolism, branched-chain amino-acid biosynthesis |
| iModulon | 162 | 25 | 65 | Translation, ppGpp, Purine, Isc System, Thiamine-1 | RpoS, Crp-1, Fnr-1 and Fnr-2, FucR/ExuR |
| KEGG pathway | 98 | 23 | 33 | Ribosome, Homologous recombination, Aminoacyl-tRNA biosynthesis, Fatty acid biosynthesis, Protein export, Lipopolysaccharide biosynthesis | Butanoate, Galactose, Pentose and glucuronate metabolism, branched-chain amino-acid biosynthesis, ABC transporters |
| KEGG module | 68 | 18 | 18 | De novo purine biosynthesis, NADH:quinone oxidoreductase, KDO2-lipid A, F-type ATPase, biotin and heme biosynthesis | Isoleucine and valine biosynthesis, TCA cycle, methionine biosynthesis |
| RegulonDB | 165 | 18 | 78 | PurR, LexA (repressed); FadR, RpoE, Fis, SoxS (activated) | Crp, RpoS, Fnr, RpoN (activated); NarL, AgaR (repressed) |

In the patient the bacteria grow fast (translation, ribosomes, purine and fatty-acid synthesis), rebuild their envelope
(lipopolysaccharide, lipid A, protein export, RpoE) and show stress and repair responses (the genes LexA represses,
the SOS response, and homologous recombination; SoxS). In urine culture they look like cells on poor, mixed carbon
sources: catabolite activation by Crp, sugar and acid catabolism, transporters, the TCA cycle, and the stationary-phase
and anaerobic regulators RpoS and Fnr. As in the oxygen example, many sets pass FDR 0.05 because the axis separates the
two conditions in every patient and most genes correlate with it; read the strongest sets, not the counts.
