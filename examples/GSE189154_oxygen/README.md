# Example: low oxygen on the atlas embedding (GSE189154)

**Question.** Does one direction of the embedding separate low-oxygen from aerobic samples of a study the model was not
trained on? GSE189154 (MG1655, aerobic against microaerobic) is a forced TEST study.

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas casestudy oxygen ... --out casestudy
```

Writes `casestudy/` (`fig_oxygen_lda_projection`, `fig_oxygen_axis_genes`, the sample and gene tables,
`oxygen_metrics.json`). Result (run 2026-11-23): 14 samples, 5 aerobic and 9 microaerobic; AUROC 1.00 in sample and
1.00 with every sample scored by an axis fitted without it. Genes along the axis towards low oxygen include `torS`,
`torC`, `hybA` and `glrK`; towards aerobic `aroG`, `puuP`, `thiH`.

The two other oxygen studies of the GEO search (GSE78756, GSE80451) are not strand-specific and are not in the atlas.
