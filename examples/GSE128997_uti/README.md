# Example: urinary tract infection against urine culture (GSE128997)

**Question.** For the same patient isolate, does the embedding tell the sample taken from the patient (UTI) from the
isolate grown in urine (UR)? GSE128997 is a forced TEST study.

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas casestudy uti ... --out casestudy
```

Writes `casestudy/` (`fig_uti_axis`, the sample and gene tables, `uti_metrics.json`). Samples are centred per patient
and each patient is scored on a discriminant fitted without it. Result (run 2026-11-23): 13 patients, all 13 pairs in
the right order (sign test and Wilcoxon p = 1.2e-4, permutation p = 0.002). Towards UTI: growth and translation genes
(`rpsB`, `tufB`, `dnaE`, `accB`, `speDE`); towards urine culture: glycerol-3-phosphate uptake and use (`glpQ`, `glpT`,
`glpK`), `lac`, and stationary-phase genes (`raiA`, `cspD`, `iraP`).
