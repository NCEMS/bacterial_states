# Example: figures of the training data and of the atlas model

**Question.** What is in the training data, how well does the model call growth phase, medium and lab strain on
held-out studies, and how much does a sample's own study still show in the embedding? The batch structure of the
raw expression between LB studies of the laboratory strains is its own example,
[atlas_lab_strain_LB_variation](../atlas_lab_strain_LB_variation/README.md).

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
| `fig_head_test_eval` | confusion matrix of each call on TEST |
| `fig_head_scores` | each head's score by class: AUROC on TEST phase 0.92, medium 1.00, lab strain 1.00 |
| `fig_head_gene_correlates` | genes most correlated with each head's probability |
| `fig_study_leakage` | share of a TEST sample's 25 neighbours from its own study: 0.18 in the expression, 0.07 in the call blocks, 0.09 in the embedding |
| `fig_threshold_curves` | balanced accuracy against the threshold, out-of-fold |
