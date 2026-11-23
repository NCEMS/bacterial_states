# Example: where a study the model never saw sits in the atlas (GSE224463)

**Question.** GSE224463 (RpoS-dependent transcription across stresses) is withheld from training and from the split, so
its 179 samples are placed by the model like new data. Which atlas studies are their neighbours, and do the neighbours'
labels agree with the study's own?

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas neighbors GSE224463 ... --out neighbors --umap
```

Writes `neighbors/` (`neighbors.tsv`, `summary.tsv`, `study_summary.tsv`, the UMAP). Result (run 2026-11-23): the
nearest samples come from 121 atlas studies (most from GSE134667, GSE141694, GSE135706); the neighbours vote
exponential phase (95% of the votes) and rich medium (100%), matching the study's metadata.
