# Example: the genes that move each call

**Question.** Which genes, and which groups of co-transcribed genes (expression units), move the model's growth-phase,
medium and lab-strain calls, and which GO terms do they represent?

```bash
conda activate ecoli_atlas
bash run.sh                  # ecoli_atlas drivers --head {phase,medium,lab_strain} --gsea, all 8 seeds of the bundle
```

Writes `drivers/<head>/`: `units_*.tsv` (read this one), `drivers_*.tsv` (genes), `gsea_*.tsv`, the figures and
`drivers_<head>.json`. Attributions are integrated gradients averaged over the seeds; `seeds_in_top` and `study_in_top`
say how often a unit stays among the top rows over the seeds and when the studies are drawn again. The attribution sign
is the effect on the score, not the direction of the gene (`direction` says up or down in the data); likewise a GO term
with a positive NES is one whose genes move the score towards the second class.

| Call | Units that move the score towards the second class (higher / lower in it) | GO terms (positive NES) |
|---|---|---|
| exponential → stationary (444 samples, 33 studies) | up: `pspABCDE`, `hyaA-appA`, `paaABCD`, `relB`; down: ribosomal protein operons (`rplQ`…), `nuo`, `murE-ftsA` | ribosome, translation |
| rich → minimal (2,603 samples, 132 studies) | up: `thi`, `his`, `liv`, `trp`, `leu`, `ilv`; down: `fdo`, `tnaCAB`, `nuo`, `rbs`, `deo` | amino-acid biosynthesis; flagellum and chemotaxis towards rich |
| non-lab → lab strain (5,762 samples, 304 studies) | up: `gat`, ribosomal proteins, `opp`, `aceBAK`; down: capsule (`etk-gfcA`), `arn`, cellulose (`bcs`) | transposition, translation |

Seed agreement of the unit rankings: 0.86 (phase), 0.86 (medium), 0.93 (lab strain).
