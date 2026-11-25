# Examples

Each folder is one example: a `README.md` (the question, the commands and what came out) and a `run.sh` that runs it
from inside the folder against the atlas model in `../../outputs/models` (`MODELS=<bundle> bash run.sh` for another
bundle). Everything a run writes stays in the folder and is not tracked by git.

| Example | What it shows | Needs |
|---|---|---|
| [GSE147611_greAB_deletion](GSE147611_greAB_deletion/README.md) | the smallest read-based example: wild type against a greA greB deletion, 3 + 3 samples of about 2 million read pairs, from reads to counts, DESeq2 and the atlas | `run.sh ingest`, then `run.sh atlas` |
| [GSE67402_growth_starvation_time_course](GSE67402_growth_starvation_time_course/README.md) | the full path from reads to counts (inGEST) to the embedding, calls and axes, explained step by step | `run.sh ingest`, then `run.sh atlas` |
| [GSE271069_nutrient_stress_time_course](GSE271069_nutrient_stress_time_course/README.md) | a two-factor design (medium × time) from reads to the atlas, and glucose against casamino acids | `run.sh ingest`, then `run.sh atlas` |
| [atlas_model_figures](atlas_model_figures/README.md) | the training data, TEST performance, head scores and study leakage | atlas tables |
| [atlas_lab_strain_LB_variation](atlas_lab_strain_LB_variation/README.md) | how LB exponential cultures of the lab strains differ between studies: genes, GO terms, regulons, KEGG, and which follow the growth rate (paper Figure 2) | atlas tables (module table for regulons and KEGG) |
| [atlas_drivers](atlas_drivers/README.md) | the genes and GO terms that move each call | atlas tables |
| [atlas_axes](atlas_axes/README.md) | growth phase, oxygen and carbon source as dimensions of the embedding, and their genes | atlas tables |
| [GSE189154_oxygen](GSE189154_oxygen/README.md) | low oxygen against aerobic growth in a held-out study | atlas tables |
| [GSE128997_uti](GSE128997_uti/README.md) | infection against urine culture, paired by patient isolate | atlas tables |
| [GSE224463_rpoS_stress_time_courses](GSE224463_rpoS_stress_time_courses/README.md) | a study the model never saw: wild type and Δ*rpoS* through stationary-phase entry, cold and high osmolarity; its atlas neighbours, calls over the seeds and growth phase along each time course (`extract`) | the bundle |

The three read-based examples also write `atlas/atlas_view.html`: one page with an interactive 3-D map of the atlas
and the example's own samples in it (`bash run.sh view` makes the page alone; see
[GSE147611_greAB_deletion](GSE147611_greAB_deletion/README.md), section 3).

"Atlas tables" are the training tables in `../../atlas_data/` (from the training-data release) and the counting
annotation in `../../resources/annotation/`. The studies of the examples are TEST studies of the model (GSE224463 is
withheld), so the model was not fitted on them.
