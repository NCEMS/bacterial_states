<a href="https://ncems.psu.edu/">
  <img src="https://ncems.psu.edu/wp-content/uploads/2024/02/psu-ncems_logo-wordmark-white33.png" style="width: 60%;">
</a>

# inGEST: Integrative Gene Expression Synthesis Tool

Open-source tools for integrating bacterial gene expression data across studies and placing new
*E. coli* RNA-seq samples in the context of thousands of public experiments.

Gene expression datasets from different studies differ in growth conditions, protocols and other
technical factors. inGEST processes raw RNA-seq data in a strain-agnostic, pangenome-graph framework
and compares samples with a conditional autoencoder that corrects for study effects while keeping a
latent space that reflects biology. The *E. coli* expression atlas built this way gives a global view
of the transcriptional programs that define cell states; researchers can embed their own samples
and read off where they sit relative to the atlas.

```
fastq ──► Ecoli_inGEST ──► counts ──► Ecoli_ATLAS embed ──► embedding ──► Ecoli_ATLAS compare ──► atlas neighbours,
         (Snakemake)                  (encoder–decoder)                    labels, novelty
```

## Two packages

| | Ecoli_inGEST | Ecoli_ATLAS |
|---|---|---|
| What | Snakemake RNA-seq workflow: QC gates, `vg giraffe` pangenome alignment, featureCounts, DESeq2, MultiQC | PyTorch encoder–decoder: train, embed new samples, compare with the atlas |
| Install | conda env from `ecoli_ingest/locks/` + `pip install -e ./ecoli_ingest` (or any Python + Apptainer image) | conda env from `ecoli_atlas/locks/` + `pip install -e ./ecoli_atlas` (or the Apptainer image, CPU) |
| Command | `ecoli_ingest init / check / run` in your run directory | `ecoli_atlas prepare / embed / compare / train` |
| Data (downloaded separately) | graph index, counting annotation and genome FASTA (175 MB download, 676 MB unpacked); Centrifuge DB (6.9 GB, optional); container image (~2-3 GB, optional) | trained model with its atlas embedding and all 8 seeds (312 MB download); training data with the metadata, its GEO text and the missing-gene calls (100 MB download) |
| Docs | [ecoli_ingest/README.md](ecoli_ingest/README.md) | [ecoli_atlas/README.md](ecoli_atlas/README.md) |

Both can be installed and used independently, on Linux or macOS (Apple Silicon or Intel). Data bundles are fetched with
`ecoli_ingest download …` and `ecoli_atlas download …` from the
[`resources-v2` release](https://github.com/NCEMS/bacterial_states/releases/tag/resources-v2) of this repository (the
Centrifuge index comes from its authors). The Apptainer images belong to the code version and are on the GitHub
container registry (`oras://ghcr.io/ncems/ecoli_ingest:<version>`, `oras://ghcr.io/ncems/ecoli_atlas:<version>`).
Both environments are pinned: `environment.yaml` pins every direct dependency, `locks/<platform>.txt` lists every
package with its checksum (linux-64, osx-64, osx-arm64), and the images are built from the linux-64 locks.

## Quick start

```bash
git clone https://github.com/NCEMS/bacterial_states.git && cd bacterial_states

# Ecoli_inGEST
conda create -n ecoli_ingest --file ecoli_ingest/locks/linux-64.txt && conda activate ecoli_ingest && pip install --no-deps -e ./ecoli_ingest
ecoli_ingest download core
mkdir ~/my_run && cd ~/my_run && ecoli_ingest init    # edit config.yaml
ecoli_ingest check && ecoli_ingest run --cores 12

# Ecoli_ATLAS
conda create -n ecoli_atlas --file ecoli_atlas/locks/linux-64.txt && conda activate ecoli_atlas && pip install --no-deps -e ./ecoli_atlas
ecoli_atlas download models                  # into outputs/models
ecoli_atlas embed   --ingest-results ~/my_run --models outputs/models --out embedded
ecoli_atlas compare --embeddings embedded --models outputs/models --out compared
```

## Worked examples

One folder per example under `examples/`, each with a README (the question, the commands and what came out) and a
`run.sh` that runs it against the model in `outputs/models`. [`examples/README.md`](examples/README.md) is the index.

| Example | What it shows | Needs |
|---|---|---|
| [`GSE147611_greAB_deletion`](examples/GSE147611_greAB_deletion/README.md) | the smallest one: K-12 W3110 wild type against a greA greB deletion in LB, 3 + 3 samples of about 2 million read pairs, from reads to counts, DESeq2 and the atlas | SRA reads; `run.sh ingest`, then `run.sh atlas` |
| [`GSE67402_growth_starvation_time_course`](examples/GSE67402_growth_starvation_time_course/README.md) | *E. coli* B REL606 in glucose minimal medium, 3 h to 2 weeks: reads to counts (inGEST) to the embedding, calls and axes, step by step | SRA reads; `run.sh ingest`, then `run.sh atlas` |
| [`GSE271069_nutrient_stress_time_course`](examples/GSE271069_nutrient_stress_time_course/README.md) | K-12 MG1655, glucose against casamino acids through exponential and stationary phase: a medium × time design from reads to the atlas | SRA reads; `run.sh ingest`, then `run.sh atlas` |
| [`atlas_model_figures`](examples/atlas_model_figures/README.md) | the training data, TEST performance, head scores and study leakage | atlas tables |
| [`atlas_lab_strain_LB_variation`](examples/atlas_lab_strain_LB_variation/README.md) | how LB exponential cultures of the lab strains differ between studies, and which differences follow the growth rate | atlas tables |
| [`atlas_drivers`](examples/atlas_drivers/README.md) | the genes, GO terms and gene modules that move each call | atlas tables |
| [`atlas_axes`](examples/atlas_axes/README.md) | growth phase, oxygen and carbon source as dimensions of the embedding, and their genes | atlas tables |
| [`GSE189154_oxygen`](examples/GSE189154_oxygen/README.md) | low oxygen against aerobic growth in a held-out study | atlas tables |
| [`GSE128997_uti`](examples/GSE128997_uti/README.md) | infection against urine culture, paired by patient isolate | atlas tables |
| [`GSE224463_rpoS_stress_time_courses`](examples/GSE224463_rpoS_stress_time_courses/README.md) | a study the model never saw: wild type and Δ*rpoS* through stationary-phase entry, cold and high osmolarity; its atlas neighbours, calls over the seeds and growth phase along each time course | the model bundle |

"Atlas tables" are the training-data release in `atlas_data/` and the counting annotation in `resources/annotation/`.

## Repository layout

```
bacterial_states/
├── ecoli_ingest/            # Python package; workflow under src/ecoli_ingest/workflow/
│   ├── environment.yaml     # conda env for running without the container (pinned)
│   ├── locks/               # the exact environment per platform (conda explicit files)
│   ├── build/               # Apptainer definition for the optional image
│   └── atlas_build/         # scripts used to build the atlas corpus (not user-facing)
├── ecoli_atlas/             # Python package (src layout), tests, SLURM template, environment.yaml, locks/, container/
├── bin/                     # run either CLI from the checkout without installing
├── release/                 # data release archives, conda lock files, container builds (maintainers)
├── examples/                # one folder per example: README.md, run.sh (+ config.yaml, samples.csv); outputs not committed
├── docs/                    # additional guides (CyVerse)
├── CLAUDE.md                # conventions for working in this repo
└── (local, not committed)   resources/ atlas_data/ outputs/ work/ notebooks/
```

## Citation

If you use inGEST in your research, please cite the associated publication or project
documentation when available.

## Contact

For questions, bug reports, or contributions, open an issue in this repository or contact the
project maintainers.

## License

MIT License. Copyright (c) 2026 National Synthesis Center for Emergence in the Molecular and
Cellular Sciences (NCEMS).
