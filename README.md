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
| Install | conda env `ecoli_ingest/environment.yaml` + `pip install -e ./ecoli_ingest` (or any Python + Apptainer image) | conda env `ecoli_atlas/environment.yaml` + `pip install -e ./ecoli_atlas` |
| Command | `ecoli_ingest init / check / run` in your run directory | `ecoli_atlas prepare / embed / compare / train` |
| Data (downloaded separately) | graph index + annotation (~650 MB); Centrifuge DB (6.9 GB, optional); container image (5.3 GB, optional) | trained models + atlas embeddings (~20 MB); training data (~56 MB) |
| Docs | [ecoli_ingest/README.md](ecoli_ingest/README.md) | [ecoli_atlas/README.md](ecoli_atlas/README.md) |

Both can be installed and used independently. Data bundles are fetched with
`ecoli_ingest download …` and `ecoli_atlas download …`; hosting is being finalised and the commands
print interim instructions until the URLs are live.

## Quick start

```bash
git clone https://github.com/NCEMS/bacterial_states.git && cd bacterial_states

# Ecoli_inGEST
mamba env create -f ecoli_ingest/environment.yaml && conda activate ecoli_ingest && pip install -e ./ecoli_ingest
ecoli_ingest download core
mkdir ~/my_run && cd ~/my_run && ecoli_ingest init    # edit config.yaml
ecoli_ingest check && ecoli_ingest run --cores 12

# Ecoli_ATLAS
mamba env create -f ecoli_atlas/environment.yaml && conda activate ecoli_atlas && pip install -e ./ecoli_atlas
ecoli_atlas download models
ecoli_atlas prepare --ingest-results ~/my_run/tutorial_results --out prepared
ecoli_atlas embed   --counts prepared/counts.tsv --models outputs/models --out embedded
ecoli_atlas compare --embeddings embedded/final_embedding.csv --atlas outputs/atlas --out compared
```

## Repository layout

```
bacterial_states/
├── ecoli_ingest/            # Python package; workflow under src/ecoli_ingest/workflow/
│   ├── environment.yaml     # conda env for running without the container
│   ├── build/               # Apptainer definition for the optional image
│   └── atlas_build/         # scripts used to build the atlas corpus (not user-facing)
├── ecoli_atlas/             # Python package (src layout), tests, SLURM template
├── bin/                     # run either CLI from the checkout without installing
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
