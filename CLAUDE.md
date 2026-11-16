# CLAUDE.md — working in `bacterial_states`

## What this repo is

Two independently installable packages in one repo (GitHub `NCEMS/bacterial_states`):

- **`ecoli_ingest/`** — Ecoli_inGEST. A Snakemake RNA-seq workflow (fastp → FastQC → Centrifuge [on unless `centrifuge: false`] →
  `vg giraffe` → featureCounts → DESeq2 → MultiQC) shipped as package data inside a small Python
  package (dependency: pyyaml). The `ecoli_ingest` CLI runs the workflow **in the current
  directory**: `init` writes `config.yaml`, `check` validates tools/resources/reads, `run` executes.
  Two execution paths: the `ecoli_ingest` conda env (`environment.yaml`, primary) or the Apptainer
  image (`--container always`); `auto` picks based on whether `snakemake` and `vg` are on PATH.
- **`ecoli_atlas/`** — Ecoli_ATLAS. Pure-Python (torch, sklearn, pandas) encoder–decoder that
  embeds samples and compares them with the atlas. CLI: `prepare`, `train`, `embed`, `compare`,
  `verify-models`, `download`, `figures` (paper figures from a bundle, no retraining except the
  out-of-fold threshold refits), `casestudy` (`uti`, `oxygen`, `glu-cas`, `all`), `neighbors`
  (where whole GSE studies sit in the atlas).
- `bin/ecoli_ingest`, `bin/ecoli_atlas` run either CLI from the checkout without `pip install`.

## Data lives outside git — hard rules

Never `git add` any of these (all in `.gitignore`): `resources/`, `old_resources/`, `atlas_data/`,
`raw_data/`, `data/`, `outputs/`, `work/`, `notebooks/`, and any `*.pt *.sif *.gbz *.cf *.fastq*
*.bam *.ipynb`. Specifically:

| Local path | What | Status |
|---|---|---|
| `resources/{vg,annotation,centrifuge,container}` | inGEST reference bundle (13 GB) | downloaded; `resources/container/rnaseq_vg.sif` is the image |
| `atlas_data/` | `merged_count_table.txt`, `metadata_revised.csv`, feature table | atlas training inputs |
| `outputs/models/`, `outputs/atlas/` | the Colab-trained bundle (+ `manifest.json`, `splits.csv`) and the atlas built from it | **the originals: never overwrite or commit.** They are published as `ecoli_atlas_models_v1.tar.gz` in the `resources-v1` release. Refits go to `outputs/models_repro*/` via `ecoli_atlas train --seeds ...` (see `ecoli_atlas/scripts/slurm_train_repro.sh`); accept a refit when the original's TEST metrics lie within the sweep's spread (`verify-models` → `old_within_new_seed_spread`) |
| `outputs/models_repro/`, `outputs/atlas_repro/` | first GPU refit (seed 77) and its atlas | kept for comparison |
| `outputs/predictions/final_embedding.csv` | baseline atlas embeddings (3342 samples) | source of the future `outputs/atlas/` bundle |
| `outputs/figures/`, `outputs/factor_gene_correlations.txt` | Colab reference figures / GSEA gene list | **never overwrite.** Mixed sessions: the stored notebook outputs for cells 25/31/33 and `fig_study_leakage.pdf` come from a later retrained run, not from `outputs/models`. Regenerated figures go to `outputs/figures_repro/` (`ecoli_atlas figures ... --out outputs/figures_repro`) |
| `work/neighbors_*/` | `ecoli_atlas neighbors` runs (oxygen studies, withheld GSE224463) | scratch |
| `examples/<GSE>_<name>/` | worked examples; only `config.yaml`, `samples.csv`, `README.md` are tracked, everything a run writes there (`sra/`, `results/`, `atlas/`, logs) is ignored | run directories with `results_dir: "results"` |
| `work/experiment3/` | the tutorial run (4 samples) used for verification | a run directory; results in `experiment3_results/` |
| `notebooks/` | the two original Colab notebooks | read-only reference; never edit or commit |
| `raw_data/` | original Drive dump incl. unused files and the Contreras paper data | `Contreras Paper/Paper1_all_samples_counts.tsv` is the inference test case |

Download URLs live in exactly two files: `ecoli_ingest/src/ecoli_ingest/resources.py` and
`ecoli_atlas/src/ecoli_atlas/download.py`. The inGEST core bundle and both ATLAS bundles are
assets of the GitHub release `resources-v1` (`.../releases/download/resources-v1/<name>`; built in
`work/release_v1/`; data releases are tagged `resources-vN`, code releases `X.X.X`),
the Centrifuge index comes from the Centrifuge authors' bucket
(`genome-idx.s3.amazonaws.com/centrifuge/`), and the container image is still a placeholder (`TBD`).
Bundle file names are `ecoli_ingest_*` / `ecoli_atlas_*` (other microbes may follow).

## Git

- **Never commit or push without asking the user first.** Prepare, show the file list, wait.
- Use `git mv` for moves so history follows.

## Environments and running things

```bash
# host (login node has no GPU; training goes through SLURM)
conda activate ecoli_ingest        # snakemake + all workflow tools
conda activate ecoli_atlas         # torch (CPU), sklearn, pandas, optuna, umap
export ECOLI_INGEST_APPTAINER=/mnt/gs21/scratch/jbarrick/ncems/build/apptainer_local/bin/apptainer  # unprivileged apptainer

pytest ecoli_atlas/tests                # in ecoli_atlas;  pytest ecoli_ingest/tests in ecoli_ingest
cd work/experiment3 && ecoli_ingest check && ecoli_ingest run -n        # tutorial run dir
ecoli_atlas figures --counts atlas_data/merged_count_table.txt --metadata atlas_data/metadata_revised.csv \
    --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt --models outputs/models --atlas outputs/atlas \
    --reference-sweep outputs/models_repro_sweep/manifest.json --out outputs/figures_repro --device cpu --threads 8 --strict
ecoli_atlas embed --counts "raw_data/Contreras Paper/Paper1_all_samples_counts.tsv" --models outputs/models \
    --feature-table atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt --out /tmp/x   # parity check vs outputs/predictions_paper1
```

Temporary files go in the session scratchpad or under `work/`, never in the package directories.

## Conventions

- The Snakefile must stay runnable with `--directory <rundir> --configfile <rundir>/config.yaml`
  and `--config resources_dir=…`; resource defaults are in the Snakefile header and mirrored in
  `ecoli_ingest/paths.py` (`RESOURCE_FILES`). Keep them in sync.
- Workflow tool list has one source of truth: `ecoli_ingest/environment.yaml`
  (`build/Singularity.def` installs it). `workflow/envs/rnaseq_vg.yaml` is the legacy image env.
- **Branch `ingest-update` is a main revision of inGEST counting** (see `docs/counting.md`); it
  stays off `main` until adopted, and there is no backward compatibility with the old counting.
  Its counts do not match the published atlas model: make counts for `outputs/models` on `main`.
- Counting annotation: `resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.{gff,bed,
  features.tsv,rfam_merge_report.tsv}`, built by `ecoli_ingest/build/build_annotation.py` from the
  unmodified NCBI GFF and the Rfam hit table (both kept in `work/ingest_update/ncbi/`). Never
  hand-edit it; the older `_genomic.gff` / `_siRNA.gff` / `.bed` there were edited by hand and are
  used by `main` only. NCBI entries win over duplicate Rfam hits (same strand, or
  opposite strand to an NCBI ncRNA gene); `RFAM_EXCLUDE` lists the families
  left out (REP repeats, DnaX frameshift element, SECIS).
- Feature ids are `<Type>_<locus_tag>_<symbol>` (`CDS_b0002_thrA`, `ncRNA_b4577_sgrS`, also `tRNA_`,
  `rRNA_`, `pseudo_`) and `ncRNA_<Rfam id>[_n]` for Rfam hits; one row per gene, no `gene` rows.
  `annotate_counts` is offline and reads only the counting annotation; symbol, locus tag, type,
  source, Rfam accession, Dbxref and product are columns of the extended table.
- Counting: `featureCounts -t gene -g ID -O -s <per BAM> [-p --countReadPairs]`. Only
  strand-specific libraries are counted (`call_strandedness.py`, checkpoint
  `strand_passed_samples`, `strandedness.tsv`); there is no unstranded mode. Multi-mapping reads
  are left where the aligner put them (one alignment per read, no NH tag).
- Atlas candidates and QC (`docs/prescreen.md`): `atlas_build/preprocessing/build_manifest.py`
  writes `work/atlas_manifest/{runs,samples,studies}.tsv` from ENA; `atlas_build/prescreen/
  prescreen.py` screens a 200,000-read subsample per sample into `work/prescreen/samples/<GSM>.json`;
  `qc_table.py` keeps `atlas_data/sample_qc.tsv` (`pre_*` prescreen, `full_*` full run, `in_model`).
  Both tiers are parsed by `qc_metrics.py`. Never run Centrifuge with `--mm` on an index that
  sits on the network file system (minutes per sample); the prescreen copies it to `/dev/shm`.
  Species check: `species_metrics` (share of genus-resolved Centrifuge reads in Escherichia or
  Shigella, largest other genus), not the workflow's `pct_ecoli` gate, which fails real E. coli
  libraries such as GSE67402. Keep downloads rate-limited (`--fetch-interval`, default 2 s).
  When stopping a run, kill by exact process name (`pkill -x python`, `pkill -x vg`); `pkill -f`
  with the script path also kills the shell that issues it.
- QC record (`docs/atlas_qc_inputs.md`): rules `bam_qc`, `species_qc`, `tool_versions`, `sample_qc`
  write `<s>/qc/<s>.qc.json` and `<s>/coverage/<s>.cov100.npz`; `featurecounts_combined` also writes
  `all_samples_antisense_counts.txt`. `species_metrics.py` (workflow script) is shared with the
  prescreen via `qc_metrics.py` and is the workflow's species gate (`species_gate`, default on with
  Centrifuge; `species_verdict`, `SPECIES_THRESHOLDS`): rule `species_early` stops clearly foreign
  samples in the QC checkpoint before alignment, rule `species_qc` gives the verdict used by
  checkpoint `strand_passed_samples` (`not_counted_reason` in `strandedness.tsv`). The old keys
  `centrifuge_organism` / `centrifuge_min_percentage` are ignored with a warning. `species_qc` needs
  Centrifuge's `initial.tsv` (a temp file), so it cannot be added to a finished run without
  re-running Centrifuge. `keep_alignments: false` makes `_sort.bam` temporary (deleted
  after the QC steps); default is to keep it. The `.gam`, the gzipped trimmed reads and Centrifuge's
  other intermediates (`centrifuge/tmp/`) are always deleted once used. `qc_table.py add-full` turns each record into
  `full_*` columns (`qc_metrics.RECORD_COLUMNS`). `dna_fraction_est` in the record is the
  coverage-based estimate (median over bins of the lower strand's depth), calibrated with
  DNA-spiked twins (`atlas_build/twins/make_twins.py`, `work/dna_calibration/`); the antisense fit
  is kept as `dna_fraction_fit` only. The atlas model design is on branch `update-atlas`
  (`work/worktrees/update-atlas`); model code does not go on `ingest-update`.
- The atlas training table uses **b-numbers** and 4,493 features counted unstranded. `ecoli_atlas`
  maps b-numbers → symbols with the feature table and reports coverage (`gene_coverage.tsv`);
  `embed` fails below `--min-gene-coverage` (0.9).
- Model bundle file layout must stay loadable for the original Colab files (`artifacts.py`).
  New bundles store tensors only (`torch.load(weights_only=True)` works).
- Class calls default to argmax (that is what produced `outputs/predictions*`); the saved operating
  thresholds are opt-in (`--apply-thresholds`).
- The results directory is `<experiment>_results/` unless the config sets `results_dir`.
- Reads given by accession: one `sra_download` job per run calls `sra_reads.py fetch`. `sra_method:
  auto` (default) uses sra-tools (prefetch + fasterq-dump, read names rewritten to the ENA style)
  and falls back to the ENA FASTQ files; both sources give byte-identical FASTQ.
- The launcher runs Snakemake with `--rerun-triggers mtime` (unless the user passes their own), so
  rule code / parameter changes never redo finished steps of an existing run directory; changed
  settings need `-- --forcerun RULE`.
- Paired-end fastp always gets `--detect_adapter_for_pe`; the config key
  `fastp_detect_adapter_for_pe` is ignored with a warning. (The atlas training counts in
  `merged_count_table.txt` were made without it. GSE271069's `results/` on disk is a run without
  it; its trimming-on run is in `run_adapter_on/`.)
- `compare --metadata/--group-by/--order-by/--color-by` adds `group_summary.tsv` and the
  `compare_plots.py` figures; without those options `summary.tsv` / `neighbors.tsv` must not change.
- Sample ids: inGEST count columns are BAM paths; `io.sample_id_from_column` strips them to the
  config sample name.
- Argparse only (no click/typer); keep both packages free of each other's imports so they can be
  split into separate repos / bioconda recipes later.
- Figures and case studies run from `outputs/models` + `atlas_data/` only (`figures/context.py`
  rebuilds the notebook state from the bundle and `splits.csv`). `ecoli_atlas figures --strict
  --atlas outputs/atlas --reference-sweep outputs/models_repro_sweep/manifest.json` must pass; TEST
  metrics agree with the manifest to all digits, the out-of-fold refits only within the seed sweep's
  CV spread. Constant genes get Spearman rho 0 (notebook convention). Run the two test suites in
  their own envs: `pytest ecoli_atlas/tests` (ecoli_atlas), `pytest ecoli_ingest/tests` (ecoli_ingest).

## Round-2 backlog

Found while running the examples (2026-11-13), not fixed: (1) input functions list passed samples
by globbing `checkpoints/passed/*.pass` instead of `checkpoints.qc_passed_samples.get()`, so when
the QC checkpoint is re-run in a directory that already has pass files, rules planned before it
(featureCounts, the checkpoint's own inputs) use the old sample list; a second `ecoli_ingest run
-- --forcerun featurecounts_combined` repairs it. (2) Centrifuge takes up to 75 min for 4 GB of
reads per mate. (3) `sra-tools` + `pigz` are in `environment.yaml` but must be installed into an
existing `ecoli_ingest` env (`mamba install -n ecoli_ingest -c bioconda sra-tools pigz`); the
container needs a rebuild to get them.

reComBat bar of `fig_study_leakage` (opt-in `--recombat`; package not installable here), GSEA,
`atlas_build/` cleanup (`combine_counts.sh` assumes the old column layout), container rebuild from
`environment.yaml` (drop entrez-direct/pytorch), LICENSE file, tutorial-data bundle, real hosting
URLs, a Snakefile rule that emits an atlas-ready counts TSV.
