# CLAUDE.md — working in `bacterial_states`

## What this repo is

Two independently installable packages in one repo (GitHub `NCEMS/bacterial_states`):

- **`ecoli_ingest/`** — Ecoli_inGEST. A Snakemake RNA-seq workflow (fastp → FastQC → Centrifuge [on unless `centrifuge: false`] →
  `vg giraffe` → featureCounts → DESeq2 → MultiQC) shipped as package data inside a small Python
  package (dependency: pyyaml). The `ecoli_ingest` CLI runs the workflow **in the current
  directory**: `init` writes `config.yaml`, `check` validates tools/resources/reads, `run` executes.
  Two execution paths: the `ecoli_ingest` conda env (`environment.yaml`, primary) or the Apptainer
  image (`--container always`); `auto` picks based on whether `snakemake` and `vg` are on PATH.
- **`ecoli_atlas/`** — Ecoli_ATLAS. Pure-Python (torch, sklearn, pandas) encoder–decoder with a
  count likelihood, trained on the recounted atlas (`docs/atlas_model_v2.md`, first stage). CLI:
  `prepare`, `train`, `embed`, `evaluate`, `figures` (data composition and the model figures), `casestudy`,
  `axes` (ordered categories as dimensions of the embedding), `drivers`.
  **On this branch it replaces the published model in place**: the v1 package (factor encoders,
  `compare`, `neighbors`, `verify-models`, `download`) is on `main`
  only; port from there (`git show main:<path>`), do not keep v1 copies here.
- `bin/ecoli_ingest`, `bin/ecoli_atlas` run either CLI from the checkout without `pip install`.

## Data lives outside git — hard rules

Never `git add` any of these (all in `.gitignore`): `resources/`, `old_resources/`, `atlas_data/`,
`raw_data/`, `data/`, `outputs/`, `work/`, `notebooks/`, and any `*.pt *.sif *.gbz *.cf *.fastq*
*.bam *.ipynb`. Specifically:

| Local path | What | Status |
|---|---|---|
| `resources/{vg,annotation,centrifuge,container}` | inGEST reference bundle (13 GB) | downloaded; `resources/container/rnaseq_vg.sif` is the image |
| `atlas_data/` | `atlas_v2_counts.tsv`, `atlas_v2_antisense_counts.tsv`, `atlas_v2_samples.tsv` (from `collect_results.py`), `metadata_v2.tsv`, `sample_qc.tsv` | training inputs of the model on this branch. `merged_count_table.txt`, `metadata_revised.csv` and the NCBI feature table there are the v1 inputs (used by `main`) |
| `work/atlas_full/results/<batch>/` | the atlas recount: 102 batches plus re-runs (`PE_R01`) | source of the three `atlas_v2_*` tables and of the `full_*` columns |
| `outputs/models_v2/`, `outputs/figures_v2/` | bundle and composition figures of the model on this branch | written by `ecoli_atlas train` / `figures` |
| `outputs/models/`, `outputs/atlas/` | the v1 Colab-trained bundle (+ `manifest.json`, `splits.csv`) and the atlas built from it; not readable by this branch's code | **the originals: never overwrite or commit.** They are published as `ecoli_atlas_models_v1.tar.gz` in the `resources-v1` release. Refits go to `outputs/models_repro*/` via `ecoli_atlas train --seeds ...` (on `main`); accept a refit when the original's TEST metrics lie within the sweep's spread (`verify-models` → `old_within_new_seed_spread`) |
| `outputs/models_repro/`, `outputs/atlas_repro/` | first GPU refit (seed 77) and its atlas | kept for comparison |
| `outputs/predictions/final_embedding.csv` | baseline atlas embeddings (3342 samples) | source of the future `outputs/atlas/` bundle |
| `outputs/figures/`, `outputs/factor_gene_correlations.txt` | Colab reference figures / GSEA gene list | **never overwrite.** Mixed sessions: the stored notebook outputs for cells 25/31/33 and `fig_study_leakage.pdf` come from a later retrained run, not from `outputs/models`. Regenerated figures go to `outputs/figures_repro/` (`ecoli_atlas figures ... --out outputs/figures_repro`) |
| `work/neighbors_*/` | `ecoli_atlas neighbors` runs (oxygen studies, withheld GSE224463) | scratch |
| `examples/<GSE>_<name>/` | worked examples; only `config.yaml`, `samples.csv`, `README.md` are tracked, everything a run writes there (`sra/`, `results/`, `atlas/`, logs) is ignored | run directories with `results_dir: "results"` |
| `work/experiment3/` | the tutorial run (4 samples) used for verification | a run directory; results in `experiment3_results/` |
| `notebooks/` | the two original Colab notebooks | read-only reference; never edit or commit |
| `raw_data/` | original Drive dump incl. unused files and the Contreras paper data | `Contreras Paper/Paper1_all_samples_counts.tsv` is the inference test case |

Download URLs live in `ecoli_ingest/src/ecoli_ingest/resources.py` (and, on `main`,
`ecoli_atlas/src/ecoli_atlas/download.py`; this branch has no atlas bundle to download yet). The
inGEST core bundle and both v1 ATLAS bundles are assets of the GitHub release `resources-v1` (`.../releases/download/resources-v1/<name>`; built in
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
conda activate ecoli_atlas         # torch, sklearn, pandas, matplotlib

pytest ecoli_atlas/tests                # in ecoli_atlas;  pytest ecoli_ingest/tests in ecoli_ingest
cd work/experiment3 && ecoli_ingest check && ecoli_ingest run -n        # tutorial run dir
T="--counts atlas_data/atlas_v2_counts.tsv --metadata atlas_data/metadata_v2.tsv --qc atlas_data/sample_qc.tsv \
   --features resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
ecoli_atlas train $T --out outputs/models_v2 --split-only                      # splits.csv only
ecoli_atlas figures --metadata atlas_data/metadata_v2.tsv --splits outputs/models_v2 --out outputs/figures_v2
ecoli_atlas train $T --out outputs/models_v2 --splits outputs/models_v2/splits.csv --device cpu --threads 8
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
- Sample metadata v2 (`docs/metadata_v2.md`): `atlas_build/metadata/` holds the label scheme, one
  curated `labels/<GSE>.json` per study and the scripts; `apply_labels.py` writes
  `atlas_data/metadata_v2.tsv` and must report `0 problems`. Change labels in the JSON files or
  `label_scheme.yaml`, never in the table. `usable` = standard RNA-seq by the text, prescreen pass,
  not a later copy of a duplicated sample. `atlas_data/metadata_revised.csv` (v1) is never overwritten.
- After the recount: `atlas_build/full_run/collect_results.py` joins the batches' count tables
  (a re-run sample is taken from the batch that counted it), `qc_table.py add-full --batches`
  fills `full_*` (never replacing a counted result by a not-counted one), and `qc_table.py
  mark-model --metadata atlas_data/metadata_v2.tsv --min-cds-reads 200000 --model atlas_v2` sets
  `in_model` (usable, counted, at least 200,000 reads on CDS features). `ecoli_atlas` uses the
  samples with `in_model` set.
- Atlas model (`ecoli_atlas/README.md`): features are the inGEST ids; mRNA and other RNAs are
  inputs and outputs, rRNA and tRNA only covariates; size factors from mRNA. Phase head: `mid_exponential` against `stationary` where `phase2` is set;
  **`late_stationary` has no label and is in no phase evaluation, not even as a check.** The embedding
  has v1's blocks (phase 6, medium 12, lab strain 4, residual 24) and v1's two stages: block encoders
  on their labelled samples, then the residual encoder and count decoder on all training samples with
  the blocks frozen. Lab strain = `strain_group` K-12/B/C/W against the other known groups.
  `ecoli_atlas axes` places samples on ordered axes of an embedding (ordinal and anchor scores, always
  scored without the sample's study; `--holdout` for extrapolation tests); its example inputs and runs
  are in `work/axes_example/`.
  `train --fit` also offers `one-pass` (same blocks, one loss) and `unstructured` (no blocks) for
  comparison on the same `splits.csv`; `two-stage` is the default. `train --phase-head ordinal` fits
  the phase head on exponential < transition < stationary (proportional odds) instead of the two
  classes; `binary` is the default and the call (`p_stationary`) is scored the same way for both.
  `ecoli_atlas drivers` (`drivers.py`) ranks the genes that move a head's score between two of its
  classes by integrated gradients; attribution sign = effect on the score, `direction` = up or down in the data.
  It averages over the `seed_<n>` bundles of a sweep and adds the attributions up over expression
  units (neighbouring same-strand genes correlated within studies); the unit table is the one to read. The medium call is rich (complex
  broth) against minimal (salts and a carbon source); every medium in between has no label. A study gives at most `study_cap` samples per epoch (default: the size
  of the second-largest study; only GSE206047 is cut), and within one class of a call it counts for at
  most `label_cap` samples (default: the second-largest study of that class; `data.label_weights`). Splits are by whole study and, as in v1, TEST and
  the folds hold reference samples only (untreated, reference genotype: `labels.is_reference`); the
  other samples of training studies are fitted but never scored, those of TEST studies are
  projected only. Cross-validation is study-blind (unlike v1): holding out a fold removes every
  sample of its studies from the fit. The phase call is scored only where `growth_phase_basis` is
  `stated`; inferred phases are fitted on but not scored, and the statistics come from the reference training samples. `GSE224463` is
  withheld and the four case studies are TEST. `embed` fails below `--min-feature-coverage` (0.9).
- Bundles (`artifacts.py`, `schema_version` 2) store tensors only (`torch.load(weights_only=True)`
  works). Seeds are ranked on out-of-fold scores (`--cv`), never on TEST.
- Class calls are at probability 0.5.
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
- Sample ids: inGEST count columns are BAM paths; `io.sample_id_from_column` strips them to the
  config sample name.
- Argparse only (no click/typer); keep both packages free of each other's imports so they can be
  split into separate repos / bioconda recipes later.
- `ecoli_atlas figures` draws the data composition (`fig_split_balance`, `fig_corpus_composition`)
  from the metadata and a `splits.csv`, and with `--models` and the tables the paper figures of v1
  for this model (`figures/context.py` puts the tables through the bundle as `evaluate` does;
  `figure_metrics.json` checks the embedding and the TEST metrics against the bundle; `--strict`).
  `ecoli_atlas casestudy` has `uti`, `oxygen` (GSE189154 only: GSE78756 and GSE80451 failed the
  prescreen) and `glu-cas` (GSE271069, now in the training pool, so no longer an outside test).
  Both write into `outputs/figures_v2/`; never into `outputs/figures` or `outputs/figures_repro`
  (v1), and never into a bundle (out-of-fold scores made for `threshold_curves` stay beside the
  figures). Run the two test suites in their own envs:
  `pytest ecoli_atlas/tests` (ecoli_atlas), `pytest ecoli_ingest/tests` (ecoli_ingest).

## Round-2 backlog

Found while running the examples (2026-11-13), not fixed: (1) input functions list passed samples
by globbing `checkpoints/passed/*.pass` instead of `checkpoints.qc_passed_samples.get()`, so when
the QC checkpoint is re-run in a directory that already has pass files, rules planned before it
(featureCounts, the checkpoint's own inputs) use the old sample list; a second `ecoli_ingest run
-- --forcerun featurecounts_combined` repairs it. (2) Centrifuge takes up to 75 min for 4 GB of
reads per mate. (3) `sra-tools` + `pigz` are in `environment.yaml` but must be installed into an
existing `ecoli_ingest` env (`mamba install -n ecoli_ingest -c bioconda sra-tools pigz`); the
container needs a rebuild to get them.

Atlas model, second stage: missing-gene inference, DNA term, corrected counts, `compare` /
`neighbors` on the single embedding, hyper-parameter search; the root `README.md` and the two
`examples/*/README.md` still describe the v1 atlas commands. GSEA,
`atlas_build/` cleanup (`combine_counts.sh` assumes the old column layout), container rebuild from
`environment.yaml` (drop entrez-direct/pytorch), LICENSE file, tutorial-data bundle, real hosting
URLs, a Snakefile rule that emits an atlas-ready counts TSV.
