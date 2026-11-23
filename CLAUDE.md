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
  `axes` (ordered categories as dimensions of the embedding), `drivers`, `compare` (new samples against the
  atlas of a bundle) and `neighbors` (atlas studies against the rest of it). Everything is on `main`
  (`update-atlas` and `ingest-update` were merged on 2026-11-22). The v1 package is only in the code release
  `v1.0.0`; see "v1 history" at the end.
- `bin/ecoli_ingest`, `bin/ecoli_atlas` run either CLI from the checkout without `pip install`.

## Data lives outside git — hard rules

Never `git add` any of these (all in `.gitignore`): `resources/`, `old_resources/`, `atlas_data/`,
`raw_data/`, `data/`, `outputs/`, `work/`, `notebooks/`, and any `*.pt *.sif *.gbz *.cf *.fastq*
*.bam *.ipynb`. Specifically:

| Local path | What | Status |
|---|---|---|
| `resources/{vg,annotation,centrifuge,container}` | inGEST reference bundle (13 GB) | downloaded; `resources/container/rnaseq_vg.sif` is the image |
| `atlas_data/` | `atlas_v2_counts.tsv`, `atlas_v2_antisense_counts.tsv`, `atlas_v2_samples.tsv` (from `collect_results.py`), `metadata_v2.tsv`, `sample_qc.tsv` | training inputs of the atlas model. `merged_count_table.txt`, `metadata_revised.csv` and the NCBI feature table there are the v1 inputs (see v1 history) |
| `work/atlas_full/results/<batch>/` | the atlas recount: 102 batches plus re-runs (`PE_R01`) | source of the three `atlas_v2_*` tables and of the `full_*` columns |
| `outputs/models/` | the current model bundle (8 seeds; GSE271069 in TEST since 2026-11-22; unversioned path: only downloaded release packages carry a version) | written by `ecoli_atlas train`. `outputs/` holds only `models/`, `sweeps/` (every sweep and variant bundle, `outputs/sweeps/models_<name>/`), `archive/` (earlier bundles, e.g. `models_2026-11-22`, and earlier paper outputs) and `v1_originals/` |
| `outputs/v1_originals/` | the v1 originals, moved there on 2026-11-22 (user decision) to free their paths: `models/`, `atlas/` (Colab-trained bundle and its atlas; published as `ecoli_atlas_models_v1.tar.gz` in the `resources-v1` release), `figures/` (Colab reference figures, mixed sessions), `predictions/` (`final_embedding.csv`, 3342 samples), `factor_gene_correlations.txt` (GSEA list) | **never overwrite or commit**; not readable by the current code |
| `outputs/v1_originals/{models_repro,models_repro_sweep,atlas_repro,figures_repro,predictions_paper1}/` | v1 refits and reproductions made on `main`, moved there with the originals (2026-11-22) | kept for comparison |
| `work/neighbors_*/` | `ecoli_atlas neighbors` runs (oxygen studies, withheld GSE224463) | scratch |
| `examples/<name>/` | one folder per example (`examples/README.md` is the index): read-based ones (`GSE67402_*`, `GSE271069_*`: `config.yaml`, `samples.csv`, `run.sh ingest|atlas`) and atlas-only ones (`atlas_model_figures`, `atlas_drivers`, `atlas_axes`, `GSE189154_oxygen`, `GSE128997_uti`, `GSE224463_withheld_study`); only `README.md`, `run.sh`, `config.yaml`, `samples.csv` are tracked, everything a run writes there is ignored | each `run.sh` runs from its folder against `../../outputs/models` (`MODELS=` for another bundle). Paper figures, case studies, axes and drivers are made by the examples, not written to `outputs/` |
| `work/experiment3/` | the tutorial run (4 samples) used for verification | a run directory; results in `experiment3_results/` |
| `notebooks/` | the two original Colab notebooks | read-only reference; never edit or commit |
| `raw_data/` | original Drive dump incl. unused files and the Contreras paper data | `Contreras Paper/Paper1_all_samples_counts.tsv` is the inference test case |

Download URLs live in exactly two files: `ecoli_ingest/src/ecoli_ingest/resources.py` and
`ecoli_atlas/src/ecoli_atlas/download.py`. The inGEST core bundle and both ATLAS bundles are assets of the GitHub
release `resources-v2` (`.../releases/download/resources-v2/<name>`; data releases are tagged `resources-vN`, code
releases `vX.Y.Z`, e.g. `v1.0.0`). They are built by `release/make_release.py --root . --out work/release_v2` (reproducible bytes;
the per-file checksums of the core bundle are in `resources.manifest.tsv`, the archive checksums are pinned in
`download.py`): core = vg index, counting annotation, genome FASTA; training-data = the `atlas_v2_*` tables,
`metadata_v2*` with its GEO text and evidence, `sample_qc.tsv`, `time_courses.tsv`, `atlas_v2_missing_genes.tsv`
(= `outputs/models/missing_genes.tsv`); models = `outputs/models/` with every seed. `resources-v1` (built in
`work/release_v1/`) holds the v1 model and the old annotation. The Centrifuge index comes from the Centrifuge
authors' bucket (`genome-idx.s3.amazonaws.com/centrifuge/`), and the container image is still a placeholder (`TBD`).
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
ecoli_atlas train $T --out outputs/models --split-only                      # splits.csv only
ecoli_atlas figures --metadata atlas_data/metadata_v2.tsv --splits outputs/models --out examples/atlas_model_figures/figures
ecoli_atlas train $T --out outputs/models --splits outputs/models/splits.csv --device cpu --threads 8
```

Temporary files go in the session scratchpad or under `work/`, never in the package directories.

## Conventions

- The Snakefile must stay runnable with `--directory <rundir> --configfile <rundir>/config.yaml`
  and `--config resources_dir=…`; resource defaults are in the Snakefile header and mirrored in
  `ecoli_ingest/paths.py` (`RESOURCE_FILES`). Keep them in sync.
- Workflow tool list has one source of truth: `ecoli_ingest/environment.yaml`
  (`build/Singularity.def` installs it). `workflow/envs/rnaseq_vg.yaml` is the legacy image env.
- inGEST counting is described in `docs/counting.md`; there is no backward compatibility with the v1 counting.
- Counting annotation: `resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.{gff,bed,
  features.tsv,rfam_merge_report.tsv}`, built by `ecoli_ingest/build/build_annotation.py` from the
  unmodified NCBI GFF and the Rfam hit table (both kept in `work/ingest_update/ncbi/`). Never
  hand-edit it; the older `_genomic.gff` / `_siRNA.gff` / `.bed` there were edited by hand and are
  used by the v1 code only. NCBI entries win over duplicate Rfam hits (same strand, or
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
  is kept as `dna_fraction_fit` only. `gc_bias` in the record (`gc_bias.py`: GC slope and curvature of the coverage
  inside genes) needs the reference FASTA `annotation/GCF_000005845.2_ASM584v2_genomic.fna`, in the core bundle since `resources-v2`
  (`check` warns when it is absent; the record then has no `gc_bias`).
- Sample metadata v2 (`docs/metadata_v2.md`): `atlas_build/metadata/` holds the label scheme, one
  curated `labels/<GSE>.json` per study and the scripts; `apply_labels.py` writes
  `atlas_data/metadata_v2.tsv` and must report `0 problems`. Change labels in the JSON files or
  `label_scheme.yaml`, never in the table. `usable` = standard RNA-seq by the text, prescreen pass,
  not a later copy of a duplicated sample. `atlas_data/metadata_revised.csv` (v1) is never overwritten.
- After the recount: `atlas_build/full_run/collect_results.py` joins the batches' count tables
  (a re-run sample is taken from the batch that counted it), `qc_table.py add-full --batches`
  fills `full_*` (never replacing a counted result by a not-counted one; add `--genome
  resources/annotation/GCF_000005845.2_ASM584v2_genomic.fna --bed <counting BED>` so that the recount's records,
  which predate `gc_bias`, get `full_gc_slope` / `full_gc_curv` from their coverage tracks), and `qc_table.py
  mark-model --metadata atlas_data/metadata_v2.tsv --min-cds-reads 200000 --max-dna-fraction 0.10 --model atlas_v2` sets
  `in_model` (usable, counted, at least 200,000 reads on CDS features, DNA estimate at most 10%). `ecoli_atlas` uses the
  samples with `in_model` set.
- Atlas model (`ecoli_atlas/README.md`): features are the inGEST ids; mRNA and other RNAs are
  inputs and outputs, rRNA and tRNA only covariates; size factors from mRNA. Phase head: `mid_exponential` against `stationary` where `phase2` is set;
  **`late_stationary` has no label and is in no phase evaluation, not even as a check.** The embedding
  has one block per call (phase 1, medium 1, lab strain 1; v1's 6 / 12 / 4 held nothing more, 2026-11-21, `train
  --block-dims`) and a residual of 24, fitted in v1's two stages: block encoders
  on their labelled samples, then the residual encoder and count decoder on all training samples with
  the blocks frozen. Lab strain = `strain_group` K-12/B/C/W (the laboratory lineages) against the other known groups.
  A K-12-against-the-rest call (`k12`) was tried on 2026-11-20 and dropped: with missing genes masked, B, C and W strains
  sit with K-12 in the blocks not trained on strain (`work/strain_axis/`). `outputs/sweeps/models_v2_k12` is that sweep; the
  current code refuses it (`ModelBundle.load` checks the bundle's calls).
  Genes a strain lacks are called from the counts (`preprocess.missing_genes`, on by default) and masked:
  out of the size factor, the gene statistics and the count likelihood, 0 as encoder input;
  `--no-mask-missing` gives the unmasked fit, and bundles made before the calls embed without them.
  The calls are also pooled over a study's samples that lack the same genes (`--no-pool` off; `embed`
  pools over the samples of its table, `--studies` splits them).
  `--antisense` (antisense count table) vetoes calls of silent genes that have antisense reads; a
  bundle trained with it needs the table again in `evaluate` / `figures` / `embed` to give the same calls.
  Genomic DNA: samples with `dna_fraction_est` above 10% are left out of the model (`mark-model --max-dna-fraction`, `high_dna`) and
  `embed` warns for them (`high_dna` in `samples.tsv`); it does not refuse. Likewise samples below the depth gate of 200,000 CDS reads
  (`low_depth`, `cds_reads`, config `min_cds_reads`): outside the trained depths. No general shallow-means-stationary
  trend in the atlas (thinned deep samples kept their call), but GSE67402 thinned to 200 k was called stationary. The two corrections (`train --dna-decoder`: additive term in the
  count mean; `--dna-inputs`: expected DNA reads taken off the encoder inputs; `data.dna_density`) are off: tried on 2026-11-21, no gain on
  the atlas and the input correction harms shallow libraries (`outputs/sweeps/models_v2_dna*`, `work/dna_term/`).
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
  units (neighbouring same-strand genes correlated within studies); the unit table is the one to read. `drivers --axis
  axis_<name>_summary.json` attributes an `axes` dimension instead of a head (its saved weights apply to the embedding
  columns standardised over the axis's fitted samples, not to the raw `embedding.tsv`); with a sweep it refits the
  axis on every seed's embedding (`refit_axis`, identical to running `axes` on that seed) and averages the seeds. Gene-set enrichment (`gsea.py`,
  gseapy, two-sided preranked, GO sets in `reference/go_gene_sets_ecocyc.tsv.gz` built by `scripts/go_gene_sets.py` from
  the EcoCyc GAF, genes matched by lower-case symbol) is the `figures` panel `gsea` (between-study ranking of
  `batch_genes`, MG1655 and BW25113) and `drivers --gsea` (genes ranked by signed attribution). The v1 paper's GSEA was
  R fgsea, one-sided (`outputs/v1_originals/GSEA/runGSEA.ipynb`); its NES and p-values are not comparable with these. The medium call is rich (complex
  broth) against minimal (salts and a carbon source); every medium in between has no label. A study gives at most `study_cap` samples per epoch (default: the size
  of the second-largest study; only GSE206047 is cut), and within one class of a call it counts for at
  most `label_cap` samples (default: the second-largest study of that class; `data.label_weights`). Splits are by whole study and, as in v1, TEST and
  the folds hold reference samples only (untreated, reference genotype: `labels.is_reference`); the
  other samples of training studies are fitted but never scored, those of TEST studies are
  projected only. Cross-validation is study-blind (unlike v1): holding out a fold removes every
  sample of its studies from the fit. The phase call is scored only where `growth_phase_basis` is
  `stated`; inferred phases are fitted on but not scored, and the statistics come from the reference training samples. `GSE224463` is
  withheld; the four case studies and GSE67402 (the worked example) are TEST. `embed` fails below `--min-feature-coverage` (0.9).
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
  from the metadata and a `splits.csv`, and with `--models` and the tables the model figures (`figures/context.py` puts the tables through the bundle as `evaluate` does;
  `figure_metrics.json` checks the embedding and the TEST metrics against the bundle; `--strict`).
  `ecoli_atlas casestudy` has `uti`, `oxygen` (GSE189154 only; GSE78756 and GSE80451 are unstranded, never used) and
  `glu-cas` (GSE271069, forced into TEST since 2026-11-22; the model trained on it before).
  **Non-stranded libraries are rejected everywhere** (user rule 2026-11-22): strand purity max(fraction_forward,
  fraction_reverse) >= 0.8, a bar that cannot be lowered (`call_strandedness.MIN_FRACTION`, `checks.MIN_STRAND_FRACTION`,
  `data.MIN_STRAND_FRACTION`, `qc_table.MIN_STRAND_FRACTION`, prescreen `--min-strand-fraction`); a `strandedness:
  forward|reverse` in the inGEST config is checked against the measured fractions (`strand_mismatch`), not trusted;
  `ecoli_atlas` `embed` / `prepare` / `train` refuse a sample below the bar or without strand fractions.
  They run as examples (`examples/atlas_model_figures`, `examples/<study>/casestudy`); never into `outputs/v1_originals/` (v1 originals and
  refits), and never into a bundle (out-of-fold scores made for `threshold_curves` stay beside the
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

Next after the DNA term is settled (user, 2026-11-21): how the QC covariates enter the model. The model is not trained
without QC (user decision: no hiding of covariates in training), and `embed` refuses samples without QC values (except, on request, a missing GC fit: `embed --fill-missing-gc
study|atlas`, user decision 2026-11-22, recorded in `gc_filled`) (done
2026-11-21: `embed --ingest-results RUN_DIR` reads counts, antisense counts and the QC records in one step; `qc_record.py`,
`data.check_qc`). Covariates are technical only (`config.COVARIATES`, 11 since the instrument flags were dropped;
`work/qc_blocks/`); the block encoders never see them.

Batch-corrected counts (planned 2026-11-21, after the block-encoder work; exploration in `work/batch_correction/`).
Corrected counts = a sample's OBSERVED log counts with the technical part swapped for that of a reference library:
log y + W (c_ref - c) (the decoder's covariate term is additive in the log rate) minus the mechanistic GC offset;
every sample keeps its own detail. Decoded counts (the decoder at z) are a separate, denoised product, not corrected
counts: they lose real within-study contrast through the embedding (stationary - exponential LFC r 0.88 against a
noise ceiling of ~0.98 on training studies, 0.71 on the TEST study). Batch removal needs a covariate term of the right
form and technical effects separable from biology (covariates are near-constant within a study, so the term is learned
from between-study differences; within one strain and condition the between-study share is still ~4x the null, and the
measured covariates remove ~0.01-0.02 of it). Order:
1. BUILT 2026-11-21 (`technical.py`, `reference/feature_gc_counting_v2.tsv`; GC covariates kept at the sample's own values in the
   swap so GC is not taken off twice). Still to do: the metrics as a report. `embed --corrected-counts` (observed + W (c_ref - c) - GC offset; reference c: paired-end, training means, paired-end
   mean fragment sizes, gc_slope = gc_curv = 0) and `--decoded-counts` (denoised), with metrics, on the current model, no
   retraining: between-study share of one reference condition (MG1655 / K-12, mid exponential, aerobic, 37 C, M9 glucose
   and LB) against a shuffled-label null; within-study `phase_lfc` and medium contrasts kept; no gene GC / length trend
   left against the covariates; twin pairs agree better than raw, once twins exist. The decoded product also reports how
   much within-study contrast it keeps against the split-half noise ceiling.
2. BUILT 2026-11-21 as `train --gc-offset` (`models.GCTerm`; GC columns out of the linear term; scale in the manifest).
   Learned scale (outputs/sweeps/models_v2_gcterm): 0.62, z took the rest (GC-slope R2 from z 0.33 -> 0.57), NLL +0.6 SD: the split
   is not identified. `--gc-scale fixed` (outputs/sweeps/models_v2_gcterm_fixed): decoded GC trend 0.13 (learned 0.42), NLL 6.224,
   z GC-slope R2 0.55 overall / 0.36 within conditions / 0.20 within replicates (linear term: 0.30 / 0.13): both variants put
   more GC into z, so the linear term stays the default and the QC module (`work/qc_module/`) takes the work over. Mechanistic GC offset in the decoder: gc_slope * x_g + gc_curv * x_g^2 from each gene's GC profile, one learned
   scale (the gene-level check gave factor 1.05, R2 0.39; `work/gc_bias/`); same idea for fragment size x gene length.
   Preview without fitting: subtracting it from the observed counts removes the per-sample GC trend entirely (corr with
   gc_slope 0.61 -> 0.02; the learned linear term only 0.61 -> 0.35) and keeps within-study LFCs (r 0.9995).
3. Technical twins (`atlas_build/twins/make_twins.py`; only the DNA variants exist): the ~200-sample set of t36/t50/t75,
   se, se_t50, d1M, d250k (about 1,400 short inGEST runs, the user submits); in training, an invariance loss that
   makes a twin pair share z, so the covariate term is identified inside a sample. Truncation does not cover fragment
   size, depletion kit or prep chemistry.
4. Replaced 2026-11-22 by the QC covariate module (user decision: no batch correction between studies; study
   clustering within a condition is not explained by the QC covariates, `work/batch_correction/`). Goal: all
   technical variation in one module, so that z carries biology only and true biological replicates agree in the
   decoded counts. Replicate groups: `metadata_v2.tsv` `replicate_group` / `replicate_basis` / `technical_of` (read
   from the GEO text, `work/replicates/`; 2,641 groups, 6,916 samples). Measures (`work/qc_module/replicates.py`):
   z -> QC R2 within replicates (baseline GC slope 0.24, strand purity 0.15, which is mostly genomic DNA), QC share of
   the decoded replicate spread (0.048; corrected counts 0.026). Steps: C1+C2 BUILT 2026-11-22 (`train --qc-module`
   = GC term at fixed scale + DNA term + GC-cleaned encoder inputs + no covariates in the encoder; each also as its own
   flag). GC offset = the fitted curve averaged over each feature's 300 bp windows (the inGEST fit window; package
   table `reference/feature_gc_counting_v2.tsv` has the window profile; `work/qc_module/gc_fragment.txt`: better than the
   gene-mean GC in every fragment-size stratum, fragment-sized windows add nothing; GC leakage into z is largest in
   long-fragment libraries, so not the approximation). With the GC term or cleaned inputs the size factors come from
   the GC-cleaned counts (`gc_size_factors`, fixed 2026-11-22): a mean-centred offset against a median-of-ratios size
   factor left a per-sample constant (SD 0.04, r 0.62 with gc_slope) that only z could fit - a likely reason both
   GC-term sweeps put more GC into z. The first qc-module sweeps were stopped for it. C3 replicate penalty (within-group cross-covariance of z and c, group sampler); C4 QC augmentation (another
   sample's GC offset, depth and DNA reads applied to the counts; z invariant); C5 a replicate pull, only if needed.
   RESULT (2026-11-22 sweeps, user decision): the encoder side alone is the DEFAULT (`clean_inputs` on,
   `encoder_covariates` off; flags kept both ways): GC in z 0.24 -> 0.03 within replicates, 0.33 -> 0.02 over all
   samples, TEST NLL 6.206. The decoder additions (`--qc-module`) are not used: they kept GC in z across samples and
   worsened NLL; corrected counts are the QC-free product. Open: strand purity (genomic DNA) still leaks into z within
   replicates (0.15-0.17) in every arm. C3-C5 on hold.
   Strand-purity diagnosis (`work/qc_module/strand_diag.txt`, `strand_fix_check.txt`): it is genomic DNA (z -> DNA
   estimate R2 0.37, z -> strand purity beyond DNA ~0), 1.4-1.7% of z's within-replicate variance; in the counts DNA
   compresses the dynamic range (slope over genes follows expression, rho 0.75, not length), so the mechanistic DNA
   term explains none of it. Taking it off the encoder inputs removes 40-70% but adds replicate noise (antisense
   subtraction +63% median; expected-DNA or smooth background +21% on genes under 50 reads).
   Removing the 6 DNA directions from z after training takes the leak to 0 and keeps kNN, but the decoded within-study
   treatment contrasts fall from r 0.80 to 0.67 (`bio_checks_k6.txt`): the directions are entangled with biology for
   the decoder. `train --dna-penalty W` (within-replicate R2 of DNA from the residual, in the loss) is the DEFAULT, W = 1
   (2026-11-22, user decision; outputs/sweeps/models_v2_dnapen_w1): z -> DNA within replicates 0.38 -> -0.03, decoded treatment
   contrasts r 0.80 kept, kNN phase / medium / lab strain 0.78 / 0.89 / 0.86, matched same-study 0.67 -> 0.60, TEST NLL
   6.220 (6.206 without). Open: GC between studies within matched conditions came back to R2 0.15 (0.03 without the
   penalty, 0.30 baseline); within replicates it stays at 0.01.
   Study adversary and twins for batch removal are dropped; read-1-only recounts of paired-end libraries could give
   real layout / fragment-size twins for C4.
5. A nonlinear covariate branch on c alone (kept out of f(z)), only if the replicate measures show it is needed.

Atlas model, second stage: refinement of the missing-gene calls (decoder mean, coverage), corrected counts,
hyper-parameter search; the re-run of the GSE271069 example from its reads (`examples/GSE271069_*/run.sh ingest`).
`atlas_build/` cleanup (`combine_counts.sh` assumes the old column layout), container rebuild from
`environment.yaml` (drop entrez-direct/pytorch), LICENSE file, tutorial-data bundle, real hosting
URLs, a Snakefile rule that emits an atlas-ready counts TSV.

## v1 history

The only place in the repository where v1 is described (user rule 2026-11-22: no v1 history, terms or paths in any
other repository file).
- Code release `v1.0.0` (factor encoders, `verify-models`, `download` of the v1 bundles); data release `resources-v1`
  (v1 model `ecoli_atlas_models_v1.tar.gz` and the old hand-edited annotation `_genomic.gff` / `_siRNA.gff` / `.bed`).
- v1 originals and refits in `outputs/v1_originals/` (never overwritten, not readable by the current code); the v1
  paper's GSEA notebook (R fgsea, one-sided) in `outputs/v1_originals/GSEA/runGSEA.ipynb`; the v1-era runs of the two read-based examples
  in `outputs/v1_originals/examples/`.
- v1 training inputs `atlas_data/merged_count_table.txt` (all libraries counted unstranded, 4,493 genes by b-number,
  made without fastp `--detect_adapter_for_pe`) and `atlas_data/metadata_revised.csv` (3,393 rows); never overwritten.
  Counts of the current inGEST do not match the v1 model.
- v1 model: three classifier encoders (blocks 6 / 12 / 4) plus a residual autoencoder (24), z-scored CPM input, MSE
  decoder; the current model keeps its two-stage fit and residual size, with one dimension per call.
- The comparison of metadata v2 with v1 labels (`atlas_data/metadata_v2_vs_v1.tsv`) and the v1-atlas columns of the
  candidate manifest were made by code that is now removed (`apply_labels.py --v1`, `build_manifest.py
  --atlas-samples/--old-qc`); `sample_qc.tsv` loses its `in_atlas_v1` column at the next `qc_table.py` write.
