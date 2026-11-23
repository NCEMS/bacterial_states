# Ecoli_inGEST

Snakemake workflow that takes raw *E. coli* RNA-seq reads to QC reports, a gene count table and a
DESeq2 analysis: fastp → FastQC → Centrifuge (contamination gate, can be switched off) → `vg giraffe` on an *E. coli*
pangenome graph → `vg surject` → RSeQC → featureCounts → DESeq2 → MultiQC.

The workflow ships inside the `ecoli_ingest` Python package and is driven by one command that runs
in **whatever directory you are in**. Reference files (graph index, annotation, optional
Centrifuge database, optional container image) live in a separate *resources* directory that you
download once.

## 1. Install

Pick one of the two paths. Both give the same `ecoli_ingest` command.

**A. Conda environment (recommended)** — all tools installed natively, no container needed:

```bash
git clone https://github.com/NCEMS/bacterial_states.git
cd bacterial_states
mamba env create -f ecoli_ingest/environment.yaml     # or: conda env create ...
conda activate ecoli_ingest
pip install -e ./ecoli_ingest
```

This works on Linux and on macOS, both Apple Silicon and Intel (macOS 10.15 or later), with the
same commands. bioconda builds vg and Centrifuge for Linux only, so `environment.yaml` also lists
the [barricklab channel](https://github.com/barricklab/conda), which has macOS builds of the same
versions (vg 1.63.1 and Centrifuge 1.0.4.2); on Linux both still come from bioconda. The
[GSE271069 example](../examples/GSE271069_nutrient_stress_time_course/README.md) was run on both
platforms. If mamba reports that `vg=1.63.1` does not exist on a Mac, clear its cached channel
index (`mamba clean --index-cache`) and try again.

**B. Apptainer image** — for Linux hosts without conda (Apptainer does not run on macOS). Any Python ≥ 3.9 plus
[Apptainer](https://apptainer.org/docs/admin/main/installation.html) or Singularity
(an unprivileged install is fine, see `docs/cyverse.md`):

```bash
pip install ./ecoli_ingest            # only dependency: pyyaml
ecoli_ingest download --with-apptainer   # core + centrifuge + the image (<resources>/container/rnaseq_vg.sif)
```

`ecoli_ingest` detects which path you are on: if `snakemake` and `vg` are on your `PATH` it runs
directly, otherwise it wraps the run in the container (`--container never|always` overrides).

Without installing you can also run `bin/ecoli_ingest` from the checkout.

## 2. Download the resources

```bash
ecoli_ingest download                       # core + centrifuge (the default set)
ecoli_ingest download --without-centrifuge  # core only, for runs with `centrifuge: false`
ecoli_ingest download --with-apptainer      # ... plus the Apptainer image (~5.3 GB), only for install path B
ecoli_ingest download core                  # or name bundles one at a time: core, centrifuge, container
```

| Bundle | Contents | Size | Source |
|---|---|---|---|
| `core` | vg graph index, counting annotation, reference genome FASTA | 175 MB download, 676 MB unpacked | this project ([release `resources-v2`](https://github.com/NCEMS/bacterial_states/releases/tag/resources-v2)) |
| `centrifuge` | `p_compressed+h+v` contamination index, not needed with `centrifuge: false` | 5.4 GB download, 6.9 GB unpacked | the Centrifuge authors' prebuilt index (`genome-idx.s3.amazonaws.com/centrifuge/`, pinned S3 object version) |
| `container` | Apptainer image `rnaseq_vg.sif` | ~5.3 GB | this project (CyVerse for now, see below) |

Downloads are checked against the packaged SHA-256 manifest. The `container` image is too large
for a GitHub release asset and has no download URL yet; for it the command prints the interim
CyVerse instructions (`gocmd get .../bacterial-states/resources.tar.gz`, the complete older
resources tarball; after extracting, rename `resources/build` to `resources/container`).

The resources directory is found in this order: `--resources DIR`, `$ECOLI_INGEST_RESOURCES`,
`$BS_DATA_ROOT/resources`, `./resources`, `~/.local/share/ecoli_ingest/resources`.
Expected layout:

```
resources/
├── vg/ecoli_graph_test.d2.{gbz,dist,min,k15.min,shortread.zipcodes}
├── annotation/GCF_000005845.2_ASM584v2_counting_v2.gff      # counting annotation: NCBI genes + Rfam ncRNAs
├── annotation/GCF_000005845.2_ASM584v2_counting_v2.bed      # the same features as BED12 (RSeQC)
├── annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv           # feature list with overlaps
├── annotation/GCF_000005845.2_ASM584v2_counting_v2.rfam_merge_report.tsv  # fate of every Rfam hit
├── annotation/GCF_000005845.2_ASM584v2_genomic.fna           # NC_000913.3; GC bias in the QC record
├── centrifuge/p_compressed+h+v.{1,2,3}.cf                    # not needed with centrifuge: false
└── container/rnaseq_vg.sif                                   # optional
```

## 3. Run

```bash
mkdir my_run && cd my_run
ecoli_ingest init                 # writes config.yaml; edit samples, conditions, QC gates
ecoli_ingest check                # tools / resources / reads all present?
ecoli_ingest run -n               # dry run
ecoli_ingest run --cores 12       # run; results land in ./<experiment>_results/
```

Running again in the same directory continues where the last run stopped. A finished step is
redone only when its output is missing or older than its input (`--rerun-triggers mtime`, set by
the launcher), so upgrading `ecoli_ingest` does not re-download or re-align an existing run. The
flip side: after changing a setting that alters a step's result (a QC gate, for instance), name the
step to redo, e.g. `ecoli_ingest run -- --forcerun qc_passed_samples`; new samples and new DESeq2
contrasts are picked up without that.

Anything after `--` is passed to Snakemake, e.g. `ecoli_ingest run -- --unlock` or
`ecoli_ingest run -- --rerun-triggers mtime params` (Snakemake's stricter triggers). `ecoli_ingest shell` opens a shell inside the
container with the run directory mounted (install path B).

### config.yaml

```yaml
experiment: "tutorial"
samples:
  wt_1: {r1: "reads/WT_1.fastq.gz", condition: "wildtype"}
  ko_1: {r1: "reads/KO_1.fastq.gz", r2: "reads/KO_1_R2.fastq.gz", condition: "knockout"}  # r2 for paired-end
deseq2:
  contrasts:
    - ["condition", "wildtype", "knockout"]   # [factor, reference, test]
centrifuge: true                  # false: skip Centrifuge entirely (not run, not in MultiQC)
species_gate: true                # with Centrifuge on: samples that are not E. coli are not counted
strandedness: "auto"              # or "forward" / "reverse" (checked, not trusted); unstranded libraries are never counted
fastp_min_kept_percentage: 75.0   # remove to disable the read-filter gate
```

Paths are relative to the run directory. If any sample lacks `r2` the whole run is treated as
single-end. Reference files default to the standard names in the resources directory; individual
files can be overridden with `vg_index`, `annotation_gff`, `annotation_bed`,
`centrifuge_index_path`.

Reads for a sample can also be several files per mate, or an SRA/GEO accession:

```yaml
samples:
  wt_2:   # several lanes: the files of each mate are concatenated in order before fastp
    r1: ["reads/WT_2_L001_R1.fastq.gz", "reads/WT_2_L002_R1.fastq.gz"]
    r2: ["reads/WT_2_L001_R2.fastq.gz", "reads/WT_2_L002_R2.fastq.gz"]
    condition: "wildtype"
  ko_3:   # fetched during the run (sra-tools, ENA as the backup); single/paired is taken from the archive
    accession: "GSM1646283"
    condition: "knockout"
  ko_4:   # several runs are merged into one sample
    accession: ["SRR1945182", "SRR1945183"]
    condition: "knockout"
```

`accession` takes SRR/ERR/DRR runs, SRX experiments, SRS/SAMN samples or GSM ids (strings or lists);
study-level ids (GSE, SRP, PRJNA) are rejected because one sample cannot be a whole study. Files go
to `sra/<run>/<run>_1.fastq.gz` (and `_2`) in the run directory (`sra_dir:` to share one download
directory between runs). `ecoli_ingest check` resolves the accessions (network needed once; the
answer is cached under `sra/manifest/`), reports how much there is to download and which method
will be used; `ecoli_ingest run --max-downloads N` caps the number of runs fetched at once
(default 4).

Each run is fetched by one job, set with `sra_method` in `config.yaml`:

| `sra_method` | Source |
|---|---|
| `auto` (default) | sra-tools first: `prefetch` downloads the SRA archive from NCBI (it checks the archive's checksum) and `fasterq-dump --split-3` converts it; the two mates must hold the same number of reads. If the tools are not installed, or the fetch fails, the run's FASTQ files are downloaded from ENA instead (size and md5 verified, HTTP 403/408/429 retried). |
| `sra-tools` | sra-tools only; a failure stops the run |
| `ena` | ENA FASTQ files only (the previous behaviour) |

Both sources give the same reads with the same names (`@SRR1945242.1 <instrument name>/1`), so
results do not depend on which one served a run. sra-tools need temporary disk space in `sra_dir`
for the uncompressed FASTQ of the runs being converted (about ten times the compressed size) and
are part of `environment.yaml`; the published container image predates them and falls back to ENA.
Runs for which ENA holds only the SRA archive can be fetched with sra-tools only.

### Tutorial data

Wild-type BW25113 versus a *cysG* knockout, four single-end libraries from SRA (BioProject
PRJNA297600). Simplest: let the workflow fetch them.

```yaml
samples:
  wt_1: {accession: "SRR2547467", condition: "wildtype"}
  wt_2: {accession: "SRR2547468", condition: "wildtype"}
  ko_1: {accession: "SRR2547469", condition: "knockout"}
  ko_2: {accession: "SRR2547470", condition: "knockout"}
```

Or download them yourself and place them under `reads/` in the run directory:

```bash
mkdir reads && cd reads
wget -O WT_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/007/SRR2547467/SRR2547467.fastq.gz
wget -O WT_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/008/SRR2547468/SRR2547468.fastq.gz
wget -O KO_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/009/SRR2547469/SRR2547469.fastq.gz
wget -O KO_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/000/SRR2547470/SRR2547470.fastq.gz
```

## 4. Outputs (`<experiment>_results/`)

Set `results_dir: "results"` in `config.yaml` for a fixed directory name instead of `<experiment>_results/`.

| File | Content |
|---|---|
| `multiqc_report.html` | all QC in one report (fastp, FastQC, Centrifuge, vg stats, RSeQC, feature overlap, PCA) |
| `all_samples_raw_counts.txt` | featureCounts table; `Geneid` is the feature id (`CDS_b0002_thrA`, `ncRNA_b4577_sgrS`, `ncRNA_Thr_leader`) |
| `all_samples_counts_extended.tsv` | `Feature_ID, Symbol, Locus_Tag, Type, Source, Rfam, Dbxref, Product, Chr, Start, End, Strand, Length, counts…` |
| `all_samples_counts.tsv` | `Feature_ID`, coordinates, counts |
| `all_samples_antisense_counts.txt` | the same counting with every sample's strand reversed: reads antisense to each feature |
| `<sample>/qc/<sample>.qc.json` | per-sample QC record: read-length and fragment-size histograms, alignment statistics, read ladder, reads per RNA family and type, antisense and DNA-background values, gene-body profile, species statistics, tool versions ([docs/atlas_qc_inputs.md](../docs/atlas_qc_inputs.md)) |
| `<sample>/coverage/<sample>.cov100.npz` | read depth per strand in 100 bp bins |
| `tool_versions.json` | versions of the tools used |
| `strandedness.tsv` | per sample: read orientation fractions, the call (forward, reverse, unstranded) and whether it was counted; a library needs at least 80% of its reads on one strand (`strandedness_min_fraction` can be raised, not lowered), and with a stated `strandedness` a library stranded the other way is not counted (`strand_mismatch`) |
| `counting_method.json` | annotation, featureCounts options and the orientation used for each sample |
| `deseq2/` | normalized counts, PCA, one `<ref>_vs_<test>_deseq_results.tsv` per contrast |
| `<sample>/` | per-sample fastp, FastQC, Centrifuge, vg (sorted BAM, `vg stats`), RSeQC, feature overlap. The trimmed reads, Centrifuge's per-read tables and the `.gam` (several GB per sample) are deleted once used; the sorted BAM too with `keep_alignments: false` |
| `../checkpoints/qc_failures.json` | samples excluded by the QC gates and why |
| `../checkpoints/strandedness_failures.json` | samples not counted after alignment: library not strand-specific, or refused by the species gate |
| `<sample>/qc/<sample>_species.json` | species statistics and the gate's verdict (`ok`, `not_ecoli`, `phage`, `mixed_organism`, `no_species`) |

How reads are counted (feature set, strandedness, overlapping features) is described in
[docs/counting.md](../docs/counting.md). Only strand-specific libraries are counted; the tutorial
data below (GSE73673) is unstranded, so it stops after `strandedness.tsv`.

**Counts from this branch do not match the published atlas model** (`ecoli_atlas_models_v1`),
which was trained on counts made with the previous method. Use the `main` branch to make counts
for [Ecoli_ATLAS](../ecoli_atlas/README.md) until the atlas has been recounted.

## 5. Notes

- Nothing needs network access at run time unless samples are given by `accession`; those are
  resolved (NCBI/ENA metadata) and fetched by the workflow itself (NCBI with sra-tools, else ENA). Feature identifiers and
  product descriptions come from the counting annotation (`workflow/scripts/annotate_counts.py`).
- Reads shorter than 50 bp automatically use the k=15 minimizer index.
- Centrifuge runs for every sample unless the config says `centrifuge: false` (`ecoli_ingest init
  --without-centrifuge` writes that); a missing key means on. With it off, Centrifuge is not run,
  there is no species gate and old Centrifuge reports are left out of MultiQC.
- **Species gate** (on with Centrifuge unless `species_gate: false`). A sample is not counted when
  it is not *E. coli*. The test uses the genus of the reads Centrifuge places uniquely, with
  *Escherichia* and *Shigella* counted together (they cannot be told apart):
  - `phage`: the largest other genus is a bacteriophage at 0.5 % or more of all reads;
  - `not_ecoli`: under 50 % of genus-resolved reads are *Escherichia* / *Shigella* and the largest
    other genus is at least 2 % of all reads;
  - `mixed_organism`: the largest other genus is 10 % or more of all reads.

  Samples that are clearly something else (phage, or under 20 % *Escherichia* / *Shigella* with
  another genus at 5 % or more of all reads) are stopped before alignment and listed in
  `checkpoints/qc_failures.json`; the aligner is very slow on reads that do not match the graph.
  The full rule is applied after alignment, leaving out reads on rRNA / tRNA genes, and its result
  is the `not_counted_reason` column of `strandedness.tsv`. Thresholds: `species_thresholds` in
  the config. The rule and its thresholds come from a screen of 10,829 GEO samples
  ([docs/prescreen.md](../docs/prescreen.md)). The former keys `centrifuge_organism` and
  `centrifuge_min_percentage` are ignored with a warning: that gate asked for the species
  *E. coli* among uniquely placed reads and failed real *E. coli* libraries.
- Paired-end runs always call fastp with `--detect_adapter_for_pe`, so adapters are trimmed from
  each mate and not only where the mates overlap (which leaves adapter in libraries with very
  short inserts). The old config key `fastp_detect_adapter_for_pe` is ignored. Single-end runs are
  not affected.
- The counting annotation is rebuilt with `build/build_annotation.py` from the unmodified NCBI
  GFF and a `cmscan` table of Rfam hits; see [docs/counting.md](../docs/counting.md).
- Maintainers: `ecoli_ingest manifest` regenerates `resources.manifest.tsv` (sha256 of every
  resource file, used by `check --verify` and `download`). The container is rebuilt from
  `build/Singularity.def`, which installs `environment.yaml`.
- Scripts used to build the atlas corpus (pangenome index, batch SRA processing) are kept under
  `atlas_build/` and are not part of the user-facing workflow.
