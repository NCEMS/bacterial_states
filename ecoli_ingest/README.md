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

**B. Apptainer image** — for hosts without conda. Any Python ≥ 3.9 plus
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
| `core` | vg graph index + annotation | 173 MB download, 674 MB unpacked | this project ([release `v1`](https://github.com/NCEMS/bacterial_states/releases/tag/v1)) |
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
├── annotation/GCF_000005845.2_ASM584v2_genomic.gff          # RefSeq (gene products)
├── annotation/GCF_000005845.2_ASM584v2_genomic_siRNA.gff    # featureCounts annotation (+ Rfam ncRNAs)
├── annotation/GCF_000005845.2_ASM584v2_genomic.bed
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
centrifuge_organism: "Escherichia coli"
centrifuge_min_percentage: 30.0   # remove to disable the contamination gate
fastp_detect_adapter_for_pe: false # true: fastp --detect_adapter_for_pe for paired reads (short-insert libraries)
fastp_min_kept_percentage: 75.0   # remove to disable the read-filter gate
```

Paths are relative to the run directory. If any sample lacks `r2` the whole run is treated as
single-end. Reference files default to the standard names in the resources directory; individual
files can be overridden with `vg_index`, `annotation_gff`, `annotation_source_gff`,
`annotation_bed`, `centrifuge_index_path`.

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
| `all_samples_raw_counts.txt` | featureCounts table; `Geneid` is the gene symbol |
| `all_samples_counts_extended.tsv` | `Gene_Symbol, Gene_IDs_Field (locus tag + Dbxrefs), Gene_Summary (RefSeq product), Chr, Start, End, Strand, Length, counts…` |
| `all_samples_gene_symbols.tsv` | `Gene_Symbol`, coordinates, counts |
| `deseq2/` | normalized counts, PCA, one `<ref>_vs_<test>_deseq_results.tsv` per contrast |
| `<sample>/` | per-sample fastp, FastQC, Centrifuge, vg (`.gam`, sorted BAM), RSeQC, feature overlap |
| `../checkpoints/qc_failures.json` | samples excluded by the QC gates and why |

The count tables feed [Ecoli_ATLAS](../ecoli_atlas/README.md): `ecoli_atlas prepare --ingest-results <experiment>_results`.

## 5. Notes

- Nothing needs network access at run time unless samples are given by `accession`; those are
  resolved (NCBI/ENA metadata) and fetched by the workflow itself (NCBI with sra-tools, else ENA). Gene identifiers and
  product descriptions come from the two annotation GFFs (`workflow/scripts/annotate_counts.py`).
- Reads shorter than 50 bp automatically use the k=15 minimizer index.
- Centrifuge runs for every sample unless the config says `centrifuge: false` (`ecoli_ingest init
  --without-centrifuge` writes that); a missing key means on. With it off, Centrifuge is not run, its
  gate is ignored and old Centrifuge reports are left out of MultiQC. The gate itself applies only
  when `centrifuge_min_percentage` is set.
- Paired-end runs call fastp with its defaults, which is how the counts the atlas model was
  trained on were made: adapters are trimmed only where the mates overlap. That leaves adapter in
  libraries with very short inserts; for those set `fastp_detect_adapter_for_pe: true`, which adds
  `--detect_adapter_for_pe`. A missing key means off. Single-end runs are not affected. In an existing run directory a
  changed setting needs `ecoli_ingest run -- --forcerun fastp_pe`.
- Maintainers: `ecoli_ingest manifest` regenerates `resources.manifest.tsv` (sha256 of every
  resource file, used by `check --verify` and `download`). The container is rebuilt from
  `build/Singularity.def`, which installs `environment.yaml`.
- Scripts used to build the atlas corpus (pangenome index, batch SRA processing) are kept under
  `atlas_build/` and are not part of the user-facing workflow.
