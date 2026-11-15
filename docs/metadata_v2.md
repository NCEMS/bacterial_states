# Sample metadata v2: labels read from the GEO text

Per-sample categorical and true/false labels for the atlas candidates, read from the text the
submitters gave to GEO. They replace `atlas_data/metadata_revised.csv` (v1, 3,393 rows) as the
source of supervised labels for the v2 model ([atlas_model_v2.md](atlas_model_v2.md)).

State on 2026-11-15: 560 of the 619 candidate studies are labelled (10,181 samples); the other 59
have no sample that passes the prescreen and were not read. 7,829 samples in 432 studies are
**usable**: standard RNA-seq by the text, passing the prescreen, and not a later copy of a sample
deposited twice.

## Files

Code and curated inputs are in `ecoli_ingest/atlas_build/metadata/`:

| File | Content |
|---|---|
| `label_scheme.yaml` | allowed values of every label, the lumped labels derived from them, inference rules |
| `labels/<GSE>.json` | the labels of one study: `default`, then `rules` on sample fields, then `samples` |
| `suppl_fields/<GSE>.tsv` | extra per-sample fields from a metadata sheet the submitters deposited (read as `sup:*`); `# key: value` header lines are fields shared by the whole study |
| `duplicate_samples.tsv` | later copies of samples deposited in two studies (same read and base counts) |
| `study_flags.tsv` | studies to be able to leave out of modelling steps (`GSE206047`, `large_study`) |
| `skipped_no_passing_sample.txt` | the 59 studies not read |
| `fetch_geo_soft.py` | downloads the GEO SOFT text of every candidate study (rate-limited) |
| `geo_digest.py` | reduces a SOFT file to a digest: study text, fields shared by all samples, table of the rest |
| `apply_labels.py` | applies the label files, validates them, derives the lumped columns, writes the table |
| `compare_readings.py` | per-label agreement between two tables written by `apply_labels.py` |
| `ANNOTATE.md`, `WT_REREAD.md` | the instructions the readers followed, including every settled rule |

Outputs go to `atlas_data/` (not tracked): `metadata_v2.tsv` (one row per GSM),
`metadata_v2_evidence.tsv` (GSM, label, value, basis, quoted text), `metadata_v2_counts.tsv`,
`metadata_v2_counts_usable.tsv`, `metadata_v2_problems.tsv`, `metadata_v2_vs_v1.tsv`. The SOFT
files and digests are working files under `work/` (`work/geo_text/soft`, `work/metadata_v2/digest`).

```bash
M=ecoli_ingest/atlas_build/metadata                       # needs pyyaml
python $M/fetch_geo_soft.py work/atlas_manifest/studies.tsv work/geo_text/soft 1.5
python $M/geo_digest.py work/geo_text/soft work/metadata_v2/digest work/atlas_manifest/samples.tsv
DIGEST_DIR=work/metadata_v2/digest python $M/apply_labels.py work/geo_text/soft \
    work/atlas_manifest/samples.tsv atlas_data/metadata_v2 \
    --v1 atlas_data/metadata_revised.csv --qc atlas_data/sample_qc.tsv
```

`apply_labels.py` must report `0 problems`: every value is in the scheme, every label has a basis
and a quote, and every quote occurs in the study's digest. `ONLY=GSE1,GSE2` restricts a run;
`LABEL_DIR` points it at another set of label files.

## How the labels were made

1. `fetch_geo_soft.py` and `geo_digest.py` for all candidate studies.
2. A pilot of 40 studies was read twice by different models. They agreed on 98% or more of every
   label except the two whose rules were changed between the readings.
3. The remaining studies were read in chunks of 40 by subagents following `ANNOTATE.md`, each
   writing `labels/<GSE>.json` and running `apply_labels.py` on its own studies.
4. A second pass (`WT_REREAD.md`) over the 384 studies with usable baseline samples split
   `reference` into `wild_type` and `reference`.
5. Checks: samples deposited twice were read independently in both studies and agree on 97% of
   1,114 label comparisons; uncertain library types were settled with the prescreen metrics
   (share of reads adapter-trimmed, read length, share on CDS); conflicts between a sample's title
   and its fields were settled from the counts where counts exist (GSE224463, GSE117737).

Every label has a **basis**: `stated` (the submitter's word), `inferred` (from OD, time or medium
by a rule in `label_scheme.yaml`) or `study_default` (said once for the study). An empty label
means the text does not say.

## Labels

Fine labels (`label_scheme.yaml` has the full value lists): `growth_phase`, `culture_mode`,
`base_medium`, `medium_class`, `amino_acids_supplied`, `carbon_source`, `oxygen`,
`growth_temperature`, `genotype_class`, `perturbed_genes`, `global_regulator_perturbed`,
`inducer_added`, `treatment`, `exposure`, `transcription_arrested`, `is_control`, `strain_group`,
`strain_name`, `assay_class`, `rrna_removal`, `mixed_organism`.

Lumped labels on the usable samples (samples / studies):

| Label | Classes | Rule |
|---|---|---|
| `phase2` | exponential 2,636 / 227, stationary 549 / 39 | `mid_exponential`; `stationary` + `late_stationary`. Only untreated, standard RNA-seq samples. Early exponential, transition and late exponential stay empty |
| `medium_type` | complex 2,496 / 195, defined 2,874 / 159 | `defined_minimal` + `defined_rich`. Minimal medium with casamino acids or another hydrolysate stays empty |
| `amino_acids_supplied` | true 4,243 / 257, false 2,631 / 133 | second medium label, used in place of a third medium class |
| `glucose` | true 3,272 / 131, false 587 / 35 | from `carbon_source` |
| `anaerobic` | true 115 / 13, false 4,454 / 221 | microaerobic stays empty |
| `is_wild_type` | true 3,119 / 295 | unmodified strain: no engineered change, plasmid, tag or marker |
| `is_reference_genotype` | true 3,774 | wild type, or the modified baseline a study compares against (empty vector, tagged strain, mutant parent) |
| `heterologous_expression` | true 1,356 / 35 | |
| `inducer_added` | true 1,430 / 52 | |
| `treated` | true 2,122 / 193, false 5,454 / 395 | any treatment other than `none` |
| `stress_family` | antibiotic 46 studies, temperature 18, metal 18, nutrient limitation 14, oxidative 11, pH 11, phage 10 | first value of `treatment`; the other families have fewer than 10 studies |
| `phage_infected` | true 119 / 10 | phage RNA does not make a sample `mixed_organism` |
| `is_control` | true 2,277 / 364 | baseline growth in a medium; an empty vector, also induced, counts |
| `standard_rnaseq` | | `assay_class` is `total_rnaseq`; an undetermined assay class is excluded |

Further columns: `pre_decision`, `pre_reasons`, `pre_pct_CDS` (from `sample_qc.tsv`),
`low_mrna_prescreen` (under 20% of prescreen reads on CDS; 713 usable samples), `usable`,
`duplicate_of`, `study_flag`.

A fine class is meant for training only if at least 10 studies and 100 samples carry it
(`min_support`); otherwise it is used through its lumped label.

## Decisions

- Excluded library types: ribosome footprints, nuclease-digested RNA, the "mRNA-seq" arm of a
  ribosome-profiling protocol (fragmented and size-selected on a gel), pulldowns, end mapping,
  small-RNA libraries, single-cell libraries. Ordinary fragmentation and a standard kit are kept,
  whatever the study is about.
- Nutrient-limited steady state (chemostat) is `nutrient_limited_growth`, not starvation.
  `medium_or_oxygen_shift`, `coculture`, `conjugation` and `other_starvation` are treatment
  classes. Nitrosative stress is not `oxidative`. Compounds and exposures that fit no class stay
  `other_chemical` / `other_physical` and count only towards `treated`.
- Antisense oligos against a gene are a `knockdown` of that gene.
- Standard toxin-negative EHEC laboratory strains (TUV93-0, Sakai stx-) are `wild_type`.
- The later copy of a sample deposited twice is not usable.
- `GSE206047` (1,077 samples of one circuit-characterisation experiment) is kept and flagged. It
  supplies 55% of the usable stationary samples and 63% of the heterologous-expression samples.
  Its labels come from the submitters' metadata sheet in GEO and from the methods of the
  preprint that deposits the data (doi:10.1101/2023.06.29.547078).
- The 59 studies without a passing sample are not read.

## Open

- Libraries dominated by rRNA or tRNA pass the prescreen. A minimum mRNA read count is to be
  applied in the full inGEST run; `low_mrna_prescreen` marks the candidates.
- About a dozen studies have a few samples whose labels were left out because the title and a
  field contradict each other; `notes` in their label files starts with `CONFLICT`. They can be
  settled from the new counts.
- Seven samples are wild type in GSE221706 and a deletion in their later copies in GSE252784.
- Apart from genotype, the labels were read once; the check is the agreement between duplicate
  deposits.
- `atlas_data/metadata_v2_vs_v1.tsv` lists the agreement with v1. Where both give a phase or a
  medium type they agree on about 97% of samples; the v1 labels of GSE224463 follow GEO fields
  that are wrong for that study.
