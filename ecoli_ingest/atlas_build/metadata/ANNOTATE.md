> Reader instructions as used for the 2026-11 labelling pass. Paths are relative to the working
> directory the readers were given (a folder holding `digest/`, `labels/` and these files); see
> `docs/metadata_v2.md` for how to run the scripts from the repository.

# Reading a study digest into sample labels

You label E. coli RNA-seq samples from the text their submitters gave to GEO. The labels train a
model, so a wrong label is worse than a missing one. **Leave a label out when the text does not
support it.** Do not use outside knowledge of the paper; only the digest.

Inputs (read both first):
- `label_scheme.yaml`: the allowed values of every label (`labels:`) and the `inference_rules`.
- `digest/<GSE>.md`: study title, summary, overall design, the fields that are the same for every
  candidate sample, a tab-separated table of the fields that differ (first column GSM), and
  numbered long texts that the table refers to as `[text N]`.

Output: one file `labels/<GSE>.json` per study, valid JSON, with this shape:

```json
{
  "gse": "GSE000000",
  "notes": "what was ambiguous, which samples you could not label and why",
  "default": {"base_medium": {"v": "LB", "b": "stated", "q": "Strains were grown in LB media"}},
  "rules": [
    {"field": "ch:growth phase", "regex": "(?i)stationary",
     "set": {"growth_phase": {"v": "stationary", "b": "stated", "q": ""}}}
  ],
  "samples": {"GSM0000001": {"is_control": {"v": true, "b": "stated", "q": "untreated"}}}
}
```

- `default` applies to every candidate sample of the study; then `rules` are applied in order to
  the samples whose `field` (a column name of the digest table, or of the "same for every sample"
  list, e.g. `title`, `source_name`, `ch:genotype`, `growth_protocol`) matches `regex`
  (Python `re.search`); then `samples` entries. Later settings replace earlier ones. `"v": null`
  removes a label set earlier.
- Prefer rules over per-sample entries whenever a field value decides the label. Check your
  regexes against the table: every sample must end up with the labels its text supports, and a
  rule must not catch samples it should not (e.g. `WT` inside another word).
- `b` (basis): `stated` = the submitter's own word for this; `inferred` = follows from OD, time,
  medium composition etc. by a rule in `inference_rules`; `study_default` = said once for the
  study (summary/design/protocol) with nothing per sample contradicting it.
- `q`: a short exact quote (copy characters exactly, at most 200) from the digest that supports
  the value. In a rule, `q` may be `""` when the matched field value itself is the evidence.
- Values must come from `label_scheme.yaml`. `growth_temperature` is a number (°C).
  `perturbed_genes` and `strain_name` are short free text (gene names separated by `;`).
  `treatment` may hold several values joined by `;`, most specific first.

Label notes:
- `is_control`: true for the study's untreated, reference-genotype, uninduced condition; false
  for every other sample of a study that has such a condition; absent if no control exists.
- `genotype_class: reference` covers wild type and the parent strain the study compares against
  (also a wild-type carrying an empty vector; then `inducer_added` as the text says).
- `treatment: none` for samples the text presents as untreated. If the study plainly applies no
  treatment to any sample (e.g. a pure mutant-versus-wild-type comparison in plain medium), set
  `treatment: none` with basis `study_default`.
- A plasmid-selection antibiotic is not a treatment.
- `exposure` only where `treatment` is not `none`.
- Growth phase: follow `inference_rules` literally. A sample taken hours after a shift or drug
  addition gets no phase unless the text gives one.
- `oxygen: aerobic` only if stated or shaking/aeration is described (then `inferred`).
- `assay_class`: `total_rnaseq` for ordinary RNA-seq of total or rRNA-depleted RNA. Anything else
  (ribosome footprints, immunoprecipitated RNA, 5'/3' end mapping, nascent RNA, size-selected
  small RNA, in vitro RNA) gets its class; this flags samples for exclusion, so look for it.
- `mixed_organism: true` when the RNA comes from more than one organism (host tissue, co-culture,
  spike-in of another species' cells); false otherwise.
- `strain_group` / `strain_name` from the strain fields or the text. BW25113, MG1655, W3110,
  DH5alpha, DH10B, NCM3722, BL21 is `B`, etc. Leave empty for unnamed clinical isolates unless
  the text gives the pathotype; plain "clinical/environmental isolate" is `natural_isolate`.
- `global_regulator_perturbed`: true if a perturbed gene is one of rpoS, rpoD, rpoN, rpoH, rpoE,
  crp, cyaA, fis, hns, stpA, ihfA, ihfB, hupA, hupB, lrp, fur, fnr, arcA, arcB, narL, hfq, csrA,
  relA, spoT, dksA, cra, lexA, soxS, oxyR, phoB, ompR, cpxR, nac, ntrC (glnG), argR; false if
  genes are perturbed and none is in the list, and false for reference genotypes.

When done, reply with one line per study: GSE, number of samples, and any label you were unsure
about. Do not write any other files.

## Settled rules (these override anything above that reads differently)

- **Output directory**: write to the directory named in your task (default `labels/`).
- **exposure**: `acute` = sampled up to 30 min after the treatment began; `short` = over 30 min
  and up to 2 h; `adapted` = longer, or grown in the condition from inoculation / at steady state.
  Leave out when the duration is not given.
- **Controls**: a vehicle or solvent control (water, DMSO, methanol, ethanol carrier, mock
  infection, empty vector without inducer) is `is_control: true` with `treatment: none`. If a study
  has several reference conditions (e.g. wild type in each of several media or phases), each
  untreated reference-genotype sample is a control.
- **False by default**: when the study plainly has no such axis, set `inducer_added: false` and
  `transcription_arrested: false` with basis `study_default` (quote the design or protocol
  sentence that shows what the study does vary). Same for `mixed_organism: false`.
- **Conflicting phase evidence** (e.g. "log phase" and "16 h overnight" for the same sample, or
  a phase word in the title against a different one in a field): leave `growth_phase` out and say
  so in `notes`.
- **Nutrient supplements** (an amino acid, nucleoside, vitamin, alternative carbon or nitrogen
  source added to the medium) are medium properties, not treatments: `treatment: none`.
  Removal or exhaustion of a nutrient is a starvation treatment.
- **Growth at a non-standard condition from inoculation** (42 °C throughout, pH 5 medium, high
  salt medium): treatment as named with `exposure: adapted`; `growth_temperature` is the
  temperature at sampling.
- **amino_acids_supplied**: true for complex media, casamino acids / hydrolysates and defined
  amino acid mixes (basis `inferred` unless stated); false for minimal media with at most one
  amino acid added for an auxotrophy; leave out when the medium is unknown.
- **Title against field conflicts**: if a sample's title and its characteristics fields
  contradict each other (genotype, treatment, phase), do not pick one. Leave the affected labels
  out for those samples and start `notes` with `CONFLICT:` and a one-line description, so the
  case can be settled from the expression data.
- **Tagged strains** (epitope tag on a native gene, reporter fusion at a neutral site) with no
  other change: `genotype_class: reference`.
- **assay_class**: decide from what RNA went into the library, not from the title. Size-selected
  fragments under ~50 nt, ribosome-protected fragments, gradient fractions, IP eluates, 5'/3' end
  libraries and tRNA sequencing are not `total_rnaseq`.
- A study still private in GEO has no text: write `{"gse": "...", "notes": "no text", "default": {}, "rules": [], "samples": {}}`.
- **Every label needs a quote, defaults included.** A `default` entry may not have `q: ""`; for
  an "everything else" default (e.g. `is_control: false` for all samples that a later rule does
  not make a control) quote the design or summary sentence that describes the comparison.
- **Checking your files**: do not write your own checker. Run the project's validator on your
  studies, with output going to your own scratch folder, and fix your JSON until it reports
  `0 problems` (the problems are listed in `<prefix>_problems.tsv`):
  `LABEL_DIR=<your output dir> ONLY=GSE1,GSE2,... python apply_labels.py <soft dir> <manifest samples.tsv> <your scratch folder>/chk`
  Then read `<prefix>.tsv` for a few samples of each study to confirm the rules gave what you meant.
- **Controls, final wording (user decisions 2026-11-15)**: a control is the baseline the
  submitters compare against: the reference genotype growing untreated in a medium. It may carry
  an empty vector, and the empty vector may be induced (same inducer as the test strains). Plain
  growth of an unmodified strain in a medium, with nothing done to it, is a control even if the
  study has no treated samples; a study may therefore consist entirely of controls.
- **Excluded library types (user decision 2026-11-15)**: ribosome footprints, nuclease-digested
  RNA, and the "mRNA-seq" arm of a ribosome-profiling protocol are not wanted. If the RNA was
  fragmented and then size-selected on a gel in the footprint protocol (Ingolia 2009, Oh 2011,
  Li 2012, "size selected via gel purification", linker-1 / miRNA cloning linker ligation of
  fragments), set `assay_class: other` even when no fragment size is given. Ordinary library
  fragmentation (heat, cations, sonication) followed by a standard kit, and size selection of the
  finished cDNA library, are normal and stay `total_rnaseq`.
- **New treatment classes (user decisions 2026-11-15)**:
  - `nutrient_limited_growth`: steady-state growth under a limiting nutrient (chemostat, or a
    stated nutrient-limited continuous culture), `exposure: adapted`. The cells are growing; this
    is not starvation. Such a sample can still be the study's control. Starvation classes are for
    a nutrient that was removed or has run out (flow stopped, shift to medium lacking it,
    exhaustion in batch).
  - `medium_or_oxygen_shift`: cells moved to a different medium or oxygen regime and sampled
    after the move (resuspension in urine or fresh medium, anaerobic to aerobic, release from
    stationary phase into fresh medium), when no more specific class applies. A shift into a
    medium that lacks a nutrient is the matching starvation class; a shift in temperature, pH or
    osmolarity is the matching stress class.
  - Nitrosative stress (NO donors, nitrite-derived NO) is **not** `oxidative`; use `other_chemical`.
  - Compounds, physical exposures and metabolites that fit no class stay `other_chemical` /
    `other_physical`; do not stretch a specific class to hold them.
- **In-host samples**: `culture_mode: in_host` (or `with_host_cells`), `growth_phase:
  not_applicable`, `mixed_organism: true`, and **no** `treatment` for the host environment itself
  (`host_factor` is for serum, bile, host peptides or host cells added to a culture). A treatment
  given to the host or the bacteria on top of that (a drug, colitis induction) is labelled as usual.
- **`other_starvation` (user decision 2026-11-15)**: removal or exhaustion of a nutrient that has
  no class of its own (sulfur, thymine/thymidine, a vitamin, magnesium as a nutrient). Carbon,
  nitrogen, phosphate and amino acids keep their own classes; iron and other metals withheld are
  `metal_limitation`.
- **Decisions of 2026-11-15 (second set)**:
  - `coculture` (grown together with another bacterial species or strain) and `conjugation`
    (mating mixtures, filter mating, donor/recipient mixes) are treatment classes; combine them
    with any other treatment using `;`. Such samples are `mixed_organism: true`.
  - Phage-infected samples: `treatment: phage`. Phage RNA alone does not make a sample
    `mixed_organism` (the validator derives `phage_infected` and sets mixed_organism false).
  - Antisense PNA or other antisense oligos against a gene: `genotype_class: knockdown` with the
    target in `perturbed_genes`, plus the chemical treatment; scrambled/mismatch oligos stay
    reference genotype.
  - `genotype_class: wild_type` = an unmodified strain (no engineered change, no plasmid, no tag);
    `reference` = the study's baseline when that baseline is itself modified (empty vector,
    tagged strain, the mutant parent the study compares against). `is_reference_genotype` is true
    for both; `is_wild_type` only for the first.
