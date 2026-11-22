> Reader instructions as used for the 2026-11 labelling pass. Paths are relative to the working
> directory the readers were given (a folder holding `digest/`, `labels/` and these files); see
> `docs/metadata_v2.md` for how to run the scripts from the repository.

# Targeted re-read: wild type versus reference

The label files in `labels/<GSE>.json` were written before `genotype_class` distinguished two
kinds of baseline. Your job is to split them, **changing nothing else**.

- `wild_type`: an **unmodified** strain: a named standard strain used as it is (MG1655, BW25113,
  W3110, BL21(DE3), Nissle, CFT073, a clinical or environmental isolate, ...) with no engineered
  change, no plasmid, no tag, no reporter, no resistance-marker insertion, and not an evolved or
  mutant derivative. The standard strain's own built-in genotype does not count as a modification.
- `reference`: the study's baseline when that baseline **is itself modified**: carries an empty
  or control vector or any plasmid, a reporter or epitope tag, a landing pad or marker, a
  spontaneous resistance mutation used for selection, or is the mutant / evolved parent that the
  study compares its other strains against.

For each study in your batch file (one GSE per line):

1. Read `digest/<GSE>.md` in full and the existing `labels/<GSE>.json`. `ANNOTATE.md` and
   `label_scheme.yaml` describe the file format; read the "Output" part of ANNOTATE.md first.
2. Find every place the file sets `genotype_class` to `reference` (in `default`, in `rules`, in
   `samples`). For the samples each one covers, decide from the digest:
   - evidently unmodified (the text calls it wild type / WT / the parent strain / an isolate, or
     names a standard strain, **and** nothing in the digest says those samples carry a plasmid,
     tag, reporter, marker or mutation) -> change the value to `wild_type`. Keep the existing
     quote if it supports this; basis `stated` if the text says wild type, `inferred` otherwise.
   - evidently a modified baseline -> leave `reference`.
   - a wild type that carries an empty vector or control plasmid, or an inducer-only control
     strain with a plasmid -> `reference`.
   - cannot tell (strain not named and nothing said) -> leave `reference` and report it.
   If one existing entry covers both kinds of sample, add a later rule (or `samples` entries) that
   sets only `genotype_class` for the subset; rules need a `field` that is a column of the digest
   table and a `regex`, and `"q": ""` is allowed in a rule.
3. Do **not** change any other label, any other value, rule order, or regex. Do not touch
   entries whose `genotype_class` is something other than `reference`. You may append one
   sentence to `notes` starting with `WT re-read:`.
4. Validate your studies with the project validator and fix until it reports 0 problems:
   `LABEL_DIR=labels ONLY=GSE1,GSE2,... python apply_labels.py <soft dir> <manifest samples.tsv> <your scratch folder>/chk`
   then check in `<scratch>/chk.tsv` that `genotype_class` is what you meant for each study.

Make the edits with small, exact changes to the JSON (load, modify the specific entries, dump
with `indent=1, ensure_ascii=False`). Use only your own scratch folder for temporary files and
write nothing else.

Final reply, brief: per study one line `GSE: n samples reference -> wild_type, m left reference`
and a few words why for anything left as reference or that you could not tell.
