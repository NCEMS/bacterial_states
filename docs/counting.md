# How inGEST counts reads

This describes the counting of inGEST (`resources-v2`), on which the atlas model is trained.

## Features

The counting annotation is built by `ecoli_ingest/build/build_annotation.py` from two inputs:
the NCBI RefSeq GFF of GCF_000005845.2, exactly as downloaded, and a table of Rfam hits
(`cmscan --cut_ga --rfam --nohmmonly --fmt 2 --clanin`, Infernal 1.1.5, clan-overlapping hits
removed). The GFF header records the checksums of both inputs and the rules below.

- Every NCBI gene and pseudogene is one feature. Nothing is removed because it lies inside or
  overlaps another feature, so leader peptides (`thrL`, `leuL`), small ORFs inside sRNAs (`sgrT`
  in `sgrS`) and toxin genes (`hokA`) are all counted.
- An Rfam hit is dropped only when an NCBI RNA gene (ncRNA, tRNA, rRNA) lies on the same strand
  and the overlap covers at least half of the shorter of the two. The NCBI entry is kept and
  records the Rfam accession.
- An Rfam hit is also dropped when an NCBI ncRNA gene lies on the opposite strand with the same
  amount of overlap. These are the annotated locus matched on the wrong strand (four `sok`
  antitoxin hits, `PssrA` opposite `stfZ`, one `ES036` opposite `fliX`). All other hits are added.
- Four Rfam families are never added because they are not RNA genes: the REP repeat `naRNA4`
  (RF02564), the `DnaX` ribosomal frameshifting element (RF00382) and the selenocysteine insertion
  sequences `SECIS_2` and `SECIS_3` (RF01988, RF01989).
- `*.rfam_merge_report.tsv` gives the fate of every Rfam hit (kept, duplicate of which NCBI gene,
  or excluded) and `*.features.tsv` lists each feature with the features it overlaps on the same
  and on the opposite strand.

### Feature ids

One id per feature, used as the first column of every count and DESeq2 table.

| Kind | Id | Example |
|---|---|---|
| protein-coding gene | `CDS_<locus_tag>_<symbol>` | `CDS_b0002_thrA` |
| NCBI non-coding RNA (incl. tmRNA) | `ncRNA_<locus_tag>_<symbol>` | `ncRNA_b4577_sgrS` |
| tRNA, rRNA | `tRNA_…`, `rRNA_…` | `tRNA_b0203_alaV`, `rRNA_b0201_rrsH` |
| pseudogene | `pseudo_<locus_tag>_<symbol>` | `pseudo_b0240_crl` |
| Rfam hit | `ncRNA_<Rfam id>` | `ncRNA_Thr_leader` |
| Rfam hit, several copies | `ncRNA_<Rfam id>_<n>`, numbered by position | `ncRNA_RtT_1` |

A gene without a symbol is `<Type>_<locus_tag>`. The extended count table carries symbol, locus
tag, type, source (NCBI or Rfam), Rfam accession, Dbxref and product as separate columns, so the
locus tag is the stable key across NCBI annotation releases.

## Strandedness

Only strand-specific libraries are counted.

- RSeQC `infer_experiment.py` measures, for every sample, the fraction of reads on the annotated
  strand and on the opposite strand. A sample is `forward` or `reverse` when that fraction is at
  least `strandedness_min_fraction` (0.8), otherwise `unstranded`.
- Unstranded samples are left out of the count table, DESeq2 and the PCA. They are listed in
  `<results>/strandedness.tsv` (`counted = no`) and in `checkpoints/strandedness_failures.json`.
  If no sample of a run is strand-specific, the run ends after the strandedness table.
- Each sample is counted with its own orientation (`featureCounts -s 1` or `-s 2`), so forward
  and reverse libraries can share a run.
- `strandedness: forward` or `reverse` in `config.yaml` states the orientation for the whole run
  instead of measuring it. There is no unstranded setting.

## Counting rule

```
featureCounts -a <annotation> -t gene -g ID -O -s <per sample> [-p --countReadPairs] <BAMs>
```

- A read (a fragment for paired-end runs) is counted for **every** feature it overlaps on its own
  strand (`-O`). A feature's count therefore depends only on its own reads, not on which
  neighbours are annotated; the sum of a column is slightly larger than the number of assigned
  reads.
- Reads on the opposite strand of a feature are not counted for it. Antisense transcription and
  convergent neighbours no longer add counts to a gene.
- Paired-end runs count fragments, not mates.
- Each read has a single alignment. A read that fits several places equally well (rRNA operons,
  IS elements, `tufA`/`tufB`) is placed on one of them by the aligner and counted there, so the
  counts of identical copies are interchangeable and only their sum is meaningful.

`<results>/counting_method.json` records the annotation, the exact featureCounts options and the
orientation used for every sample.

## Read trimming

Paired-end reads are always trimmed with `fastp --detect_adapter_for_pe`. The former config key
`fastp_detect_adapter_for_pe` is ignored (with a warning).
