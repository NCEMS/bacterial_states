"""Paths and settings shared by the scripts of this example.

The example folder holds the tracked tables (reads.tsv, sample_states.tsv) and everything a run writes (samples.tsv,
reference/, results/, figures/). Reads, breseq runs and the loci BAMs go to $SCR (default: scratch/ in the example
folder); a full run needs several hundred GB there."""
import os
from pathlib import Path

EX = Path(__file__).resolve().parents[1]
ROOT = EX.parents[1]
SCR = Path(os.environ.get("SCR", EX / "scratch"))
RESULTS, FIGURES, REFERENCE = EX / "results", EX / "figures", EX / "reference"
ANNOTATION = ROOT / "resources/annotation"     # the inGEST core bundle (`ecoli_ingest download core`)
GBK = ANNOTATION / "GCF_000005845.2_ASM584v2_genomic.gbff"      # NC_000913.3 (= U00096.3), NCBI GenBank flat file
BUNDLE_FASTA = ANNOTATION / "GCF_000005845.2_ASM584v2_genomic.fna"
FASTA = REFERENCE / "NC_000913.fasta"          # the bundle FASTA, renamed NC_000913 as in the GenBank LOCUS line
GENOME_INDEX, LOCI_INDEX = REFERENCE / "NC_000913", REFERENCE / "loci"
FEATURES = ROOT / "resources/annotation/GCF_000005845.2_ASM584v2_counting_v2.features.tsv"
SAMPLES, READS, STATES = EX / "samples.tsv", EX / "reads.tsv", EX / "sample_states.tsv"

SETS = ["FlhDC", "FliA", "Crp", "RpoS", "ArcA", "Fnr", "NarL", "IHF", "Fur"]   # RegulonDB regulons (whole)

# Studies left out of the trend-line fit, the lineage markers, the tests and the figures; kept, flagged, in the
# per-sample tables.
EXCLUDE_STUDIES = {
    "GSE224463": "genotype labels swapped: the LSP_A 0 and 90 min samples titled wild-type lack rpoS (GSM7024640, "
                 "GSM7024641) and the ones titled ΔrpoS have it (GSM7024712, GSM7024713); found from rpoS coverage",
    "GSE64021": "abnormal MG1655 clone (GSM1562981): CGSC 7740 alleles at IS5U, glpR and gatC but crl intact, and private "
                "rpoS frameshift, spoT A26E, glpD and cspC deletions",
}
