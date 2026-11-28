"""reference/: the genome FASTA of the inGEST core bundle renamed NC_000913 (the LOCUS name of the bundle's GenBank
file, which breseq uses), its bowtie2 index (to place clipped read parts) and reference/loci.fasta with its bowtie2
index: the flhDC, crl, oppA and fim regions as contigs named <name>:<offset> (genome position = contig position +
offset), and the fim region again with the fim switch inverted (fiminv). Needs the core bundle (`ecoli_ingest download
core`), samtools and bowtie2-build on PATH.
    python scripts/make_reference.py"""
import subprocess, sys
from common import REFERENCE, GBK, BUNDLE_FASTA, FASTA, GENOME_INDEX, LOCI_INDEX

REGIONS = {"flhDC": (1975001, 1982000), "crl": (254001, 262000), "oppA": (1296001, 1304000),
           "fim": (4539001, 4548000)}
FIMS = (4542682, 4542995)            # fim switch, 314 bp between and including the 9-bp inverted repeats
LENGTH = 4641652
rc = lambda s: s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def main():
    REFERENCE.mkdir(exist_ok=True)
    for f in (GBK, BUNDLE_FASTA):
        if not f.exists():
            sys.exit(f"{f} missing: download the inGEST core bundle (ecoli_ingest download core)")
    if not FASTA.exists():
        seq = "".join(l.strip() for l in open(BUNDLE_FASTA) if not l.startswith(">")).upper()
        if len(seq) != LENGTH:
            sys.exit(f"{BUNDLE_FASTA}: sequence of {len(seq)} bp, expected {LENGTH}")
        FASTA.write_text(">NC_000913\n" + "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80)) + "\n")
        subprocess.run(["samtools", "faidx", str(FASTA)], check=True)
    if not GENOME_INDEX.with_suffix(".1.bt2").exists():
        subprocess.run(["bowtie2-build", "-q", str(FASTA), str(GENOME_INDEX)], check=True)

    def region(a, b):
        out = subprocess.run(["samtools", "faidx", str(FASTA), f"NC_000913:{a}-{b}"],
                             capture_output=True, text=True, check=True).stdout
        return "".join(out.split("\n")[1:])
    loci = REFERENCE / "loci.fasta"
    with open(loci, "w") as f:
        for name, (a, b) in REGIONS.items():
            f.write(f">{name}:{a - 1}\n{region(a, b)}\n")
        a, b = REGIONS["fim"]
        s = region(a, b)
        i, j = FIMS[0] - a, FIMS[1] - a + 1
        f.write(f">fiminv:{a - 1}\n{s[:i] + rc(s[i:j]) + s[j:]}\n")
    subprocess.run(["bowtie2-build", "-q", str(loci), str(LOCI_INDEX)], check=True)
    print("reference ready:", REFERENCE)


if __name__ == "__main__":
    main()
