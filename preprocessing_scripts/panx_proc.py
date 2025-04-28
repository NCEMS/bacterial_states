import os
from glob import glob

input_dir = "core_genes"
output_fasta = "panx_core_transcriptome.fa"
mapping_file = "transcript_to_orthogroup.tsv"

fasta_out = open(output_fasta, "w")
map_out = open(mapping_file, "w")
map_out.write("transcript_id\torthogroup\n")

fa_files = sorted(glob(os.path.join(input_dir, "*_refined_na_aln.fa")))

for i, fa in enumerate(fa_files, 1):
    orth_id = f"orth_{i:05d}"
    with open(fa) as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip()[1:]
                new_header = f"{header}|{orth_id}"
                fasta_out.write(f">{new_header}\n")
                map_out.write(f"{header}\t{orth_id}\n")
            else:
                fasta_out.write(line)

fasta_out.close()
map_out.close()
