#File contains commands for making the indices used in the SnakeMake workflow
#Required software: salmon,

#Getting transcriptome/orthologs from PanX, retrieved April 17, 2025
wget https://data.master.pangenome.org/dataset/Escherichia_coli/core_gene_alignments.zip
unzip core_gene_alignments.zip -d core_genes
#Combining the orthologs into one file
python panx_proc.py
#Creating Salmon dictionary with transcriptome
salmon index -t panx_core_transcriptome.fa -i salmon_index --threads 8
