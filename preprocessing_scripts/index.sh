#File contains commands for making the indices used in the SnakeMake workflow
#Required software: salmon, entrez-direct, mash, cactus, 






##SALMON INDEX##
#Getting transcriptome/orthologs from PanX, retrieved April 17, 2025
wget https://data.master.pangenome.org/dataset/Escherichia_coli/core_gene_alignments.zip
unzip core_gene_alignments.zip -d core_genes
#Combining the orthologs into one file
python panx_proc.py
#Creating Salmon dictionary with transcriptome
salmon index -t panx_core_transcriptome.fa -i salmon_index --threads 8






##VG INDEX##
#Getting ecoli genomes, retrived April 22, 2025
mkdir genomes
cd genomes
esearch -db assembly -query "Escherichia coli[Organism] AND latest[filter] AND (complete genome[filter] OR chromosome level[filter])" | efetch -format docsum | xtract -pattern DocumentSummary -element FtpPath_RefSeq > ftp_links.txt
while read link; do
    base=$(basename "$link")
    file="${base}_genomic.fna.gz"
    wget -nc "$link/$file"
done < ftp_links.txt
gunzip -f *.fna.gz
cd ..

#Running mash to get divergence estimates
mash sketch -o ecoli_panel -k 21 -s 10000 genomes/*.fna -p 10
mash dist ecoli_panel.msh ecoli_panel.msh -p 10 > mash_distances.tsv

#Selecting 50 genomes the most divergent from K-12 (our reference), figures made in script
python select_genomes.py
while read filepath; do
    filename=$(basename "$filepath")
    id=${filename%.fna}
    id=${id%.gz} 
    id_clean=$(echo "$id" | tr '.' '_')
    echo -e "$id_clean\t$filepath" >> genomes.txt
done < selected_genomes.txt

#Running cactus with 50 selected genomes (using singuarity to run, can be changed to other install)
singularity exec cactus_latest.sif cactus-pangenome ./jobstore genomes.txt \
  --outDir cactus_out_test \
  --outName ecoli_graph_test \
  --reference GCF_000005845_2_ASM584v2_genomic \
  --vcf --giraffe --gfa --gbz --xg



grep -v "^#" GCF_000005845.2_ASM584v2_genomic.gtf | awk -v OFS='\t' '{print $1,$4,$5,$24":"$10,$6,$7}' | sed 's/\.3//g; s/[";]//g' > GCF_000005845.2_ASM584v2_genomic.bed

