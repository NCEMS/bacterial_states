#File contains commands for making the indices and annotation files used in the SnakeMake workflow
#Required software: entrez-direct, mash, cactus, vg


##GRAPH ASSEMBLY AND VG INDEX##
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

#Running mash to get the divergence estimates for genomes
mash sketch -o ecoli_panel -k 21 -s 10000 genomes/*.fna -p 10
mash dist ecoli_panel.msh ecoli_panel.msh -p 10 > mash_distances.tsv

#Selecting a subset of genomes for graph genome
python select_genomes.py
awk -F [','] '{print $2}' representative_genomes.csv | sed '1d' > selected_genomes.txt
while read filepath; do
    filename=$(basename "$filepath")
    id=${filename%.fna}
    id=${id%.gz}
    id_clean=$(echo "$id" | tr '.' '_')
    echo -e "$id_clean\t$filepath" >> genomes.txt
done < selected_genomes.txt

#Running cactus with selected genomes (using singuarity to run, can be changed to other install)
cactus-pangenome ./jobstore genomes.txt \
  --outDir cactus_out_test \
  --outName ecoli_graph_test \
  --reference GCF_000005845_2_ASM584v2_genomic \
  --vcf --giraffe --gfa --gbz --xg




##ANNOTATION FILES##
grep -v "^#" GCF_000005845.2_ASM584v2_genomic.gtf | awk -v OFS='\t' '{print $1,$4,$5,$24":"$10,$6,$7}' | sed 's/\.3//g; s/[";]//g' > GCF_000005845.2_ASM584v2_genomic.bed

