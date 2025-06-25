# CyVerse instructions
This page gives a detailed tutoral of how to use the inGEST pipeline with CyVerse. This tutorial is geared towards beginner users of command line tools and may be redundant for those with more experience.

## Basic setup

....... 

## Dependencies and file download
CyVerse does not automatically have Apptainer installed so we must install two dependencies and then Apptainer (we have sudo access on CyVerse's virtual machines). 
```
#Refreshing package list
sudo apt update 
#Installing dependencies
sudo apt install -y rpm cpio
#Installing Apptainer
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - ./build/apptainer_local
```

This may take a moment to install. To check installation once it is finished, you can check for the help info:
```
build/apptainer_local/bin/apptainer
```
![image](https://github.com/user-attachments/assets/3eb8a203-9d3d-49e5-90ed-bb4a6b183ba2)

Next, we need to clone this repository and pull down the code for the pipeline. We will make a new directory called analysis and change into that directory first. Then we will use git to clone the repo:
```
mkdir analysis
cd analysis
git clone https://github.com/NCEMS/bacterial_states.git
```

You can view the current files in this folder with the command "ls" and then change directories into the directory we just downloaded called "bacterial_states":
```
ls
cd bacterial_states
```
![image](https://github.com/user-attachments/assets/acfc6a3c-eb15-44a2-a130-3cdbeb6e0d6e)

We next need to download the folder for the index and annotation files used in this pipeline. This folder is available on CyVerse and can be downloaded using their gocmd tool. We must first config this tool using our CyVerse account:
```
gocmd init
```
Enter the following information when prompted:
![image](https://github.com/user-attachments/assets/3c110861-b600-4c5c-bbef-40475c4e2039)

Now we can download the resource folder and unzip it from CyVerse. This may take a moment as this directory is quite large.
```
gocmd get --progress -f -K --icat /iplant/home/shared/NCEMS/working-groups/bacterial-states/resources.tar.gz
tar -xvzf resources.tar.gz
```

Now we are ready to run the pipeline!

## Data analysis
For this tutorial, we will be using a publically available dataset from NCBI. We will be comparing a wildtype of BW25113 with a knockout of CysG. To download this data, we get it directly from SRA using wget and rename the files for simplicity. We will place the raw sequencing files in a folder called data:
```
mkdir data
cd data
wget -O WT_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/009/SRR2547469/SRR2547469.fastq.gz
wget -O WT_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/008/SRR2547468/SRR2547468.fastq.gz
wget -O KO_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/007/SRR2547467/SRR2547467.fastq.gz
wget -O KO_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/000/SRR2547470/SRR2547470.fastq.gz
ls
```

![image](https://github.com/user-attachments/assets/693a05ab-1464-49c2-ae7c-1bddda4b9281)

Now that we have the data files, code, and index files, we are ready to run the entire thing. First will we go into the pipeline folder. Then we will copy the example config file from the help folder in our currently folder.
Note: ".." represents going up one directory and "." means the current directory. When using the "cp" command we are saying, copy this file to our current directory. Feel free to use the "ls" command to view what is in each directory. Additionally, the command "pwd" gives information about which directory you are currently in.
```
cd ../pipeline/
cp ../help/config.yaml .
ls
```

Next we need to edit the config files to match our current setup. The config file currently looks like:

![image](https://github.com/user-attachments/assets/1f71cca2-df02-4de6-bc83-f449bacbf5e6)


To change files on the command line, the simplest text editor to use is nano. To use nano on our file:
```
nano config.yaml
```

This command will bring up the text editor and allow us to change the config file based on our current experiment. You must use the arrow keys in order to move around the file. We will be removing the R2 lines as we are dealing with single-end sequencing data. We will also need to change the file names to the current file names in our data folder. We can leave the condition names the same in the file as we still have a knockout and wildtype setup. We can also leave the sample names the same. The experiment name is the name of the folder that will hold the entire analysis, which we can rename to "tutorial". To save your file, hit ctrl+X then Y (for yes) then enter. You can view your file with:
```
cat config.yaml
```
![image](https://github.com/user-attachments/assets/9e76ad6c-9eed-443e-a6fc-d0c1da897d1d)

Make sure you are in the correct directory with "ls" or "pwd":

Time to run the pipeline! We are telling apptainer to pull down the image that includes all the packages needed in the ".sif" file. It will use our "config.yaml" file as input for the SnakeMake pipeline found in the "Snakefile". We are also telling it to run "all" of the rules in that Snakefile:
```
../../build/apptainer_local/bin/apptainer exec \
  --bind "$(pwd)/..:/mnt" \
  ../resources/build/rnaseq_vg.sif \
  snakemake --cores 12 all
```
