<a href="https://ncems.psu.edu/">
  <img src="https://ncems.psu.edu/wp-content/uploads/2024/02/psu-ncems_logo-wordmark-white33.png" style="width: 60%;">
</a>

# Integrative Gene Expression Synthesis Tool (inGEST)

An open-source computational pipeline for integrating bacterial gene expression data to support cross-study analysis and bacterial cell-state characterization.

## Table of Contents

* [Overview](#overview)
* [Workflow](#workflow)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data](#data)
* [Usage](#usage)
* [Results](#results)
* [Citation](#citation)
* [Contact](#contact)
* [License](#license)

## Overview

Gene expression datasets generated across different studies can vary due to experimental conditions, growth environments, sequencing protocols, and other technical factors. Integrating these datasets provides an opportunity to investigate bacterial transcriptional patterns across diverse biological contexts.

inGEST provides a workflow for processing, harmonizing, and analyzing bacterial gene expression data across studies.

To overcome bacterial RNA-seq batch correction and sample comparison issues, we created the Integrative Gene Expression Synthesis Tool (inGEST). The inGEST pipeline analyzes raw RNA-seq data in a strain agnostic framework and compares samples using a conditional autoencoder. This approach allows for batch correction across diverse datasets and creates a latent space of embeddings that reflects biological patterns in the data. The E. coli gene expression atlas we created by using inGEST on thousands publicly available RNA-seq datasets provides a global view of the transcriptional programs that define cell states in this model bacterium and reveals conserved and condition-specific mechanisms underlying growth transitions. Researchers can use inGEST to analyze new E. coli RNA-seq datasets and gain biological insights by placing them within the context of prior experiments.

## Workflow

<p align="center">
  <img width="1676" height="672" alt="image" src="https://github.com/user-attachments/assets/ec85364c-aafd-4156-a194-15e99ee847f9" />
</p>
Figure 1. Overview of the inGEST pipeline.

## Features

- **Data preprocessing:** Prepare gene expression data for downstream integration.
- **Workflow automation:** Organize analysis steps into reproducible computational workflows.
- **Post-processing:** Support downstream analysis and interpretation of processed data.
- **Extensibility:** Provide a framework for incorporating additional datasets and analyses.

## Repository Structure

```text
bacterial_states/
├── help/                     # Documentation and help resources
├── pipeline/                 # Main pipeline workflows
├── preprocessing_scripts/    # Data preprocessing
├── postprocessing_scripts/   # Downstream analysis
└── README.md
```

## Requirements

For the basic setup of this pipeline, you will need a few dependencies installed. These dependencies allow us to access our apptainer image and download necessary files from GitHub and CyVerse. Please follow the links below to install these dependencies:

[Apptainer](https://apptainer.org/docs/admin/main/installation.html)

[Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

[GoCommands](https://learning.cyverse.org/ds/gocommands/#table-of-contents)

Note: Apptainer requires sudo (super user) access to properly install. Contact an administrator if you do not have these permissions on your system.


## Installation

First, we need to clone this repository and pull down the code for the pipeline. We will make a new directory called analysis and change into that directory first. Then we will use git to clone the repo:
```
mkdir analysis
cd analysis
git clone https://github.com/NCEMS/bacterial_states.git
```

We can change directories into bacterial_states and examine what we pulled down from GitHub:
```
cd bacterial_states
ls
```
![image](https://github.com/user-attachments/assets/f32a1053-c14f-4709-b799-17655ff17aef)

We next need to download the folder for the index and annotation files used in this pipeline. This folder is available on CyVerse and can be downloaded using their gocmd tool. We must first config this tool using our CyVerse account:
```
gocmd init
```
Enter the following information when prompted:
![image](https://github.com/user-attachments/assets/3c110861-b600-4c5c-bbef-40475c4e2039)

Now we can download the resource folder and unzip it from CyVerse. This may take a while as this directory is quite large.
```
gocmd get --progress -f --icat /iplant/home/shared/NCEMS/working-groups/bacterial-states/resources.tar.gz
tar -xvzf resources.tar.gz
```

Now we are ready to run the pipeline!

## Data
For this tutorial, we will be using a publically available dataset from NCBI. We will be comparing a wildtype of BW25113 with a knockout of CysG. To download this data, we get it directly from SRA using wget and rename the files for simplicity. We will place the raw sequencing files in a folder called data:
```
mkdir data
cd data
wget -O WT_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/007/SRR2547467/SRR2547467.fastq.gz
wget -O WT_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/008/SRR2547468/SRR2547468.fastq.gz
wget -O KO_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/009/SRR2547469/SRR2547469.fastq.gz
wget -O KO_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR254/000/SRR2547470/SRR2547470.fastq.gz
ls
```

![image](https://github.com/user-attachments/assets/22ce84d8-11c9-457b-9353-e2e3246fb2f8)

Now that we have the data files, code, and index files, we are ready to run the entire thing. First will we go into the pipeline folder. Then we will copy the example config file from the help folder in our current folder.

```
cd ../pipeline/
cp ../help/config.yaml .
ls
```

Next we need to edit the config files to match our current setup. The config file currently looks like:

![image](https://github.com/user-attachments/assets/1f71cca2-df02-4de6-bc83-f449bacbf5e6)


Next, we need to edit the config file based on our current experiment. We will be removing the R2 lines as we are dealing with single-end sequencing data. We will also need to change the file names to the current file names in our data folder. We can leave the condition names the same in the file as we still have a knockout and wildtype setup. We can also leave the sample names the same. The experiment name is the name of the folder that will hold the entire analysis, which we can rename to "tutorial". You should have something like this:
```
cat config.yaml
```
![image](https://github.com/user-attachments/assets/3d53f738-4bf4-4b75-88fc-73d55bcec877)


## Usage

Time to run the pipeline! We are telling apptainer to pull down the image that includes all the packages needed in the ".sif" file. It will use our "config.yaml" file as input for the SnakeMake pipeline found in the "Snakefile". We are also telling it to run "all" of the rules in that Snakefile and to use 12 processors. Change the number of processors as is appropriate for your resources:

```
apptainer exec \
  --bind "$(pwd)/..:/mnt" \
  ../resources/build/rnaseq_vg.sif \
  snakemake --cores 12 all
```

## Results

During the run you will see some output as it completes each step. The final results will be in this directory as shown, with subdirectories for specific steps:

![image](https://github.com/user-attachments/assets/887a0193-094f-4143-8653-879cac80b88d)

For differential expression analysis, the final results will be in the deseq2 folder. For example:

![image](https://github.com/user-attachments/assets/32d74d65-1b3c-4b5d-8e61-6076a799f55f)


## Citation

If you use inGEST in your research, please cite the associated publication or project documentation when available.


## Contact

For questions, bug reports, or contributions, please open an issue in this repository or contact the project maintainers.

## License

This project is licensed under the [MIT License]. Copyright (c) 2026 National Synthesis Center for Emergence in the Molecular and Cellular Sciences (NCEMS)

