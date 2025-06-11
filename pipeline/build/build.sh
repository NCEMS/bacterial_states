#!/bin/bash
#SBATCH --account=open
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40GB


#Apptainer version 1.4.0-1.el8
/usr/bin/apptainer build rnaseq_vg.sif Singularity.def
