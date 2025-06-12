# Bacterial Cell State Atlas

## Command line instructions
### Basic setup
If you do not have Singularity or Apptainer already installed, this must be done first as the image for the workflow is built in Apptainer. Two dependencies require super user (sudo) access to install Apptainer (rpm2cpio and cpio). If you do not have these dependencies installed, you must contact a system admin in order to add them. Change the location of the local install if you wish (current going to root directory). 
```
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - ~/apptainer_local
```
During installation, 


## CyVerse instructions
### Basic setup
CyVerse does not automatically have Apptainer installed so we must install two dependencies and then Apptainer (we have sudo access on CyVerse's virtual machines). 
```
#Refreshing package list
sudo apt update 
#Installing dependencies
sudo apt install -y rpm cpio
#Installing Apptainer
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - ./build/apptainer_local
```
To check installation (should give help info):
```
build/apptainer_local/bin/apptainer
```
