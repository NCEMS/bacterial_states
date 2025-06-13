# CyVerse instructions
## Basic setup
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
