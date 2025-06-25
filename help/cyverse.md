# CyVerse instructions
## Basic setup

After logging onto CyVerse, navigate to the Discovery Environment page and hit launch:
![image](https://github.com/user-attachments/assets/74bda542-cdef-46a2-a37f-8bb2948efc3a)

This will show you the many option available on CyVerse for analysis. We want to use the Cloud Shell which can be found on the bar to the left here:
![image](https://github.com/user-attachments/assets/84bf62da-37b1-4de2-8407-ba32547466c0)

The Cloud Shell will give us access to a Linux command line on a virtual computer. This command line is how we can run our SnakeMake workflow. After launching, the window should look like this:
![image](https://github.com/user-attachments/assets/18847dee-bde8-4427-b414-e36f06db180f)


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
For this tutorial, we will be using a publically available dataset from NCBI.
