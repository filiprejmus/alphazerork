#!/bin/bash

# fetch Tubit username 
echo "Fetching Tubit login data"
read -p 'Tubit Username: ' username 
echo

echo "=====================================================================
Give the path to folder having the data to share on HPC.
PRESS 'Enter' if you just wish to clone the HPC filesystem locally.
The folder could have files and sub folders.
The path could be 'relative' or 'absolute' !
NOTE: if the given folder contains some data having the same name as 
some data on the HPC then those data on the HPC will get overwritten!
====================================================================="
echo

# get the path of the folder to share on HPC
read -p 'Path to sharing folder: ' folder


# mountpoint name
mountpoint="hpc"

# umount the old existing mountpoint if it still exists and create new one  
if [ -d "$mountpoint" ]; then
	sudo umount $mountpoint
  	rm -rf $mountpoint
  	mkdir -p $mountpoint
else
	mkdir $mountpoint
fi

# get initial char of username: important to poll the HPC filesystem path
initial="$(echo $username | head -c 1)"

# connect to the hpc filesystem and mount it locally in the created mountpoint
echo "Connecting and mounting TU HPC filesystem locally in the folder: $mountpoint..."
sshfs $username@gateway.hpc.tu-berlin.de:/home/users/$initial/$username $mountpoint/ 
echo "Done Connecting and cloning HPC Data locally !"
echo

# copy the shared folders and files to HPC id wished
if [ -d "$folder" ]; then
	echo "Start copying data to HPC..."
	cp -a $folder/. $mountpoint/ > /dev/null 2>&1
	cp run_hpc.sbatch $mountpoint/ > /dev/null 2>&1
	cp run_hpc.sh $mountpoint/ > /dev/null 2>&1
	cp requirements.txt $mountpoint/ > /dev/null 2>&1
	cp run.py $mountpoint/ > /dev/null 2>&1
	echo "Done copying data to HPC !"
else
	echo "NO data to copy from local machine to HPC."
fi
echo

echo "done preparing data !"
exit 0

