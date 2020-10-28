#!/bin/bash

# fetch VM username 
read -p 'your Username: ' username 
echo

echo "checking for sshpass..."
if ! [ -x "$(command -v sshfs)" ]; then
	echo "installing sshfs..."
  	sudo apt-get install sshfs
fi
echo "done installing"
echo "====================================="
echo

echo "Start Connecting..."

# mountpoint name
mountpoint="shared"

if [ -d "$mountpoint" ]; then
	sudo umount $mountpoint
  	rm -rf $mountpoint
  	mkdir -p $mountpoint
else
	mkdir $mountpoint
fi

# get initial char of username: important to poll the HPC filesystem path
#initial="$(echo $username | head -c 1)"

echo "Connecting and mounting shared filesystem locally in the folder: $mountpoint..."

sshfs -o IdentityFile=/home/$username/.ssh/id_rsa  \
sa_111128313555639196723@35.214.138.21:/home/sa_111128313555639196723/racing-kings-player/shared \
$mountpoint 

echo "Done Connecting"
echo 
