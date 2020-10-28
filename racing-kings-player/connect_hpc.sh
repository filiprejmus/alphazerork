#!/bin/bash

# checking if sshpass is alrewady installed otherwise install it
echo "checking for sshpass..."
if ! [ -x "$(command -v sshpass)" ]; then
	echo "installing sshpass..."
  	sudo apt-get install sshpass
fi
echo "====================================="
echo

# fetching Tubit login data
echo "Connecting to HPC..."
read -p 'Tubit Username: ' username 
read -sp 'Tubit Password: ' Password 
echo

# connecting a shell console to HPC using ssh
sshpass -p $Password ssh "$username@gateway.hpc.tu-berlin.de"

exit 0