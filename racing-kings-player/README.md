# Alpha Zero Racing Kings Player

This project implements the AlphaZero generative Algorithm for the game board 'Racing Kings'.

---
**Important**

Development is done in python version 3.

---

## Getting Started

### Virtual Environment (optional)

Create a virtual environment.

```bash
python3 -m venv venv
```
Activate the virtual environment.
```bash
source venv/bin/activate
```

### Dependency Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages this projects depends on.

#### Using python3 virtual environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Without virtual environment

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Testing
### Running unit-tests

#### Using python3 virtual environment

```bash
python -m unittest discover -v
```

#### Without virtual environment

```bash
python3 -m unittest discover -v
```
This runs all unit-tests with filenames starting with test.

## HPC 
Hhe training is done on the TU Berlin High Performance Cluster (HPC).
Here is a small tutorial how to use the HPC using Linux distribution.
1. Connecting to TU Berlin using VPN :
    
    to use the HPC you should use gateway from TU Berlin which you cannot access from your home only using VPN.
    Here we use openconnect.
    *  install openconnect:
    
        ```bash
        sudo apt-get install openconnect
        ```
    *   connect to VPN using openconnect:
    
        ```bash
        echo "$password" | sudo openconnect vpn.tu-berlin.de --user='$username' --passwd-on-stdin
        ```
        with `$username` is your TuBit username and `$password` your TuBit password.

2. Preparing Data:
    
    here we are going to prepare the scripts and the data that we will run or need on the HPC. We use here sshfs protocol.
    * create folder and name it `shared` for example.
    * prepare python wrapper script named `run.py` that contains your main Method of the project to be run. be sure that this script is in the same folder as the bash script `prepare_hpc.sh` and the folder `shared`.
    * open the file `run_hpc.sbatch` and on the top in the section `HPC paramneters` give the needed nodes, CPUs, GPUs and other parameters that you want to allocate and use.
    * prepare a file  ` requirments.txt` having all the needed libraries and dependencies in the project.
    * put all your folders and data that you want to share to HPC inside this folder for exemple here you will put all the folders on this Gitlab (rkplayer,requirments.txt,...).
    * run the preparing script in your bash from the location where you have created the folder to share :

        ```bash
        ./prepare_hpc.sh
        ```
        this script is interactive and will ask you for some inputs like TuBit username and password, the name of the folder to share and sudo access password. So write the information and validate with `enter`.
        When the script is done successfully it will prompt `done preparing data !` in the console otherwise with error message because you did something wrong
        
3.  Connect to HPC:
    
    now we need to connect a shell to the HPC and run commands on thge cluster. For this purpose you should run the following script:
        
    ```bash
    ./connect_hpc.sh
    ```
    this script is interactive and will ask you for some inputs like TuBit username and password. So write the information and validate with `enter`.
    When the script is done successfully it will prompt a new bash session on the cluster.
        
4. running scripts and wrappers on the HPC:
    
    Now we are going the main scripts and wrappers on the HPC.
    *   first of all check if the shared folders were successfully prepared by typing in the bash console the command:
        
        ```bash
        ls
        ```
    *   the give execution writes to the script `run_hpc.sh` by running this command:
        
        ```bash
        chmod +x run_hpc.sh
        ```
    *   now run the script `run_hpc.sh ` wich will create python virtual environment, install the dependencies and libraries in the file  `requirments.txt ` and then run the sbatch script `run_hpc.sbatch` which in turn will run the wrapper  `run.py `:
    
        ```bash
        ./run_hpc.sh
        ```
        after this script is done you will find the results of the wrapper in the file `results.txt`, the node stdout in the file `%x.%j.%N.out` and the node stderr in the file `%x.%j.%N.err` with %x is job name, %j is job id and %N is node.
        the script will send you emails on the given E-mail adress in the sbatch parameters to inform you with the status of your job.
5.  important Notes:
    *   make sure where you place your folders and scripts. they should be in the write places as specified above.
    *   the parameters to be adjusted in `run_hpc.sbatch` should begin with one `#` if they are relevant or more than one `#` if you want to comment them.
    *   the existing partitions and nodes are to be found <a href=https://hpc.tu-berlin.de/doku.php?id=hpc:hardware>here<a>.
    *   some useful commands to run on the cluster after connecting with `connect_hpc.sh` are :
        *   `scancel <jobid>` (signal/stop running jobs with id 'jobid').
        *   `sstat` (infos wrt. running jobs).
        *   `squeue` (show queues).
        *   `sinfo` (show node infos).
        *   `batch` jobs: sbatch (batch jobs to queue).
        *   more infos about each command are to be triggered with `man 1 sinfo`. `sstat -u username` to get infos of running job of specefic user.
    