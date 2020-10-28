#!/bin/bash

##################### HPC parameters #######################
###### to be modified manually according to your needs #####

#SBATCH --job-name=RacingKingsZero	    # Job Name
#SBATCH --nodes=1
#SBATCH --ntasks=2		        # number of processes P (CPU-Cores) 
# #SBATCH --ntasks-per-node=2   # number of tasks per node
# #SBATCH --cpus-per-task=64	    # number of CPU-Cores pro Process P
# #SBATCH --hint=nomultithread     # IMPORTANT: hyper-threading is activated; switch off and attribute whole core to task
#SBATCH --gres=gpu:tesla:2	    # number of GPUs
#SBATCH --mem=1000G              # amount of resident main memory PER NODE(!)
# #SBATCH --mem-per-cpu=1GB     # amount of resident main memory PER CORE(!) (set only, if needed)
#SBATCH --exclusive

## %x: job name; %j: job id; %N: node; %t: task id; %a: array id (and others)
#SBATCH -o %x.%j.%N.out  #redirect SLURM's stdout (please see man 1 sbatch for explanation of replacement symbols)
#SBATCH -e %x.%j.%N.err  #redirect SLURM's stderr (please see man 1 sbatch for explanation of replacement symbols)

## Max Walltime:
#SBATCH --time=7-00:00:00         # estimated runtime 

## specify partition: smp or gpy or standard
#SBATCH --partition=smp

## get current Job-Status per Mail:
#SBATCH --mail-type=ALL     #BEGIN|END|FAIL|ALL|NONE
#SBATCH --mail-user=khaled.kahouli@campus.tu-berlin.de

##############################################################


##################### run skript #######################
a=0
while true; do
	module load nvidia/cuda/10.1
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-6.0.1.5/lib
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beegfs/home/cluster/cuda/cuda-10.1/lib64
	echo "running ..."
	echo $a
	py="AlphaZero_venv"
	srun --cpus-per-task=64 --mem=490GB  --unbuffered --ntasks=1 --exclusive -o %x.%j.%N.%t$a.out -e %x.%j.%N.%t$a.err --gres=gpu:tesla:1 ${py}/bin/python3 rkplayer/GameGenerator.py &
	srun --cpus-per-task=64 --mem=490GB --unbuffered --ntasks=1 --exclusive -o %x.%j.%N.%t$a.out -e %x.%j.%N.%t$a.err  --gres=gpu:tesla:1  ${py}/bin/python3 rkplayer/GameGenerator.py & 
	wait
	a=`expr $a + 1`
        sleep 5
	echo "done running..."
done
##echo "resultst are in file result.txt and errors in file %job_name.%job_id.err"

########################################################



