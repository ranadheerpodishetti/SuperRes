#!/bin/bash
#   last-update: 2019-03-21 16:00
#SBATCH -J unetl1
#SBATCH -N 1 # Zahl der nodes, we have one only
#SBATCH --gres=gpu:1          # number of GPU cards (max=8)
#SBATCH --mem-per-cpu=6000    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks (max 40)
#SBATCH --cpus-per-task 10     # max 10/GPU CPU-threads needed (physcores*2)
#SBATCH --time 167:59:00 # set 0h59min walltime
## outcommended (does not work at the moment, ToDo):
#
exec 2>&1      # send errors into stdout stream
#env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG
#
# we have 8 GPUs, ToDo: add cublas-test as an example
# replace next 6 lines by your commands:
#Commandline Arguments:
#If no arguments supplied, it will use the default parameters written in this script
#First argument: name of the python main files, located inside the programROOT. If only one, then can supply just the value. If more than one, then should be enclosed by inverted coma ' ' to make them one argument.
#Second argument: path to the programROOT, starting from /scratch/tmp/schatter/Code. No need to add this to the argument, just rest of the path after this
#can supply either no arguent, only first or both arguments

#Default Parameters
programROOT=/scratch/podishet/code/
pythonMains=(main_new.py main_new_1.py) # If multiple 
#pythonMains=(gpu17TestAttempt.py) #If one


#If parameters were supplied
if [ $# -gt 0 ]; then
    pythonMains=() 
    IFS=' ' read -ra fileName <<< "$1"
    for i in "${fileName[@]}"; do
        pythonMains+=($i)
    done
    if [ $# -gt 1 ]; then
        programROOT=/home/schatter/Code/$2
    fi
fi

#Activate conda envioronment
source /scratch/podishet/condapytorch/etc/profile.d/conda.sh

conda activate SR

for i in "${pythonMains[@]}"; do 
  pythonMain=$i
  
  #Create full path of the program
  pyFullPath=$programROOT/$pythonMain
  
  #execute one python code add at it to the background, and launch the next ones
  python $pyFullPath & 
done

wait

echo "All Done"
#
# END of SBATCH-script