#!/bin/bash --login

#### Author: Yair Litman (with Hugo's adjustments)
#### Date: 13-Jun-24

#SBATCH --job-name=<INSERT:job name>
#SBATCH --nodes=<INSERT:number_of_nodes> #default 1 
#SBATCH --ntasks-per-node=<INSERT:number_of_tasks> #default 128
#SBATCH --cpus-per-task=1
#SBATCH --time=<INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs> # Format=Hours:Mins:Seconds. Select a time within job queue limits. See --qos=standard below.

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=<INSERT:budget_allocation>
# Choices here: standard, highmem, gpu
#SBATCH --partition=standard
# Choices here: short, standard, long, with max walltime of 20 min, 24 hours and 48 hours, respectively.
#SBATCH --qos=<INSERT:qos>

# Setup the batch environment
module purge
module load epcc-job-env
module load PrgEnv-gnu # If you compiled the code with Cray compilers remove this line!

# Set the number of threads to 1. This prevents any threaded system libraries from automatically using threading.
export OMP_NUM_THREADS=1

# Set stacksize to unlimited for FHI-aims
ulimit -s unlimited

# Define path to the FHI-aims executable
folder=/mnt/lustre/a2fs-work2/work/e05/e05/har492/bin
aims=aims.x
output=aims.out

# srun launches the parallel program based on the SBATCH options. Would --cpu-bind=cores be useful?
srun --distribution=block:block --hint=nomultithread ${folder}/${aims}  > ${output}
