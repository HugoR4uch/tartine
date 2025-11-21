#!/bin/bash

#SBATCH -J <INSERT:job name>
#SBATCH -A <INSERT:budget_allocation> ##THEORYRIG-SL2-CPU
#SBATCH -p icelake ##icelake
#SBATCH -t <INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs>
#SBATCH -N <INSERT:number_of_nodes> ##1
#SBATCH -n <INSERT:number_of_tasks> ##76

##SBATCH --qos=<INSERT:qos>

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=hr492@cam.ac.uk
#SBATCH --no-requeue


# Setup the batch environment
module purge
module load intel/mkl/2020.4
module load rhel8/default-icl

# Set the number of threads to 1. This prevents any threaded system libraries from automatically using threading.
export OMP_NUM_THREADS=1

# Set stacksize to unlimited for FHI-aims
ulimit -s unlimited

# Define path to the FHI-aims executable
folder=/rds/user/hr492/hpc-work/software/bin
aims=aims.250320.scalapack.mpi.x
output=aims.out

# srun launches the parallel program based on the SBATCH options. Would --cpu-bind=cores be useful?
srun --distribution=block:block --hint=nomultithread ${folder}/${aims}  > ${output}
