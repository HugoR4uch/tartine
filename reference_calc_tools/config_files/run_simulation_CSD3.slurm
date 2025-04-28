#!/bin/bash
#SBATCH --job-name=<INSERT:job_name>
#SBATCH -A <INSERT:budget_allocation> ##MICHAELIDES-SL2-GPU
#SBATCH --nodes=<INSERT:number_of_nodes>
#SBATCH --ntasks=<INSERT:number_of_tasks>
#SBATCH --cpus-per-task=<INSERT:number_of_cpus_per_task>
#SBATCH --time=<INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs>
#SBATCH --gres=gpu:<INSERT:number_of_gpus>
##SBATCH --qos=INTR
#SBATCH -p <INSERT:partition> ##ampere
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=hr492@cam.ac.uk


module purge
module load intel-mkl-2017.4-gcc-5.4.0-2tzpyn7
module load rhel8/default-amp
module load cudnn
# Python

echo "Running job"

source /home/xr223/software/mace-lat-venv/bin/activate


python run_MACE_simulation.py > sim_interface.out



    dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

