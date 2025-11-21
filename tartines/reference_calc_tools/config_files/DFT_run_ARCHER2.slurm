#!/bin/bash --login

#### Author: Yair Litman ( + Hugo's modifications)
#### Date: 31-May-2024


#SBATCH --job-name=<INSERT:job_name>
#SBATCH --nodes=<INSERT:number_of_nodes>
#SBATCH --ntasks-per-node=<INSERT:number_of_tasks>
#SBATCH --cpus-per-task=<INSERT:cpus_per_tasks>
#SBATCH --time=<INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs>

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=<INSERT:budget_allocation>
# Choices here: standard, highmem, gpu
#SBATCH --partition=<INSERT:partition>
# Choices here: short, standard, long, with max walltime of 20 min, 24 hours and 48 hours, respectively.
#SBATCH --qos=<INSERT:qos>


# Load the relevent CP2K module
module load cp2k

export OMP_NUM_THREADS=1

# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

input_filename=<INSERT:cp2k_input_filename>.inp

output_file=cp2k.out
error_file=cp2k.err

rm -f cp2k.out cp2k.err
echo  {"START",$( date -u)} >>LIST
srun --hint=nomultithread --distribution=block:block cp2k.psmp -i ${input_filename}  >> ${output_file} 2>> ${error_file} || exit 1
echo  {"END",$( date -u)} >>LIST
~                                     