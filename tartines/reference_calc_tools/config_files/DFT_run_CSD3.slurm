#!/bin/bash

#SBATCH -J <INSERT:job_name>
#SBATCH -A <INSERT:budget_allocation> ##MICHAELIDES-SL2-CPU
#SBATCH -p <INSERT:partition> ##icelake
#SBATCH -t <INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs>
#SBATCH -N <INSERT:number_of_nodes> ##1
#SBATCH -n <INSERT:number_of_tasks> ##76

#SBATCH --mail-type= <INSERT:mail_option> #BEGIN,END
#SBATCH --mail-user=hr492@cam.ac.uk
#SBATCH --no-requeue



#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')


source ~/.bashrc
. /etc/profile.d/modules.sh

    module purge
    module load rhel8/default-icl
    module load cp2k/2023.2/intel/intel-oneapi-mkl/intel-oneapi-mpi/35tqki2w

module list

workdir="$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}]
export I_MPI_PIN_DOMAIN=omp:compact
export I_MPI_PIN_ORDER=scatter



#------------------CP2K variables------------------------------------
cp2k_input=<INSERT:cp2k_input_filename>
cp2k_output=cp2k.out
cp2k_error=cp2k.err
options="-i ${cp2k_input} > ${cp2k_output} 2> ${cp2k_error}  "
CMD_CP2K="mpirun -ppn $mpi_tasks_per_node -np $np cp2k.popt $options"


cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

#-----------------------------------------
echo  {"init",$( date -u)} >>LIST
echo 'launch cp2k'
    eval ${CMD_CP2K}
echo  {"Final ",$( date -u)} >>LIST
