import ase
import numpy as np



def create_cp2k_CSD3_slurm_file(slrum_file_path,time_hrs,project_name,default_file_path,mail=True,n_tasks=76,num_nodes=1,partition='icelake',budget_allocation='MICHAELIDES-SL2-CPU'):
    """
    Creates a SLURM batch script for running a single CP2K calc on the CSD3 cluster.
    Parameters:
    slrum_file_path (str): Path (including filename) to the output SLURM script file.
    time_hrs (float): Total job time in hours. 
    project_name (str): Name of the project/job. Will also be taken as cp2k input filename.
    default_file_path (str): Path to the default SLURM script template.
    mail (bool, optional): Whether to send email notifications. Defaults to True.
    n_tasks (int, optional): Number of tasks. Defaults to 76 = CSD3 num cores/node.
    num_nodes (int, optional): Number of nodes. Defaults to 1.
    partition (str, optional): Partition to submit the job to. Defaults to 'icelake'.
    budget_allocation (str, optional): Budget allocation for the job. Defaults to 'MICHAELIDES-SL2-CPU'.
    Returns:
    None
    """
    
    #SBATCH -J <INSERT:job_name>
    #SBATCH -A <INSERT:budget_allocation> ##MICHAELIDES-SL2-CPU
    #SBATCH -p <INSERT:partition> ##icelake
    #SBATCH -t <INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs>
    #SBATCH -N <INSERT:number_of_nodes> ##1
    #SBATCH -n <INSER:number_of_tasks> ##76
    #SBATCH --mail-type= <INSERT:mail_option> #BEGIN,END

    
    time_mins = (time_hrs - int(time_hrs))*60
    time_secs = (time_mins - int(time_mins))*60
    time_hrs = int(time_hrs)
    time_mins = int(time_mins)
    time_secs = int(time_secs)


    with open(default_file_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('<INSERT:job_name>', project_name)
    filedata = filedata.replace('<INSERT:budget_allocation>', budget_allocation)
    filedata = filedata.replace('<INSERT:partition>', partition)
    filedata = filedata.replace('<INSERT:time_limit_hrs>', str(time_hrs))
    filedata = filedata.replace('<INSERT:time_limit_mins>', str(time_mins))
    filedata = filedata.replace('<INSERT:time_limit_secs>', str(time_secs))
    filedata = filedata.replace('<INSERT:number_of_nodes>', str(num_nodes))
    filedata = filedata.replace('<INSERT:number_of_tasks>', str(n_tasks))
    filedata = filedata.replace('<INSERT:number_of_tasks>', str(n_tasks))
    filedata = filedata.replace('<INSERT:cp2k_input_filename>', project_name)
    if mail == True:
        filedata = filedata.replace('<INSERT:mail_option>', 'BEGIN,END')
    else:
        filedata = filedata.replace('<INSERT:mail_option>', 'NONE')

    with open(slrum_file_path, 'w') as file:
        file.write(filedata)

    return 


def create_cp2k_input_file(atoms, inp_file_path, coords_path, default_file_path, config_file_path='../config_files',project_name='project', wave_cutoff=1200,smearing=False):
    """
    Creates a CP2K input file by replacing placeholders in a default input file template.
    Parameters:
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure and cell information.
    inp_file_path (str): Path where the generated CP2K input file will be saved.
    coords_path (str): Path to the coordinates file to be used in the CP2K input.
    default_file_path (str): Path to the default CP2K input file template.
    config_file_path (str, optional): Path to the CP2K config files. Default is '../config_files'.
    project_name (str, optional): Name of the project. Default is 'project'.
    wave_cutoff (int, optional): Wavefunction cutoff energy in Ry. Default is 1200.
    Returns:
    None
    """

    cell = np.array(atoms.get_cell())

    with open(default_file_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('<INSERT:path_to_coord>', coords_path)
    filedata = filedata.replace('<INSERT:wave_cutoff>', str(wave_cutoff))
    filedata = filedata.replace('<INSERT:project_name>', project_name)
    filedata = filedata.replace('<INSERT:config_file_path>', config_file_path)
    if smearing ==True:
        filedata = filedata.replace('<INSERT:smear_on>', 'ON')
        filedata = filedata.replace('<INSERT:added_MOs>', 'ADDED_MOS 30')
        filedata = filedata.replace('<INSERT:OT_true>', 'OFF')
        filedata = filedata.replace('<INSERT:diagonalization>', 'ON')
    else: 
        filedata = filedata.replace('<INSERT:smear_on>', 'OFF')
        filedata = filedata.replace('<INSERT:added_MOs>', '')
        filedata = filedata.replace('<INSERT:OT_true>', 'ON')
        filedata = filedata.replace('<INSERT:diagonalization>', 'OFF')

    filedata = filedata.replace('<INSERT:config_file_path>', config_file_path)

    filedata = filedata.replace('<INSERT:A_vector>', str(cell[0][0]) + ' ' + str(cell[0][1]) + ' ' + str(cell[0][2]))
    filedata = filedata.replace('<INSERT:B_vector>', str(cell[1][0]) + ' ' + str(cell[1][1]) + ' ' + str(cell[1][2]))
    filedata = filedata.replace('<INSERT:C_vector>', str(cell[2][0]) + ' ' + str(cell[2][1]) + ' ' + str(cell[2][2]))
    with open(inp_file_path, 'w') as file:
        file.write(filedata)
    
    return




def create_cp2k_ARCHER2_slurm_file(slrum_file_path,time_hrs,project_name,default_file_path,qos='short',mail=True,n_tasks=128,cpus_per_task=1,num_nodes=1,partition='standard',budget_allocation='e05-surfin-mic'):
    """
    Creates a SLURM batch script for running CP2K on the ARCHER2 supercomputer.
    Parameters:
    -----------
    slrum_file_path (str): The path where the SLURM file will be saved.
    time_hrs (float): The walltime for the job in hours. (If fraction convert to mins and secs).
    project_name (str): The name of the job.
    default_file_path (str): The path to the default SLURM file template.
    qos (str, optional): The quality of service for the job. Default is 'standard' (24hrs) can also be 'short' (20mins) and 'long' (48hrs).
    mail (bool, optional): Whether to receive email notifications. Default is True.
    n_tasks (int, optional): The number of tasks per node. Default is 128.
    cpus_per_task (int, optional): The number of CPUs per task. Default is 1.
    num_nodes (int, optional): The number of nodes to use. Default is 1.
    partition (str, optional): The partition to submit the job to. Default is 'standard'. 
    budget_allocation (str, optional): The budget allocation code. Default (for me) is 'e05'.
    Returns:
    --------
    None
    """

    

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

    
    time_mins = (time_hrs - int(time_hrs))*60
    time_secs = (time_mins - int(time_mins))*60
    time_hrs = int(time_hrs)
    time_mins = int(time_mins)
    time_secs = int(time_secs)


    with open(default_file_path, 'r') as file:
        filedata = file.read()


    filedata = filedata.replace('<INSERT:job_name>', project_name)
    filedata = filedata.replace('<INSERT:budget_allocation>', budget_allocation)
    filedata = filedata.replace('<INSERT:partition>', partition)
    filedata = filedata.replace('<INSERT:time_limit_hrs>', str(time_hrs))
    filedata = filedata.replace('<INSERT:time_limit_mins>', str(time_mins))
    filedata = filedata.replace('<INSERT:time_limit_secs>', str(time_secs))
    filedata = filedata.replace('<INSERT:number_of_nodes>', str(num_nodes))
    filedata = filedata.replace('<INSERT:number_of_tasks>', str(n_tasks))
    filedata = filedata.replace('<INSERT:cpus_per_tasks>', str(cpus_per_task))
    filedata = filedata.replace('<INSERT:qos>', qos)
    filedata = filedata.replace('<INSERT:cp2k_input_filename>', project_name)

    if mail == True:
        filedata = filedata.replace('<INSERT:mail_option>', 'BEGIN,END')
    else:
        filedata = filedata.replace('<INSERT:mail_option>', 'NONE')

    with open(slrum_file_path, 'w') as file:
        file.write(filedata)

    return 