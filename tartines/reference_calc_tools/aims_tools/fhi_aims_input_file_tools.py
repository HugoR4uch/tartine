import ase
import numpy as np
import os
import ase.io




def make_FHI_AIMS_calc_dir( atoms, 
                            calculation_dir,
                            time_hrs,
                            basis_sets_dir='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/fhi-aims/fhi-aims.240920_2/species_defaults/defaults_2020/light',
                            default_control_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/control_default.in',
                            default_slurm_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/run_fhi_aims_DFT_ARCHER2.slurm',
                            project_name='project',
                            qos='short',
                            n_tasks=128,
                            num_nodes=1,
                            budget_allocation='e05-surfin-mic',
                            mixer= 'pulay',
                            charge_mixing = '0.1', 
                            occupation_type= 'fermi',
                            smearing = 0.2,
                            preconditioner = 2,
                            max_scf_cycles = 200,
                            dipole_correction=True,
                            ):
    
    """
    Create FHI-aims input files for a given set of atoms.
    This method generates the necessary input files for an FHI-aims calculation,
    including the geometry.in and control.in files.

    Parameters:

    Returns:
    None
    """

    # Create calculation directory if it doesn't exist
    if not os.path.exists(calculation_dir):
        os.makedirs(calculation_dir)
        print(f"Directory {calculation_dir} created")
    else:
        print(f"Directory {calculation_dir} already exists")


    # Create geometry.in file
    ase.io.write(os.path.join(calculation_dir,'geometry.in'), atoms, format='aims',scaled=False)

    # List of elements in system
    all_elements = []
    for atom in atoms:
        all_elements.append(atom.symbol)
    all_unique_elements = np.unique(all_elements) 

    print(all_unique_elements)

    create_FHI_AIMS_control_file(os.path.join(calculation_dir,'control.in'), 
                                 all_unique_elements,
                                 basis_sets_dir,
                                 default_control_file_path,
                                 mixer,
                                 charge_mixing, 
                                 occupation_type,
                                 smearing,
                                 preconditioner,
                                 max_scf_cycles,
                                 dipole_correction=dipole_correction,
                                 )



    create_fhi_aims_ARCHER2_slurm_file(calculation_dir,
                                       time_hrs,
                                       project_name,
                                       default_slurm_file_path,
                                       qos=qos,
                                       n_tasks=n_tasks,
                                       num_nodes=num_nodes,
                                       budget_allocation=budget_allocation,
                                       )




def create_FHI_AIMS_control_file(new_input_file_path,
                                list_of_elements,
                                basis_sets_dir,
                                default_control_file,
                                mixer= 'pulay',
                                charge_mixing = '0.1', 
                                occupation_type= 'fermi',
                                smearing = 0.2,
                                preconditioner = 2,
                                max_scf_cycles = 200,
                                dipole_correction=True,
                                ):
    """
    Creates a new FHI-aims control file by appending basis set data for specified elements.
    NOTE: Methfessel-Paxton smearing is not supported in this code! (As you need to specify order)

    Parameters:
    new_input_file_path (str): The path where the new control file will be saved.
    list_of_elements (list): A list of elements for which the basis sets will be included.
    basis_sets_dir (str): The directory containing the basis sets for the elements.
    default_control_file (str): The path to the default control file.
    mixer (str, optional): The type of mixer to use. Can be 'linear', 'broyden' or 'pulay'. Default is 'pulay'. 
    charge_mixing (str, optional): The charge mixing parameter. Default is '0.1'.
    occupation_type (str, optional): The type of occupation. Can be 'fermi', 'methfessel-paxton' or 'gaussian'. Default is 'fermi'.
    smearing (float, optional): The smearing temperature in eV. Default is 0.2.
    preconditioner (int, optional): The preconditioner value for kerker preconditioner. Default is 2.
    max_scf_cycles (int, optional): The maximum number of SCF cycles. Default is 200.

    Returns:
    None
    """


    """
    #Physical settings
    xc            revPBE
    spin          none
    relativistic  atomic_zora scalar
    d3

    sc_iter_limit 80

    #Mixing settings
    mixer              pulay
    n_max_pulay        10
    charge_mix_param   0.05

    #Smear settings
    occupation_type gaussian 0.1

    #k-point grid
    k_grid   1  1    1 

    #Efficiency and accuracy flags
    use_dipole_correction

    elsi_restart read_and_write 10

    #Output dipole
    compute_forces .true.
    final_forces_cleaned .true.
    """


    basis_sets_dirs = os.listdir(basis_sets_dir)
    # print(basis_sets_dirs)

    with open(default_control_file, 'r') as file:
        filedata = file.read()


    filedata = filedata.replace('<INSERT:scf_limit>', str(max_scf_cycles))
    filedata = filedata.replace('<INSERT:mixer>', str(mixer))
    filedata = filedata.replace('<INSERT:charge_mixing>', str(charge_mixing))
    filedata = filedata.replace('<INSERT:occupation_type>', occupation_type)
    filedata = filedata.replace('<INSERT:smearing>', str(smearing))
    filedata = filedata.replace('<INSERT:preconditioner>', str(preconditioner))

    if dipole_correction == True:
        dipole_keyword=str('use_dipole_correction')
    else:
        dipole_keyword=str('#use_dipole_correction')
    filedata = filedata.replace('<INSERT:dipole_correction>', dipole_keyword)


    for element in list_of_elements:
        element_basis_filename = [f for f in basis_sets_dirs if '.' not in f and f.split('_')[1] == element]


        element_basis_set_file = os.path.join(basis_sets_dir, element_basis_filename[0])

        with open(element_basis_set_file, 'r') as basis_file:
            basis_data = basis_file.read()
        
        filedata = filedata + "\n" + basis_data

    
            

    with open(new_input_file_path, 'w') as new_file:
        new_file.write(filedata)

    return 


def create_fhi_aims_ARCHER2_slurm_file(calculation_dir,time_hrs,project_name,default_slurm_file_path,qos='standard',n_tasks=128,num_nodes=1,budget_allocation='e05-surfin-mic',partition='standard'):
    """
    Creates a SLURM batch script for running FHI-aims on the ARCHER2 supercomputer.

    Parameters:
    calculation_dir (str): The directory where the SLURM file will be saved.
    time_hrs (float): The walltime for the job in hours. (If fraction, convert to mins and secs).
    project_name (str): The name of the job.
    default_slurm_file_path (str): The path to the default SLURM file template.
    qos (str, optional): The quality of service for the job. Default is 'short'.
    n_tasks (int, optional): The number of tasks per node. Default is 128.
    num_nodes (int, optional): The number of nodes to use. Default is 1.
    budget_allocation (str, optional): The budget allocation code. Default is 'e05-surfin-mic'.

    Returns:
    None
    """
    # Default SLURM file template:

    #SBATCH --job-name=<INSERT:job name>
    #SBATCH --nodes=<INSERT:number_of_nodes> #default 1 
    #SBATCH --ntasks-per-node=<INSERT:number_of_tasks> #default 128
    #SBATCH --cpus-per-task=1
    #SBATCH --time=<INSERT:time_limit_hrs>:<INSERT:time_limit_mins>:<INSERT:time_limit_secs> # Format=Hours:Mins:Seconds. Select a time within job queue limits. See --qos=standard below.

    # Replace [budget code] below with your project code (e.g. t01)
    #SBATCH --account=e05-surfin-mic
    # Choices here: standard, highmem, gpu
    #SBATCH --partition=standard
    # Choices here: short, standard, long, with max walltime of 20 min, 24 hours and 48 hours, respectively.
    #SBATCH --qos=<INSERT:qos>



    time_mins = (time_hrs - int(time_hrs))*60
    time_secs = (time_mins - int(time_mins))*60
    time_hrs = int(time_hrs)
    time_mins = int(time_mins)
    time_secs = int(time_secs)


    with open(default_slurm_file_path, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('<INSERT:job name>', project_name)
    filedata = filedata.replace('<INSERT:time_limit_hrs>', str(time_hrs))
    filedata = filedata.replace('<INSERT:time_limit_mins>', str(time_mins))
    filedata = filedata.replace('<INSERT:time_limit_secs>', str(time_secs))
    filedata = filedata.replace('<INSERT:number_of_nodes>', str(num_nodes))
    filedata = filedata.replace('<INSERT:number_of_tasks>', str(n_tasks))
    filedata = filedata.replace('<INSERT:budget_allocation>', budget_allocation)
    filedata = filedata.replace('<INSERT:qos>', qos)
    filedata = filedata.replace('<INSERT:partition>', partition)


    with open(os.path.join(calculation_dir,'run_DFT.slurm'), 'w') as file:
        file.write(filedata)

    return 


