from software.tartines.reference_calc_tools import cp2k_input_file_tools
import numpy as np 
import ase.io
import ase
import copy 
from ase.data import vdw_radii
import os
import matplotlib.pyplot as plt

def plot_binding_curves(calc_dirs_path,plots_dir_path):
    
    binding_dirs = [d for d in os.listdir(calc_dirs_path) if os.path.isdir(os.path.join(calc_dirs_path, d)) and d.startswith('binding_')]
    system_names = list(set([d.split('binding_')[1]  for d in binding_dirs]))
    unqiue_system_names =np.unique( [name.split('_')[0] +'_'+ name.split('_')[1]   for name in system_names] )



    print(binding_dirs)
    print(unqiue_system_names)

    

    for system_name in unqiue_system_names:
        dirs = [d for d in binding_dirs if system_name in d]
        #print('dirs:',dirs)
        energies = {}
        displacements = {}
        convergence = {}
        cell_z_vals = {}
        substrate_min_z_vals = {}
        substraet_max_z_vals = {}
        O_z_vals = {}

        for dir in dirs:
            binding_index = dir.split('_')[-1]
            print('binding_index:',binding_index)
            binding_dir_path = os.path.join(calc_dirs_path,dir)
            #print('binding_dir_path:',binding_dir_path)
            cp2k_output_file = os.path.join(binding_dir_path,'cp2k.out')
            #print('cp2k_output_file:',cp2k_output_file)
            with open(cp2k_output_file, 'r') as file:
                filedata = file.read()

            if 'SCF run NOT converged.' in filedata:
                convergence[binding_index] = False
            else:
                convergence[binding_index] = True
            

            print('Converged:',convergence[binding_index])

            energy_line = [line for line in filedata.split('\n') if 'Total energy:' in line]
            if energy_line:
                energy = float(energy_line[0].split()[-1])
                print('Energy:', energy)
                energies[binding_index] = energy


            cp2k_input_file = os.path.join(binding_dir_path,f'{dir}.inp')

            with open(cp2k_input_file, 'r') as file:
                    filedata = file.read()

            cell_lines = [line for line in filedata.split('\n') if line.strip().startswith(('A', 'B', 'C'))]
            cell = np.zeros((3, 3))
            for line in cell_lines:
                parts = line.split()
                if parts[0] == 'A':
                    cell[0] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'B':
                    cell[1] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'C':
                    cell[2] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])


            cell_z_vals[binding_index] = cell[2][2]


            system = ase.io.read(os.path.join(binding_dir_path,dir+'.xyz'),format='xyz')
            n_atoms = len(system)
            substrate= system[:-3]
            water = system[-3:]
            substrate_z_vals = substrate.positions[:,2]
            water_z_vals = water.positions[:,2]
            displacements[binding_index] = float(np.min(water_z_vals) - np.max(substrate_z_vals))

            O_z_vals[binding_index] = np.min(water_z_vals)
            substrate_min_z_vals[binding_index] = np.min(substrate_z_vals)
            substraet_max_z_vals[binding_index] = np.max(substrate_z_vals)




    convergence_mask = np.array(list(convergence.values()))

    displacement_vals =  np.array(list(displacements.values()))
    energy_vals = np.array(list(energies.values())) 
    O_z_vals = np.array(list(O_z_vals.values()))
    substrate_min_z_vals = np.array(list(substrate_min_z_vals.values()))
    substraet_max_z_vals = np.array(list(substraet_max_z_vals.values()))
    cell_z_vals = np.array(list(cell_z_vals.values()))

    print('substrate_min_z_vals:',substrate_min_z_vals)
    print('substraet_max_z_vals:',substraet_max_z_vals)
    print('O_z_vals:',O_z_vals)
    print('cell_z_vals:',cell_z_vals)
    print('Displacements:',displacement_vals)

    # min_e = np.min(energies)
    energy_vals_mev_per_atom = (energy_vals ) *  1000  * 27.2114 /n_atoms  
    print('Energies:',energy_vals*n_atoms)
    print('Energies peratom:',energy_vals_mev_per_atom)

    plt.scatter(displacement_vals[convergence_mask], energy_vals_mev_per_atom[convergence_mask], marker='.',color='black',label='Converged')
    plt.scatter(displacement_vals[~convergence_mask], energy_vals_mev_per_atom[~convergence_mask], marker='x',color='black',label='Not converged')
    
    plt.xlabel('Displacement (A)')
    plt.ylabel('Energy (meV/atom)')

    name,miller = system_name.split('_')
    if miller is '': 
        print_name = name
    else:
        print_name = name + ' ' + f"({miller})"

    
    # data = np.loadtxt('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/examples/water_graphene_0_leg_xavi_binding_curve.csv',
    #                   skiprows=1,delimiter=',',
    #                   dtype=float)
    # x=data[:,0]
    # y=data[:,1]
    # y_min = np.min(y)
    # y = y/abs(y_min) * 60
    # plt.scatter(x,y,marker='x',color='black',label='Xavi')


    plt.legend()
    plt.title(print_name + ' Binding Curve')
    plt.grid() 

    plt.savefig(os.path.join(plots_dir_path,system_name+'binding_curve.png'))
    plt.close()

    #system_name = system_dir_path.split('/')[-2]


def make_binding_curve_calc_dirs(substrates_dir,
                                 cluster,
                                 calc_dir_path,
                                 time_hrs = 0.2,
                                 adsorp_elements_dict=None,
                                 wave_cutoff=1000,
                                 config_file_path = '../config_files',
                                 queue_type = 'standard',
                                 default_cp2k_inp_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/single_point.inp',
                                 smearing_dict=None):
    
    """
    Creates directories for binding curve calculations based on substrate files.

    Args:
        substrates_dir (str): Directory containing substrate files.
        cluster: Cluster information (type not specified).
        calc_dir_path (str): Path to the directory where calculation directories will be created.
        time_hrs (float, optional): Time in hours for the calculation. Default is 0.2 (12 mins).
        adsorp_elements_dict (dict, optional): Dictionary mapping substrate names to adsorption elements. Defaults to None.
        wave_cutoff (int, optional): Wave cutoff value for calculations. Defaults to 1000.
        config_file_path (str): Path to the DFT config file (pseudopotentials, basis sets, etc.). Defaults to '../config_files'.
        queue_type (str, optional): Queue type for the slurm file. Default is 'standard'. Can also be 'quick' or 'standard'.        
        default_cp2k_inp_file_path (str, optional): Path to the default CP2K input file. Defaults to '/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/single_point.inp'.
        smearing_dict (dict, optional): Dictionary for smearing parameters. Defaults to None.

    Returns:
        bool: True if directories are created successfully.
    """

    if smearing_dict is None:
        smearing_dict = {}



    if cluster not in ['cp2k','archer2']:
        print('Cluster not recognised. Use "cp2k" or "archer2".')
        return False

    substrate_files = [f for f in os.listdir(substrates_dir) if os.path.isfile(os.path.join(substrates_dir, f))]
    print("Found substrate files:", substrate_files)

    if adsorp_elements_dict is None:
        adsorp_elements_dict = {}

    for substrate_file in substrate_files:
        substrate_name = substrate_file.split('.')[0]
        if substrate_name not in adsorp_elements_dict:
            print('No adsorption element specified for',substrate_name)
            adsorp_elements_dict[substrate_name] = None

    for substrate_file in substrate_files:
        substrate_name = substrate_file.split('.')[0]
        adsorp_element = adsorp_elements_dict[substrate_name]

        
        if substrate_name not in smearing_dict.keys():
            smearing=False
        else:
            smearing=smearing_dict[substrate_name]
        print('Smearing: ',smearing)

        print('Creating binding curve directories for',substrate_name)

        cp2k_binding_curve_calc_dir(substrate_path = substrates_dir+'/'+substrate_file,
                                    cluster = cluster,
                                    calc_dir_path= calc_dir_path,
                                    default_cp2k_inp_file_path = default_cp2k_inp_file_path,
                                    config_file_path = config_file_path,
                                    smearing = smearing,
                                    queue_type=queue_type,
                                    time_hrs = time_hrs,
                                    wave_cutoff = wave_cutoff,
                                    adsorp_element = adsorp_element)
    
    return True


def cp2k_binding_curve_calc_dir(substrate_path,cluster,calc_dir_path,default_cp2k_inp_file_path,config_file_path,smearing,queue_type='standard',time_hrs = 0.2, adsorp_element=None,wave_cutoff=1000,inter_slab_distance=15.0):
    """     
    Generates the CP2K input files for the binding curve calculations for water. z-values are sampled up to the distance of closest approach (taken as the sum of van der Waals radii).

    Parameters:
    substrate_path (str): Path to the substrate file.
    cluster (str): Name of the cluster to run the calculations on. Either 'cp2k' or 'archer2'.
    calc_dir_path (str): Path to the directory where the CP2K input and xyz files will be saved.
    default_cp2k_inp_file_path (str): Path to the default CP2K input file template.
    default_slurm_file_path (str): Path to the default slurm file template.
    config_file_path (str): Path cp2k input fil will use during the calculation to the DFT config file (pseudopotentials,basis sets etc...).
    smearing (bool): Whether to use Fermi smearing in the calculation.
    time_hrs (float, optional): Time in hours for the calculation. Default is 0.2 = 12 mins.
    adsorp_element (str, optional): Element symbol of the species the water will adsorb on. Default is None, in which case will randomly pick a topmost (largest z val) atom.
    project_name (str, optional): Name of the project. Default is 'project'.
    queue_type (str, optional): Queue type for the slurm file. Default is 'standard'. Can also be 'quick' or 'long'
    wave_cutoff (float, optional): Wavefunction cutoff energy in Ry. Default is 1200.
    inter_slab_distance (float, optional): Distance (along z-axis) between adsorbate water and nearest image substrate. Default is 15.

    Returns:
    bool: True if the function executes successfully.
    """
    

    if cluster not in ['cp2k','archer2']:
        print('Cluster not recognised. Use "cp2k" or "archer2".')
        return False

    substrate=ase.io.read(substrate_path,format='proteindatabank')
    substrate_name = substrate_path.split('/')[-1].split('.')[0]
    substrate_z_vals = substrate.positions[:,2]

    # Shifting to 5 A above bottom of cell
    substrate_z_vals = substrate.positions[:,2]
    substrate_bottom_z_val = np.min(substrate_z_vals)
    substrate.positions[:,2] += 5 - substrate_bottom_z_val
    substrate_z_vals = substrate.positions[:,2]

    #Choosing the atom the water will bind to
    if adsorp_element is not None:
        adsorp_element_indices = [atom.index for atom in substrate if atom.symbol == adsorp_element]
    else:
        adsorp_element_indices = [atom.index for atom in substrate]
    max_adsorp_element_z_val = np.max(substrate_z_vals[adsorp_element_indices])
    substrate_bottom_z_val = np.min(substrate_z_vals)
    candidate_substrate_z_vals_adsorption_indices= np.where([substrate_z_vals==max_adsorp_element_z_val])[1]
    adsorp_index = np.random.choice(candidate_substrate_z_vals_adsorption_indices)
    if adsorp_element is None:
        adsorp_element = substrate[adsorp_index].symbol
    print('Water binding to:',adsorp_element)
    x = substrate.positions[adsorp_index][0]
    y = substrate.positions[adsorp_index][1]

    # Approximating equilibrium distance as sum of vdw radii
    substrate_vdw_radius = vdw_radii[ase.data.atomic_numbers[adsorp_element]]
    print('vdw radius:',substrate_vdw_radius)
    if np.isnan(substrate_vdw_radius):
        print('No vdw radius for ',adsorp_element,' using vdw radius of 1.5 A')
        substrate_vdw_radius = 1.5
    O_vdw_radius = vdw_radii[ase.data.atomic_numbers['O']]
    z_closest_approach = O_vdw_radius + substrate_vdw_radius
    z_equil_displ_vals  = np.array([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2]) # 11 points
    z_approach_vals = z_closest_approach + z_equil_displ_vals 

    z_approach_min = np.min(z_approach_vals)
    if z_approach_min < 2.5:
        diff_z = 2.5 - z_approach_min
        z_approach_vals += diff_z

    print('z-approach vals:',z_approach_vals)
    angle = 14.5 /180 * np.pi
    water = ase.Atoms('H2O', positions=np.array([[0,-0.95,0],[0.95*np.cos(angle),0.95*np.sin(angle),0],[0,0,0]]))



    # Creating binding curve configs
    for i,z in enumerate(z_approach_vals):
        # Add water
        adsorbate_water = copy.deepcopy(water)
        adsorbate_water.positions[:,2] = z + max_adsorp_element_z_val
        adsorbate_water.positions[:,0] += x
        adsorbate_water.positions[:,1] += y
        substrate.extend(adsorbate_water)

        # Adjust cell dimensions
        substrate.cell[2][2] = (z+ max_adsorp_element_z_val)+(inter_slab_distance - 5)# -5 as already shifted up 
        
        dir_path = os.path.join(calc_dir_path, f'binding_{substrate_name}_{i}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        #Adding files for calc
        ase.io.write(dir_path+f'/binding_{substrate_name}_{i}.xyz',
                     substrate,
                     format='xyz')
        
        cp2k_input_file_tools.create_cp2k_input_file(substrate,
                                                dir_path+f'/binding_{substrate_name}_{i}.inp',
                                                f'binding_{substrate_name}_{i}.xyz', 
                                                default_cp2k_inp_file_path,
                                                config_file_path = config_file_path, 
                                                project_name = f'binding_{substrate_name}_{i}', 
                                                wave_cutoff=wave_cutoff,
                                                smearing = smearing)

        slurm_submit_filename = f'binding_{substrate_name}_{i}.slurm'
        

        if cluster == 'csd3':
            default_slurm_file_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/DFT_run_CSD3.slurm'

            cp2k_input_file_tools.create_cp2k_CSD3_slurm_file(slrum_file_path = dir_path+'/'+slurm_submit_filename,
                                                time_hrs = time_hrs,
                                                project_name = f'{substrate_name}_{i}',
                                                default_file_path=default_slurm_file_path,
                                                mail=True,
                                                n_tasks=76,
                                                num_nodes=1,
                                                partition='icelake',
                                                budget_allocation='MICHAELIDES-SL2-CPU')
        if cluster == 'archer2':
            default_slurm_file_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/DFT_run_ARCHER2.slurm'
            cp2k_input_file_tools.create_cp2k_ARCHER2_slurm_file(slrum_file_path= dir_path+'/'+slurm_submit_filename,
                                                            time_hrs =time_hrs ,
                                                            project_name = f'{substrate_name}_{i}',
                                                            default_file_path = default_slurm_file_path,
                                                            qos=queue_type,
                                                            mail=True,
                                                            n_tasks=128,
                                                            cpus_per_task=1,
                                                            num_nodes=1,
                                                            partition='standard',
                                                            budget_allocation='e05-surfin-mic')

        # Remove water - will add at new position in next iteration
        del substrate[-3:]
        
    print()
    
    return True