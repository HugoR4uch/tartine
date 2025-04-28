import ase
import ase.io
import numpy as np
import os
from typing import List, Dict, Optional


from . import fhi_aims_input_file_tools as fhi_aims_input_file_tools

def get_sim_details_from_dir_name(dir_name):
    # This is specific to the convention I have used
    parts = dir_name.split('_')  
    system_name = parts[0]
    miller_index = parts[1]
    traj_index = parts[2]
    return system_name, miller_index, traj_index


def load_scf_params(scf_params):

    if scf_params is None:
        mixer= 'pulay'
        charge_mixing = '0.1' 
        occupation_type= 'fermi'
        smearing = 0.2
        preconditioner = 2
        max_scf_cycles = 200
    else:
        mixer = scf_params['mixer']
        charge_mixing = scf_params['charge_mixing']
        occupation_type = scf_params['occupation_type']
        smearing = scf_params['smearing']
        preconditioner = scf_params['preconditioner']
        max_scf_cycles = scf_params['max_scf_cycles']

    return mixer, charge_mixing, occupation_type, smearing, preconditioner, max_scf_cycles



def make_ref_calc_files(trajectory_dirs : Dict[str,str],
                        systems_not_to_sample : Optional[Dict[str, List[str]]] = None,
                        equilibration_fraction : float = 0.5,
                        num_frames_to_sample_per_traj : int = 8,
                        training_frames_dir : str = './training_frames',
                        vacuum_above_top_atom : float = 30.0,
                        randomly_choose_frames : bool = True,
                        scf_params = None):
    


    mixer, charge_mixing, occupation_type, smearing, preconditioner, max_scf_cycles = load_scf_params(scf_params)

    
    mixer, charge_mixing, occupation_type, smearing, preconditioner, max_scf_cycles





    # Making the training frames directory (if not already made)
    if not os.path.exists(training_frames_dir):
        os.makedirs(training_frames_dir)

    # Loop over all simulation_runs directory paths
    for sim_dir_key in trajectory_dirs.keys():
        simulation_runs_dir = trajectory_dirs[sim_dir_key]
        print('Extracting Frames from: ',simulation_runs_dir)
        print('')

        # Getting Simulation details from directory names
        # Note, dir name format is <system_name>_<miller_index>_<traj_index>
        # Note, this format is used to extract info from simulation files in the dir
        simulation_details = {}
        for dir_name in os.listdir(simulation_runs_dir):
            if os.path.isdir(os.path.join(simulation_runs_dir, dir_name)):
                details = get_sim_details_from_dir_name(dir_name)
                simulation_details[dir_name] = list(details)

        # Grouping simulation details by system_name
        grouped_simulation_details = {}
        for dir_name, details in simulation_details.items():
            system_name, miller_index, traj_index = details
            key = (system_name+'_'+miller_index)
            if key not in grouped_simulation_details:
                grouped_simulation_details[key] = []
            grouped_simulation_details[key].append(dir_name)


        # Loop over all unique systems 
        for key, dirs in grouped_simulation_details.items():
            system_name = f"{key}"

            
            # Skipping systems which should'nt be sampled
            if systems_not_to_sample is not None:
                if sim_dir_key not in systems_not_to_sample:
                    print(f"Sampling all systems in {sim_dir_key}")
                else:
                    if system_name in systems_not_to_sample[sim_dir_key]:
                        print(f"Skipping sampling system: {system_name}")
                        continue


            #Getting name of system
            print('')
            print('Analysing System: ',system_name)
            
            #Loading trajectories
            trajectories = []
            traj_names = []
            first_sampled_frame_indices = []

            for i,traj_name in enumerate(dirs):

                print(f"  Loading trajectory: {traj_name}")
                traj_dir = simulation_runs_dir + '/' + traj_name

                #Handling traj loading for system names without a miller index
                traj_file = traj_name + '.xyz'
                if traj_file not in os.listdir(traj_dir):
                    traj_name_parts = traj_name.split('_')
                    print('traj_name_parts:',traj_name_parts)
                    if len(traj_name_parts) == 3:
                        traj_name_alt = f"{traj_name_parts[0]}_{traj_name_parts[2]}"
                        traj_file = traj_name_alt + '.xyz'
                    if traj_file not in os.listdir(traj_dir):
                        print(f"  Trajectory {traj_file} not found in {traj_dir}")
                        continue
                    else:
                        traj_name = traj_name_alt

                # NOTE: every 10th frame loaded
                traj = ase.io.read(traj_dir + '/' + traj_name + '.xyz',index='::10')     
                trajectories.append(traj)
                traj_names.append(traj_name)

            #Truncating equilibration frames
            for i in range(len(trajectories)):
                print(len(trajectories[i]))
                equilib_end_frame = int(  equilibration_fraction * len(trajectories[i]) ) 
                trajectories[i] = trajectories[i][equilib_end_frame:]
                first_sampled_frame_indices.append(equilib_end_frame)


            #Making ref calc dirs for sampled frames
            system_ref_calcs_dir = os.path.join(training_frames_dir, system_name) # Directory where ref calc files will be saved
            if not os.path.exists(system_ref_calcs_dir):
                os.makedirs(system_ref_calcs_dir)
            for i , traj in enumerate(trajectories):
                print(len(traj))
                num_frames = len(traj)
                
                
                #Either randomly sampling frames or picking frames at regular intervals
                if randomly_choose_frames:
                    frame_indices_to_sample = np.random.choice(num_frames, size=num_frames_to_sample_per_traj, replace=False)
                else:
                    #NOTE: regular sampling has not been tested
                    frame_indices_to_sample = np.linspace(0,num_frames-1,num_frames_to_sample_per_traj).astype(int)


                for frame_index in frame_indices_to_sample:
                    frame = traj[frame_index]
                    frame_name = f"{traj_names[i]}_{frame_index}"
                    frame_calc_dir = os.path.join(system_ref_calcs_dir, frame_name)
                    if not os.path.exists(frame_calc_dir):
                        os.makedirs(frame_calc_dir)
                        
                    print(f"  Extracted frame: {frame_name}")

                    cell = np.array( frame.get_cell() )

                    #cut cell so that distnace between water and substrate is  
                    #Move atoms to bottom of the cell
                    z_vals = frame.get_positions()[:,2]
                    max_z = np.max(z_vals)
                    min_z = np.min(z_vals)
                    height = cell[2][2] 
                    excess_height = height - (max_z - min_z)- vacuum_above_top_atom
                    new_z_vals = z_vals - min_z
                    frame.positions[:,2] = new_z_vals          
                    new_cell = cell
                    new_cell[2][2] = height - excess_height
                    frame.set_cell(new_cell)
                    print('cell: ', frame.cell)
                    ase.io.write(frame_calc_dir + f"/{frame_name}.xyz",frame,format='xyz')


                    #Creating FHI-AIMS geometry.in and control.in and SLURM submission file
                    fhi_aims_input_file_tools.make_FHI_AIMS_calc_dir(frame,
                                                                    frame_calc_dir,
                                                                    time_hrs=1.5,
                                                                    basis_sets_dir='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/fhi-aims/fhi-aims.240920_2/species_defaults/defaults_2020/light',
                                                                    default_control_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/control_default.in',
                                                                    default_slurm_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/run_fhi_aims_DFT_ARCHER2.slurm',
                                                                    project_name=f"{frame_name}",
                                                                    qos='standard',
                                                                    n_tasks=128,
                                                                    num_nodes=1,
                                                                    budget_allocation='e05-surfin-mic',
                                                                    mixer= mixer,
                                                                    charge_mixing = charge_mixing, 
                                                                    occupation_type= occupation_type,
                                                                    smearing =smearing,
                                                                    preconditioner = preconditioner,
                                                                    max_scf_cycles = max_scf_cycles,)


    print('Done')


def make_isolated_atom_ref_calc_dirs(elements,calc_dir='./isolated_atom_ref_calcs',scf_params=None):

    mixer, charge_mixing, occupation_type, smearing, preconditioner, max_scf_cycles = load_scf_params(scf_params)


    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)       

    
    for element in elements:
        z = ase.data.atomic_numbers[element]

        atom = ase.Atoms(element,positions=[[0,0,0]],cell=[20.5,21.0,22],pbc=[True,True,True])
        atom_calc_dir = os.path.join(calc_dir,element)
        if not os.path.exists(atom_calc_dir):
            os.makedirs(atom_calc_dir)
        fhi_aims_input_file_tools.make_FHI_AIMS_calc_dir(atom,
                                                    atom_calc_dir,
                                                    time_hrs=0.333,
                                                    default_control_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/control_default.in',
                                                    default_slurm_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/run_fhi_aims_DFT_ARCHER2.slurm',
                                                    project_name=f"{element}",
                                                    qos='short',
                                                    n_tasks=1, #only one thread as VERY few basis functions
                                                    num_nodes=1,
                                                    budget_allocation='e05-surfin-mic',
                                                    mixer= mixer,
                                                    charge_mixing = charge_mixing, 
                                                    occupation_type= occupation_type,
                                                    smearing = smearing,
                                                    preconditioner = preconditioner,
                                                    max_scf_cycles = max_scf_cycles)


