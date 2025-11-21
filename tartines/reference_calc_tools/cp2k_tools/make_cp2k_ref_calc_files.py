import ase
import ase.io
import numpy as np
import os
import sys

import software.tartines.reference_calc_tools.cp2k_input_file_tools as cp2k_input_file_tools

simulation_runs_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/metal_interfaces_screening/17-12-2024_metal_interface_simulations/simulation_runs'
#simulation_runs_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/oxide_interfaces/simulation_runs'
#simulation_runs_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces/dyn_substrate_trajs'
#simulation_runs_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces/fixed_substrate_trajs'


#path_to_default_cp2k_input_file ='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/single_point.inp'
#plane_wave_cutoff=1200

path_to_default_fhi_aims_inp_file='.'
equilibration_fraction = 0.5
cluster= 'ARCHER2' # 'CSD3' or 'ARCHER2'
num_frames_to_sample_per_traj = 7 # 5  (+ 1 for validation)
run_walltime_hours = 0.75



training_frames_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/training_frames'
if not os.path.exists(training_frames_dir):
    os.makedirs(training_frames_dir)



specific_systems_to_sample= systems = [
    "BN",
    "graphene",
    "WS2",
    "WSe2",
    "MoS2",
    "MoSe2",
    "SiO2-H_0001",
    "kaolinite-Al_",
    "kaolinite-Si_" #Yair said no
    "NaF",
    "NaCl" #Yair said no
    "KF",
    "KCl",
    "KI",
    "AgCl",
    "AgI",
    "a-TiO2_101",
    "Au_100",
    "Au_111",
    "Au_110",
    "Cu_111",
    "Cu_110",
    "Cu_100",
    "Pt_100",
    "Pt_110",
    "Pt_111",
    "Mg_0001",
    'Ru_0001' #Yair said no
    "Al_111",
    "Ti_0001",
    "Pd_111"
]



#specific_systems_to_sample = ['graphene_'] #['Cu_100'] # If None, all systems will be sampled
#specific_systems_to_sample = ['graphene_','NaF_','NaCl_','NaBr_','KCl_','KBr_','KI_','AgCl_','BN_','MoS2_','MoSe2_','WS2_','WSe2_']
#specific_systems_to_sample = ['NaI_','KF_','AgI_']

# Systems where we want to use smearing
# smearing_dir = {'Al_111':True,
#                 'Au_100':True,
#                 'Au_110':True,
#                 'Au_111':True,
#                 'Pd_100':True,
#                 'Pd_110':True,
#                 'Pd_111':True,
#                 'Pt_100':True,
#                 'Pt_110':True,
#                 'Pt_111':True,
#                 'Cu_100':True,
#                 'Cu_110':True,
#                 'Cu_111':True,
#                 'Ru_0001':True,
#                 'Ti_0001':True,
#                 'Mg_0001':True,
#                 'Pt-non-orth_111':True,
#                 'graphene_':True,
#                 }

print('Extracting Frames')
print('')


def get_sim_details_from_dir_name(dir_name):
    # This is specific to the convention I have used
    parts = dir_name.split('_')  
    system_name = parts[0]
    miller_index = parts[1]
    traj_index = parts[2]
    return system_name, miller_index, traj_index

# Getting Simulation details from directory names
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

    if specific_systems_to_sample is not None and system_name not in specific_systems_to_sample:
        print(f"Skipping sampling system: {system_name}")
        continue

    #Getting name of system
    print('')
    print('Analysing System: ',system_name)
    
    #Loading trajectories
    trajectories = []
    traj_names = []
    for i,traj_name in enumerate(dirs):

        if i != 0:
            continue 

        print(f"  Loading trajectory: {traj_name}")
        traj_dir = simulation_runs_dir + '/' + traj_name

        #Handling case where system name does not have a miller index
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

        traj = ase.io.read(traj_dir + '/' + traj_name + '.xyz',index='::10')    
        trajectories.append(traj)
        traj_names.append(traj_name)

    #Truncating equilibration frames
    for i in range(len(trajectories)):
        print(len(trajectories[i]))
        equilib_end_frame = int(  equilibration_fraction * len(trajectories[i]) ) 
        trajectories[i] = trajectories[i][equilib_end_frame:]


    #Making ref calc dirs for randomly sampled frames
    for i , traj in enumerate(trajectories):
        print(len(traj))
        num_frames = len(traj)
        frame_indices_to_sample = np.random.choice(num_frames, size=num_frames_to_sample_per_traj, replace=False)
        for frame_index in frame_indices_to_sample:
            frame = traj[frame_index]
            frame_name = f"{traj_names[i]}_{frame_index}"
            if not os.path.exists(training_frames_dir +'/'+frame_name):
                os.makedirs(training_frames_dir +'/'+frame_name)
           
            print(f"  Extracted frame: {frame_name}")

            #Writing cp2k input file
            #NOTE: We assume that the pseudopotentials and basis sets are in the /training_frames_dir/data directory directory
            cell = np.array( frame.get_cell() )

            #cut cell so that distnace between water and substrate is 15A 
            #Move atoms to bottom of the cell
            z_vals = frame.get_positions()[:,2]
            max_z = np.max(z_vals)
            min_z = np.min(z_vals)
            height = cell[2][2] 
            excess_height = height - (max_z - min_z)- 15
            new_z_vals = z_vals - min_z
            frame.positions[:,2] = new_z_vals          
            new_cell = cell
            new_cell[2][2] = height - excess_height
            frame.set_cell(new_cell)
            print('cell: ', frame.cell)
            frame_dir=training_frames_dir +'/' + frame_name +'/'
            ase.io.write(frame_dir + f"{frame_name}.xyz",frame,format='xyz')
    
            if system_name not in smearing_dir.keys():
                smearing=False
            else:
                smearing=smearing_dir[system_name]
            print('Smearing: ',smearing)

            #makes input/submission files

            


            # cp2k_input_file_tools.create_cp2k_input_file(frame,
            #                                         frame_dir+f'{frame_name}.inp',
            #                                         f"{frame_name}.xyz", 
            #                                         path_to_default_cp2k_input_file,
            #                                         config_file_path='../../config_files', 
            #                                         project_name=frame_name, 
            #                                         wave_cutoff=1200,
            #                                         smearing = smearing)
            
            if cluster=='ARCHER2':
                cp2k_input_file_tools.create_cp2k_ARCHER2_slurm_file(frame_dir+'run_dft.slurm',
                                                                0.25,
                                                                f"{frame_name}",
                                                                '/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/DFT_run_ARCHER2.slurm',
                                                                qos='short',
                                                                mail=True,
                                                                n_tasks=128,
                                                                cpus_per_task=1,
                                                                num_nodes=1,
                                                                partition='standard',
                                                                budget_allocation='e05-surfin-mic')
                    
            if cluster=='CSD3':
                cp2k_input_file_tools.create_cp2k_CSD3_slurm_file(frame_dir+'run_dft.slurm',
                                                             0.25,
                                                             f"{frame_name}",
                                                             '/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/run_simulation_CSD3.slurm',
                                                             mail=True,
                                                             n_tasks=76,
                                                             num_nodes=1,
                                                             partition='icelake',
                                                             budget_allocation='MICHAELIDES-SL2-CPU')
                