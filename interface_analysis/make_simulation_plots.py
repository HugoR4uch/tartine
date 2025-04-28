from ase.visualize.plot import plot_atoms
import ase.io
import ase
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/interface_analysis')
import interface_analysis_tools
import os
import numpy as np


simulation_runs_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/oxide_interfaces/simulation_runs'
T_target = 400
equilib_end_frame = 4000
thermodynamics_figures_dir = './thermodynamics_figures'
default_num_layers = 4
density_vs_z_figures_dir = './density_vs_z_figures'
contact_layer_density_profiles_dir = './contact_layer_xy_density_profiles'
system_sampling_region_z_values = {
    'Au_111': (0, 5.0),
    # Add more systems as needed
}
snapshot_dir = './snapshots'


min_distance_between_z_density_peaks=1
min_prominence_of_z_density_peaks=0.5



n_layers_dict = {'Al_111': 4,
                 'Al2O3_0001': 6,
                 'Al2O3-H_0001': 7,
                 'CaF2_111': 9,
                 'Fe2O3_0001': 7,
                 'Fe2O3_0001': 7,
                 'Fe2O3-H_0001': 8,
                 'kaolinite-Al_':6,
                 'kaolinite-Si_':6,
                 'Mg_0001':4,
                 'MgO_001':4,                    
                 'Pd_100': 4,
                 'Pt-non-orth_111':4 ,
                 'r-TiO2_110':6,
                 'Ru_0001': 4,
                 'SiO2_0001':6,
                 'SiO2-H_0001':7,
                 'Ti_0001_1':4,
                 }





print('Starting Analysis')
print('')



def get_sim_details_from_dir_name(dir_name):
    # This is specific to the convention I have used
    parts = dir_name.split('_')  
    material = parts[0]
    miller_index = parts[1]
    traj_index = parts[2]
    return material, miller_index,traj_index

# Getting Simulation details from directory names
simulation_details = {}
for dir_name in os.listdir(simulation_runs_dir):
    if os.path.isdir(os.path.join(simulation_runs_dir, dir_name)):
        details = get_sim_details_from_dir_name(dir_name)
        simulation_details[dir_name] = list(details)

# Grouping simulation details by (material + miller_index)
grouped_simulation_details = {}
for dir_name, details in simulation_details.items():
    material, miller_index, traj_index = details
    key = material+f'_{miller_index}'
    if key not in grouped_simulation_details:
        grouped_simulation_details[key] = []
    grouped_simulation_details[key].append(dir_name)

# Loop over all unique systems 
for key, dirs in grouped_simulation_details.items():
    system_name = f"{key}"

 
    #Getting name of system
    name = f"{key}"
    material = key.split('_')[0]
    miller_index = key.split('_')[1]   
    print('')
    print('Analysing System: ',name)
    print('Miller Index: ',miller_index)
    print('Material: ',material)
    print('')
    for dir_name in dirs[:3]:  # Assuming there are at least 3 trajectories
        print(f"  Trajectory: {dir_name}")

    
    #Loading trajectories, substrates, logfiles
    substrates = []
    trajectories = []
    logfile_paths = []
    
    for i,traj_name in enumerate(dirs):
        traj_dir = simulation_runs_dir + '/' + traj_name

        substrate_filename = '_'.join(traj_name.split('_')[:2])
        substrate = ase.io.read(traj_dir +'/' + substrate_filename + '.pdb')
        substrates.append(substrate)

        traj = ase.io.read(traj_dir + '/' + traj_name + '.xyz',index=':')    
        trajectories.append(traj)

        logfile_path = traj_dir + '/' + traj_name + '_md.log'
        logfile_paths.append(logfile_path)



    # Checking that number of atoms in substrates are the same
    for sub in substrates:
        if len(substrates[0]) != len(sub):
            print(f"Skipping system {key} due to inconsistent number of atoms in substrates")
            continue

    # Checking that number of atoms in trajectories are the same
    for traj in trajectories:
        if len(trajectories[0]) != len(traj):
            print(f"Skipping system {key} due to inconsistent number of frames in trajectories")
            continue

   
    for traj in trajectories:
        print('Number of frames before removing equilibration frames: ',len(traj))

    # Truncating equilibration frames
    for i in range(len(trajectories)):
        trajectories[i] = trajectories[i][equilib_end_frame:]

    for traj in trajectories:
        print('Number of equilibrated frames for sampling: ',len(traj))


    #Finding the z-density profile


    #Snapshot of interface
    interface_analysis_tools.make_interface_snapshot(name,trajectories[0][-1],substrate,snapshot_dir=snapshot_dir)
    

    #Plotting Thermodynamics
    print('Analysing Thermodynamics')
    Temp_mean_vals,Temp_std_vals = interface_analysis_tools.make_thermodynamics_plot(name,logfile_paths,T_target,equilib_end_frame=equilib_end_frame,end_frame=-1,figures_dir=thermodynamics_figures_dir)
    print('Temp_mean_vals: ',Temp_mean_vals)
    print('Temp_std_vals: ',Temp_std_vals)


    #Plotting Density vs z profile
    print('Analysing density vs z profile')
    
    #Finding num layers 
    #(takes top 1/num layers of substrate atoms as interface)
    if name in n_layers_dict:
        num_layers = n_layers_dict[name]
        print('Num layers found for ',name,': ',num_layers)
    else:
        print('No num_layers found for ',name,'. Using default value of ',default_num_layers)
        num_layers = default_num_layers
    
    bin_centers, average_density, errors  = interface_analysis_tools.get_z_density_profile(trajectories,substrates[0],z_min=-1,z_max=20, plot_all_profiles=False, num_layers = num_layers)
    
    
    
    bin_centers, all_densities  = interface_analysis_tools.get_z_density_profile(trajectories,substrates[0],z_min=-1,z_max=20, plot_all_profiles=True, num_layers = num_layers)
    
    max_density = np.max(average_density)
    plot_density_max = max_density * 1.2 #20% above max density shown

    fig,ax = plt.subplots( nrows=1, ncols=2, figsize=(10, 5), sharey=True, sharex=True)

    #Plot all density profiles
    for i,density in enumerate(all_densities):
        ax[0].plot(bin_centers,density,label = 'Trajectory '+str(i+1))
    ax[0].set_xlabel(r'Distance From Interface z [$\AA$]')
    ax[0].set_ylabel(r'Density [$gcm^{-3}$]')
    ax[0].set_ylim([0,plot_density_max])  
    ax[0].grid()
    ax[0].legend()

    #Plot mean density
    ax[1].errorbar(bin_centers,average_density,yerr = errors)
    ax[1].set_title(name+' Water Density Profile')
    ax[1].set_ylim([0,plot_density_max])  
    ax[1].set_xlabel(r'Distance From Interface z [$\AA$]')
    ax[1].set_ylabel(r'Density [$gcm^{-3}$]')
    ax[1].grid()


    #Creating directory to save snapshots
    if not os.path.exists(density_vs_z_figures_dir):
        os.makedirs(density_vs_z_figures_dir)
    

    plt.savefig(density_vs_z_figures_dir + '/' + name + '_density_vs_z.png')
    plt.close()





    #Finding z-coords of contact layer
    z_min_contact,z_max_contact=interface_analysis_tools.get_z_density_turning_points(np.array(bin_centers), np.array(average_density), distance=1, prominence=0.5)


    if z_max_contact is None:
        print('No Peaks Found')
        continue # Skip to next system
    else:
        print('Contact layer starts at z=',z_min_contact,' and ends at z=',z_max_contact)


    #Plotting 2D snapshot of interface
    
    L=15


    # Creating directory to save contact layer density profiles
    if not os.path.exists(contact_layer_density_profiles_dir):
        os.makedirs(contact_layer_density_profiles_dir)

    # Finding O atom density
    bin_centers_x, bin_centers_y, o_density, o_errors = interface_analysis_tools.get_xy_density_profile(
        trajectories, species='O', z_min=z_min_contact, z_max=z_max_contact, L=L, substrate=substrates[0], return_all_profiles=False, num_layers=4)

    # Finding H atom density
    bin_centers_x, bin_centers_y, h_density, h_errors = interface_analysis_tools.get_xy_density_profile(
        trajectories, species='H', z_min=z_min_contact, z_max=z_max_contact, L=L, substrate=substrates[0], return_all_profiles=False, num_layers=4)

    # Plotting heatmap of average density 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    xticks = np.linspace(-L/2, L/2, num=10)
    yticks = np.linspace(-L/2, L/2, num=10)

    # Heatmap of H atom density
    cax1 = ax[0, 0].imshow(h_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', aspect='auto')
    ax[0, 0].set_title('H Atom Density')
    ax[0, 0].set_xlabel('X [Å]')
    ax[0, 0].set_ylabel('Y [Å]')
    ax[0, 0].grid(True)
    ax[0, 0].set_xticks(xticks)
    ax[0, 0].set_yticks(yticks)
    #fig.colorbar(cax1, ax=ax[0, 0], label='Density [atoms/A^3]')

    # Heatmap of O atom density
    cax2 = ax[0, 1].imshow(o_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', aspect='auto')
    ax[0, 1].set_title('O Atom Density')
    ax[0, 1].set_xlabel('X [Å]')
    ax[0, 1].set_ylabel('Y [Å]')
    ax[0, 1].grid(True)
    ax[0, 1].set_xticks(xticks)
    ax[0, 1].set_yticks(yticks)
    #fig.colorbar(cax2, ax=ax[0, 1], label='Density [atoms/A^3]')

    # Plotting z-density profile to show contact layer
    ax[1, 0].errorbar(bin_centers, average_density, yerr=errors)
    ax[1, 0].set_title(name + ' Water Density Profile')
    ax[1, 0].set_ylim([0, plot_density_max])
    ax[1, 0].set_xlabel(r'Distance From Interface z [$\AA$]')
    ax[1, 0].set_ylabel(r'Density [$gcm^{-3}$]')
    ax[1, 0].grid()
    ax[1, 0].plot([z_min_contact, z_min_contact], [0, plot_density_max], 'k--', label=f'Contact Layer Start: {z_min_contact:.1f} Å')
    ax[1, 0].plot([z_max_contact, z_max_contact], [0, plot_density_max], 'r--', label=f'Contact Layer End: {z_max_contact:.1f} Å')
    ax[1, 0].legend()

    # Plots of substrate top layer positions
    for i, traj in enumerate(trajectories):
        x_values, y_values = interface_analysis_tools.get_xy_trajectory(traj, substrate=substrates[0], L=15, stride=10, num_layers=4)
        ax[1, 1].scatter(x_values, y_values, label=f'Trajectory {i+1}')
    ax[1, 1].set_xlim([-L/2, L/2])
    ax[1, 1].set_ylim([-L/2, L/2])
    ax[1, 1].set_xlabel('X [Å]')
    ax[1, 1].set_ylabel('Y [Å]')
    ax[1, 1].set_title('Top Layer Substrate Atom Positions')
    ax[1, 1].grid(True)
    ax[1, 1].set_xticks(xticks)
    ax[1, 1].set_yticks(yticks)
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{contact_layer_density_profiles_dir}/{system_name}_contact_layer_xy_density.png')
    plt.close()

