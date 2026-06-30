from ase.visualize.plot import plot_atoms
import ase.io
import ase
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
from tartines.interface_analysis import interface_analysis_tools
from tartines.interface_analysis import water_analyser
import os
import time
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import json


def convert_numpy_types_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars that have .item() method
        return obj.item()
    elif hasattr(obj, 'tolist'):  # Handle any numpy-like objects with tolist method
        return obj.tolist()
    else:
        return obj


def extract_simulation_data(simulation_runs_dir,
                            name,
                            dirs,
                            equilib_end_frame,
                            ):
                           
    
    #Loading trajectories, substrates, logfiles
    substrates = []
    trajectories = []
    logfile_paths = []
    traj_indices = []
    

    for i,traj_name in enumerate(dirs):
        traj_dir = simulation_runs_dir + '/' + traj_name
        traj_index = int(traj_name.split('_')[-1])
        traj_indices.append(traj_index)
        substrate_filename = '_'.join(traj_name.split('_')[:2])
        print('Loading substrate: ', substrate_filename + '.pdb')
        substrate = ase.io.read(traj_dir +'/' + substrate_filename + '.pdb',format='proteindatabank')
        
        substrates.append(substrate)

        traj_path = traj_dir + '/' + traj_name + '.xyz'
        print('Loading trajectory: ', traj_dir)

        if os.path.isfile(traj_path):
            traj = ase.io.read(traj_dir + '/' + traj_name + '.xyz',index=':')    
        else:
            print(traj_path,' does not exist! Skipping this trajectory.')
            continue
        
        trajectories.append(traj)
        print('Substrate and trajectory successfully loaded')

        print('Number of trajectories: ',len(trajectories))
        print('Number of substrates: ',len(substrates))

        logfile_path = traj_dir + '/' + traj_name + '_md.log'
        logfile_paths.append(logfile_path)


    # Checking length/size of trajectories / substrates are the same (detects explosions)
    for sub in substrates:
        if len(substrates[0]) != len(sub):
            print(f"Skipping system {name} due to inconsistent number of atoms in substrates")
            continue

    for traj in trajectories:
        if len(trajectories[0]) != len(traj):
            print(f"Skipping system {name} due to inconsistent number of frames in trajectories")
            continue

    # Equilibration 
    for traj in trajectories:
        print('Number of frames before removing equilibration frames: ',len(traj))
        
    for i in range(len(trajectories)):
        trajectories[i] = trajectories[i][equilib_end_frame:]

    for traj in trajectories:
        print('Number of equilibrated frames for sampling: ',len(traj))


    return substrates, trajectories, logfile_paths, traj_indices



def find_cumul_temp(Temperatures):
    cumul_temp = []
    for i,temp in enumerate(Temperatures):
        cumul_temp.append(np.mean(Temperatures[:i]))
    return cumul_temp



def make_thermodynamics_plot(name,logfile_paths,T_target,equilib_end_frame=0,end_frame=-1,figures_dir='.'):
    #Creating directory to save snapshots
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    num_traj = len(logfile_paths)

    #Plotting Snapshot
    

    #Temp and Energy Stats Initialize
    Temp_mean_vals =[]
    Energy_std_vals = []
    Temp_std_vals = []

    if len(logfile_paths) == 1:
        fig, ax = plt.subplots(1,2,figsize=(12, 4*num_traj),sharex=True,sharey=False)
        logfile = logfile_paths[0]
        data = np.loadtxt(logfile,skiprows=1)
        #Giving Title to the first row
        ax[0].set_title('Energy vs Time')
        ax[1].set_title('Temperature vs Time')
        #Getting data
        Temp = data[:,4][equilib_end_frame:end_frame]
        Energy = data[:,1][equilib_end_frame:end_frame]
        time = data[:,0][equilib_end_frame:end_frame]
        #Getting statistics
        temp_mean_minus_target = np.mean(Temp) - T_target
        temp_std = np.std(Temp)
        Temp_mean_vals.append(np.mean(Temp))
        Temp_std_vals.append(temp_std)
        cumul_temp = find_cumul_temp(Temp)
        #Plotting E vs t
        ax[0].plot(time,Energy,label = logfile.split('/')[-1].split('.')[0])
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('Energy/Atom (eV)')
        ax[0].set_ylim([np.mean(Energy)-0.025,np.mean(Energy)+0.025])
        ax[0].grid()
        #Plotting T vs t
        ax[1].plot(time,Temp,label = logfile.split('/')[-1].split('.')[0])
        ax[1].plot(time,cumul_temp,label = 'cum-Temp')
        ax[1].legend(loc='upper left')
        ax[1].set_ylabel('Temperature (K)')
        ax[1].set_xlabel('Time (ps)')
        ax[1].grid()
        T_targets = np.ones(len(Temp))*T_target
        ax[1].plot(time,T_targets,label = 'Target Temperature',linestyle = '--')
        # Add textbox
        textstr = f'Mean - Target: {temp_mean_minus_target:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    else:
        fig, ax = plt.subplots(num_traj,2,figsize=(8, 8),sharex=True,sharey=False)
        for i,logfile in enumerate( logfile_paths ):
            data = np.loadtxt(logfile,skiprows=1)
            if len(logfile_paths) == 1:
                ax[i] = ax[i]
            if i ==0:
                ax[i][0].set_title('Energy vs Time')
                ax[i][1].set_title('Temperature vs Time')
            Temp = data[:,4][equilib_end_frame:end_frame]
            Energy = data[:,1][equilib_end_frame:end_frame]
            time = data[:,0][equilib_end_frame:end_frame]
            temp_mean_minus_target = np.mean(Temp) - T_target
            print(type(temp_mean_minus_target))
            temp_std = np.std(Temp)
            Temp_mean_vals.append(np.mean(Temp))
            Temp_std_vals.append(temp_std)
            cumul_temp = find_cumul_temp(Temp)
            ax[i][0].plot(time,Energy,label = logfile.split('/')[-1].split('.')[0])
            ax[i][0].set_xlabel('Time (ps)')
            ax[i][0].set_ylabel('Energy/Atom (eV)')
            ax[i][0].set_ylim([np.mean(Energy)-0.025,np.mean(Energy)+0.025])
            ax[i][0].grid()
            ax[i][1].plot(time,Temp,label = logfile.split('/')[-1].split('.')[0])
            ax[i][1].plot(time,cumul_temp,label = 'cum-Temp')
            ax[i][1].legend(loc='upper left')
            ax[i][1].set_ylabel('Temperature (K)')
            ax[i][1].set_xlabel('Time (ps)')
            ax[i][1].grid()
            T_targets = np.ones(len(Temp))*T_target
            ax[i][1].plot(time,T_targets,label = 'Target Temperature',linestyle = '--')
            textstr = f'Mean - Target: {temp_mean_minus_target:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax[i][1].text(0.5, 0.15, textstr, transform=ax[i][1].transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)
    plt.tight_layout()
    plt.savefig(figures_dir+'/'+name+'_thermodynamics.png')
    plt.close()
        
    return Temp_mean_vals,Temp_std_vals







def get_sim_details_from_dir_name(dir_name):
    """Extracts material, miller index and traj index from directory name."""

    # This is specific to the convention I have used
    print('Extracting simulation details from directory name:', dir_name)
    parts = dir_name.split('_')  
    material = parts[0]
    miller_index = parts[1]
    traj_index = parts[2]
    return material, miller_index,traj_index


def get_grouped_simulation_details(simulation_runs_dir):
    """Takes path to dir containing the data for MD sims of different systems. 
    Returns dict with <material>_<miller_index> as key and list of directories with MD data for those systems
    as value.
    """

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


    return grouped_simulation_details





def make_interface_snapshot(name,interface,substrate,snapshot_dir='.'):
    from ase.visualize.plot import plot_atoms

    #Creating directory to save snapshots
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    #Plotting Snapshot
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))  
    plot_atoms(interface, ax=axes[0], rotation=('-90x,0y,0z'))
    plot_atoms(substrate, ax=axes[1], rotation=('0x,0y,0z'))

    #Adding Title
    plt.tight_layout() 
    plt.title(name+ ' interface')
    plt.savefig(snapshot_dir+'/'+name+'.png')
    plt.close()





def plot_density_vs_AIMD(O_bin_centers,
                         O_average_density,
                         name,
                         current_model_name,
                         AIMD_density_profiles,
                         density_vs_aimd_figures_dir='.',
                         plot_name=None,
                         z_plot_max=None):

    if plot_name is None:
        plot_name = name

    
    max_density = np.max(O_average_density) 
    plt.ylim([0, max_density * 1.2])  
    plt.xlabel(r'Distance From Interface z [$\AA$]')
    plt.ylabel(r'Density [$gcm^{-3}$]')
    plt.grid()
    plt.title(plot_name + ' Water Density Profile')

    for density_profile_file in AIMD_density_profiles[name]:
        profile_data = density_profile_file.split('/')[-1].split('.')[0]
        print('Extracting data from:', profile_data)
        temp = profile_data.split('_')[1]
        XC_functional = profile_data.split('_')[2]
        Year = profile_data.split('_')[3]
        author = profile_data.split('_')[4]
        profile_name = f'{temp}K {XC_functional}'

        data = np.loadtxt(density_profile_file, delimiter=',', skiprows=1)
        z = data[:, 0]
        density = data[:, 1]
        plt.plot(z, density, label=profile_name)
        new_max_density = np.max(density)
        if new_max_density > max_density:
            max_density = new_max_density



    plt.plot(O_bin_centers,O_average_density,linestyle='--',color='black', linewidth=2,label = current_model_name)

    plt.legend()
    plt.xlim([0,z_plot_max]) if z_plot_max is not None else plt.xlim([0, 20])
    plt.ylim([0,max_density*1.2])
    plt.tight_layout()

    plt.savefig(density_vs_aimd_figures_dir + '/' + name + '_density_vs_z.png',dpi=600)

    plt.close()


def plot_all_trajectory_density_profiles(name,
                                         trajectories,
                                         substrate,
                                         z_min=-1,
                                         z_max=30,
                                         num_layers=None,
                                         all_traj_density_figures_dir='./all_traj_density_figures',
                                         species='O',
                                         plot_name=None):
    """
    Plot density profiles for each trajectory independently in subplots with shared y-axis.
    
    Parameters
    ----------
    name : str
        System identifier
    trajectories : list
        List of ASE trajectory objects
    substrate : ase.Atoms
        Substrate structure
    z_min : float
        Minimum z-coordinate for plotting
    z_max : float
        Maximum z-coordinate for plotting
    num_layers : int, optional
        Number of substrate layers to consider
    all_traj_density_figures_dir : str
        Directory to save figures
    species : str
        Atom species to analyze ('O' or 'H')
    plot_name : str, optional
        Display name for plots
    """
    
    if not os.path.exists(all_traj_density_figures_dir):
        os.makedirs(all_traj_density_figures_dir)
    
    if plot_name is None:
        plot_name = name
    
    # Get individual trajectory density profiles
    bin_centers, all_densities = interface_analysis_tools.get_z_density_profile(
        trajectories,
        substrate,
        z_min=z_min,
        z_max=z_max,
        plot_all_profiles=True,
        num_layers=num_layers,
        species=species
    )
    
    n_trajectories = len(all_densities)
    max_density = np.max(all_densities) * 1.2
    
    # Create subplots
    n_cols = 3
    n_rows = (n_trajectories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), sharey=True)
    axes = axes.flatten()
    
    # Plot each trajectory
    for traj_idx, density in enumerate(all_densities):
        ax = axes[traj_idx]
        ax.plot(bin_centers, density, linewidth=2, color='steelblue')
        ax.fill_between(bin_centers, density, alpha=0.3, color='steelblue')
        ax.set_xlabel(r'Distance From Interface z [$\AA$]')
        ax.set_ylabel(r'Density [$gcm^{-3}$]')
        ax.set_title(f'Trajectory {traj_idx + 1}', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([z_min, z_max])
        ax.set_ylim([0, max_density])
    
    # Hide unused subplots
    for idx in range(n_trajectories, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'{plot_name} - Individual Trajectory {species} Density Profiles', fontsize=20)
    plt.tight_layout()
    
    filename = os.path.join(all_traj_density_figures_dir, f'{name}_all_trajectories_density_vs_z.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Saved: {filename}')





def multi_gen_density_plot(
                    current_O_bin_centers,
                    current_O_average_density,
                    name,
                    name_of_this_gen,
                    older_gen_sim_dirs,
                    multi_gen_density_figures_dir= './multi_gen_density_figures',
                    equilib_end_frame = 4000,
                    plot_name=None,
                    z_plot_max=None,
                    plot_legend=True,
                    ):
    
    if plot_name is None:
        plot_name = name


    for gen_name in older_gen_sim_dirs.keys():
        if gen_name == name_of_this_gen:
            continue

        gen_sim_dir = older_gen_sim_dirs[gen_name]
        if not os.path.exists(gen_sim_dir):
            print('The',gen_name, 'does not exist:'+gen_sim_dir)
            return

        sim_dirs = os.listdir(gen_sim_dir)

        simulation_in_gen = False
        for dir in sim_dirs:
            if name in dir:
                simulation_in_gen = True
                break
        if not simulation_in_gen:
            print(f'The system ',name, f'does not exist for gen {gen_name} in:',gen_sim_dir)
            print(f'Skipping this generation: {gen_name}')
            continue


        system_sim_dirs=[dir for dir in sim_dirs if name in dir]

                

        substrates, trajectories,_, _ = extract_simulation_data(
                                                    gen_sim_dir,
                                                    name,
                                                    system_sim_dirs, # note, dirs for system only
                                                    equilib_end_frame,
                                                    )
        
        print(f'Number of substrates for gen {gen_name}:', len(substrates))
        print('Substrates   :', substrates)
        print(f'Number of trajectories for gen {gen_name}:', len(trajectories))

        O_data = interface_analysis_tools.get_z_density_profile(trajectories,
                                                                substrates[0],
                                                                z_min=-1,
                                                                z_max=30,
                                                                plot_all_profiles=False,
                                                                num_layers = None)

        O_bin_centers, O_average_density, _ = O_data


        

        
        plt.plot(O_bin_centers,O_average_density,label = gen_name)


    
    plt.plot(current_O_bin_centers,current_O_average_density,label = name_of_this_gen)
    plt.xlim([0, 20]) if z_plot_max is None else plt.xlim([0, z_plot_max])
    if plot_legend:
        plt.legend()
    plt.grid()
    plt.xlabel(r'Distance From Interface z [$\AA$]')
    plt.ylabel(r'Water Density [$gcm^{-3}$]')
    plt.title(plot_name + ' Density Profile vs Gen')
    plt.tight_layout()
    plt.savefig(multi_gen_density_figures_dir + '/' + name + 'multi_gen_density_vs_z.png',dpi=600)
    plt.close()




def plot_default_density_vs_z(name,
                                O_bin_centers,
                                O_average_density,
                                O_errors,
                                H_bin_centers,
                                H_average_density,
                                H_errors,
                                density_vs_z_figures_dir='./',
                                density_ylim=None
                                ):

    if not os.path.exists(density_vs_z_figures_dir):
        os.makedirs(density_vs_z_figures_dir)



    O_max_density = np.max(O_average_density)
    H_max_density = np.max(H_average_density)

    maximum_density = max(O_max_density, H_max_density)

    # Plotting density and contact layer info 

    fig, ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12, 6))


    ax[0].errorbar(O_bin_centers,O_average_density,yerr = O_errors)
    ax[0].set_title('O Density')

    
    ax[1].errorbar(H_bin_centers,H_average_density,yerr = H_errors)
    ax[1].set_title('H Density')

    fig.suptitle(name + ' Water Density Profile', fontsize=16)

    plot_ylim = density_ylim if density_ylim is not None else maximum_density * 1.2

    for a in ax:
        a.set_ylim([0, plot_ylim])
        a.set_xlim([0, 25])
        a.set_xlabel(r'Distance From Interface z [$\AA$]')
        a.set_ylabel(r'Density [$gcm^{-3}$]')
        a.grid()
        
        
    fig.tight_layout()

    ax[1].legend()

    plt.savefig(density_vs_z_figures_dir + '/' + name + '_density_vs_z.png')

    plt.close()




def plot_contact_layer_z_density(name,
                                 O_bin_centers,
                                 O_average_density,
                                 O_errors,
                                 H_bin_centers,
                                 H_average_density,
                                 H_errors,
                                 peak_O_z_vals, 
                                 O_contact_layer_start,
                                 O_contact_layer_end,
                                 density_vs_z_figures_dir='./',
                                 density_ylim=None
                                 ):

    if not os.path.exists(density_vs_z_figures_dir):
        os.makedirs(density_vs_z_figures_dir)


    peaks_in_contact_layer = [z for z in peak_O_z_vals if z < O_contact_layer_end]
    contact_layer_z_mask = [ O_contact_layer_start <= z <= O_contact_layer_end for z in O_bin_centers]
    contact_layer_mean_density = np.mean(np.array(O_average_density)[contact_layer_z_mask]) 
    

    O_max_density = np.max(O_average_density)
    H_max_density = np.max(H_average_density)

    maximum_density = max(O_max_density, H_max_density)
    plot_ylim = density_ylim if density_ylim is not None else maximum_density * 1.2

    # Plotting density and contact layer info 

    fig, ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12, 6))


    ax[0].errorbar(O_bin_centers,O_average_density,yerr = O_errors)
    ax[0].set_title('O Density')
    ax[0].text(
        0.95, 0.95,  # top-right corner in axes fraction
        f'Contact Layer Mean Density: {contact_layer_mean_density:.1f} $gcm^{{-3}}$',
        transform=ax[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',  # anchor text to the right
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax[1].errorbar(H_bin_centers,H_average_density,yerr = H_errors)
    ax[1].set_title('H Density')

    fig.suptitle(name + ' Water Density Profile', fontsize=16)

    for a in ax:

        a.set_ylim([0, plot_ylim])
        a.set_xlim([0, 10])

        for i, peak_z in enumerate(peaks_in_contact_layer):
            a.plot([peak_z, peak_z], [0, plot_ylim], 'r--', label=f'Water O Peak {i+1}: {peak_z:.1f} Å')

        a.plot([O_contact_layer_start, O_contact_layer_start], [0, plot_ylim], 'k--', label=f'Contact Layer Start: {O_contact_layer_start:.1f} Å')
        a.plot([O_contact_layer_end, O_contact_layer_end], [0, plot_ylim], 'k--', label=f'Contact Layer End: {O_contact_layer_end:.1f} Å')
        a.set_xlabel(r'Distance From Interface z [$\AA$]')
        a.set_ylabel(r'Density [$gcm^{-3}$]')
        
        a.grid()
        
    fig.tight_layout()

    ax[1].legend()

    plt.savefig(density_vs_z_figures_dir + '/' + name + '_density_vs_z.png')

    plt.close()


def plot_species_resolved_density_profiles(
    name,
    profiles,
    figures_dir="./",
    plot_name=None,
    z_plot_max=10,
    plot_species_sum=True,
    O_contact_layer_start=None,
    O_contact_layer_end=None,
    species_z_bounds=None,
    partition_name=None,
    density_ylim=None,
):
    """Plot default and custom Euler-species resolved rho(z) profiles."""

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    if plot_name is None:
        plot_name = name

    plt.figure(figsize=(8, 6))

    all_profile = profiles["all"]
    plt.plot(
        all_profile["z"],
        all_profile["density"],
        color="black",
        linewidth=2.5,
        label="All water O",
    )

    max_density = np.max(all_profile["density"])

    for species_name, species_profile in profiles["species"].items():
        plt.plot(
            species_profile["z"],
            species_profile["density"],
            linewidth=2,
            label=species_name,
        )
        species_max = np.max(species_profile["density"])
        if species_max > max_density:
            max_density = species_max

    if plot_species_sum and "species_sum" in profiles:
        species_sum = profiles["species_sum"]
        plt.plot(
            species_sum["z"],
            species_sum["density"],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Species sum",
        )
        sum_max = np.max(species_sum["density"])
        if sum_max > max_density:
            max_density = sum_max

    if O_contact_layer_start is not None:
        plt.axvline(
            O_contact_layer_start,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label=f"Contact start: {O_contact_layer_start:.2f} Å",
        )

    if O_contact_layer_end is not None:
        plt.axvline(
            O_contact_layer_end,
            color="grey",
            linestyle="--",
            linewidth=1.5,
            label=f"Contact end: {O_contact_layer_end:.2f} Å",
        )

    title = f"{plot_name}: species-resolved water O density"
    if partition_name is not None:
        title += f"\n{partition_name}"
    if species_z_bounds is not None:
        title += f", species z = [{species_z_bounds[0]:.2f}, {species_z_bounds[1]:.2f}] Å"

    plt.title(title)
    plt.xlabel(r"Distance From Interface z [$\AA$]")
    plt.ylabel(r"Density [$gcm^{-3}$]")
    plt.xlim([0, z_plot_max])
    plt.ylim([0, density_ylim if density_ylim is not None else max_density * 1.2])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(
        figures_dir,
        f"{name}_species_resolved_density_vs_z.png",
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_contact_layer_xy_free_energy(
    x_bins,
    y_bins,
    free_energy_xy,
    filename=None,
    title="Contact-layer F(u, v)",
    plot_name=None,
    z_bounds=None,
    reference_z_bounds=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    primitive_cell=None,
    origin_xy=None,
    plot_cartesian=False,
):
    """
    Plot a contact-layer in-plane free-energy map.

    Parameters
    ----------
    x_bins, y_bins : array-like
        Histogram bin edges. For current F(u,v) data these are primitive
        fractional coordinates u and v.
    free_energy_xy : array-like or masked array
        Shifted F(u, v) in eV with shape (n_u_bins, n_v_bins).
    filename : str, optional
        Save path. If omitted, the figure is shown but not saved.
    """

    F_xy = np.ma.masked_invalid(np.ma.asarray(free_energy_xy, dtype=float))

    finite_values = F_xy.compressed()
    if finite_values.size == 0:
        raise ValueError("free_energy_xy contains no finite values to plot")

    if vmin is None:
        vmin = float(np.nanpercentile(finite_values, 5))

    if vmax is None:
        vmax = float(np.nanpercentile(finite_values, 95))
    
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(finite_values))
        vmax = float(np.nanmax(finite_values))

    if plot_name is not None:
        title = f"{plot_name}: contact-layer F(u, v)"

    subtitle_parts = []
    if z_bounds is not None:
        subtitle_parts.append(f"contact z = {z_bounds[0]:.2f}-{z_bounds[1]:.2f} Å")
    if reference_z_bounds is not None:
        subtitle_parts.append(f"reference z = {reference_z_bounds[0]:.2f}-{reference_z_bounds[1]:.2f} Å")
    if subtitle_parts:
        title = title + "\n" + ", ".join(subtitle_parts)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    if plot_cartesian:
        if primitive_cell is None:
            raise ValueError("primitive_cell must be provided when plot_cartesian=True")

        u_grid, v_grid = np.meshgrid(x_bins, y_bins, indexing="ij")
        edge_frac = np.stack([u_grid, v_grid], axis=-1)
        edge_xy = interface_analysis_tools.primitive_fractional_to_xy(
            edge_frac.reshape(-1, 2),
            primitive_cell,
            origin_xy=origin_xy,
        ).reshape(u_grid.shape + (2,))

        mesh = ax.pcolormesh(
            edge_xy[:, :, 0],
            edge_xy[:, :, 1],
            F_xy,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(r"$x$ [$\AA$]")
        ax.set_ylabel(r"$y$ [$\AA$]")
    else:
        mesh = ax.pcolormesh(
            x_bins,
            y_bins,
            F_xy.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("u")
        ax.set_ylabel("v")

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, label=r"$F(u,v)$ [eV]")

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_top_layer_xy_motion_transform(
    trajectories,
    substrate,
    filename=None,
    stride=10,
    num_layers=None,
    tolerance=0.1,
    account_for_substrate_xy_motion=True,
    title=None,
):
    """
    Plot raw and transformed substrate top-layer xy coordinates.

    This is a diagnostic for the substrate-motion correction used by
    get_xy_free_energy_profile. The transform is fitted using only top-layer
    substrate atoms.
    """
    if len(trajectories) > 0 and hasattr(trajectories[0], "get_positions"):
        trajectories = [trajectories]

    top_layer_indices = interface_analysis_tools.find_top_layer_indices(
        substrate,
        num_layers=num_layers,
        tolerance=tolerance,
    )

    primitive_cell, origin_shift = interface_analysis_tools.get_substrate_primitive_cell_data(
        substrate,
    )
    origin_xy = origin_shift[:2]

    raw_xy = []
    transformed_xy = []
    transformed_uv = []

    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue

        reference_top_layer_xy = trajectory[0].positions[top_layer_indices, :2]

        for frame in trajectory[::stride]:
            raw_positions = frame.positions[top_layer_indices]
            raw_xy.append(raw_positions[:, :2])

            if account_for_substrate_xy_motion:
                transformed_positions = interface_analysis_tools._align_sample_positions_to_top_layer_xy(
                    frame,
                    top_layer_indices,
                    top_layer_indices,
                    reference_top_layer_xy,
                )
            else:
                transformed_positions = raw_positions.copy()

            transformed_xy.append(transformed_positions[:, :2])
            transformed_frac = interface_analysis_tools.xy_to_primitive_fractional(
                transformed_positions[:, :2],
                primitive_cell,
                origin_xy=origin_xy,
            )
            transformed_uv.append(transformed_frac)

    if len(raw_xy) == 0:
        raise ValueError("No trajectory frames available for top-layer xy diagnostic")

    raw_xy = np.vstack(raw_xy)
    transformed_xy = np.vstack(transformed_xy)
    transformed_uv = np.vstack(transformed_uv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].scatter(raw_xy[:, 0], raw_xy[:, 1], s=8, alpha=0.25, edgecolors="none")
    axes[1].scatter(
        transformed_xy[:, 0],
        transformed_xy[:, 1],
        s=8,
        alpha=0.25,
        edgecolors="none",
    )
    axes[2].scatter(
        transformed_uv[:, 0],
        transformed_uv[:, 1],
        s=8,
        alpha=0.25,
        edgecolors="none",
    )

    axes[0].set_title("Raw top-layer xy")
    axes[1].set_title("Aligned top-layer xy")
    axes[2].set_title("Aligned top-layer unwrapped uv")

    all_xy = np.vstack([raw_xy, transformed_xy])
    x_pad = 0.05 * max(np.ptp(all_xy[:, 0]), 1.0)
    y_pad = 0.05 * max(np.ptp(all_xy[:, 1]), 1.0)
    xlim = (float(np.min(all_xy[:, 0]) - x_pad), float(np.max(all_xy[:, 0]) + x_pad))
    ylim = (float(np.min(all_xy[:, 1]) - y_pad), float(np.max(all_xy[:, 1]) + y_pad))

    for ax in axes[:2]:
        ax.set_xlabel(r"$x$ [$\AA$]")
        ax.set_ylabel(r"$y$ [$\AA$]")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    axes[2].set_xlabel("u")
    axes[2].set_ylabel("v")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(alpha=0.25)

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)





def get_dissociation_statistics(trajectories, substrate, z_min, z_max,sampling_interval=10,save_results=False,system_name='system',fragment_data_dir='./',trajectory_indices=None):

    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

    all_traj_H3O_populations = []
    all_traj_H2O_populations = []
    all_traj_OH_populations = []
    all_traj_substrate_H_populations = []
    all_traj_times= []

    traj_count=  0

    for traj in trajectories:
        
        times = []
        H2O_population = []
        H3O_population = []
        OH_population = []
        substrate_H_population = []

        H3O_O_atoms_dict = {}
        OH_O_atoms_dict = {}
        substrate_H_atoms_dict = {}

        frame_count = 0

        for frame in tqdm(traj[::sampling_interval], desc=f"Processing frames for dissociation statistics"):
            
            frame_count += 1
            H2O_count = 0
            H3O_count = 0
            OH_count = 0
            substrate_H_count = 0

            substrate_indices = np.arange(len(substrate))
            analyser = water_analyser.Analyser(frame, substrate_indices=substrate_indices)

            voronoi_dict = analyser.get_voronoi_dict(include_substrate=True)

            interface_indices = interface_analysis_tools.interfacial_water_criterion(frame, substrate, z_min, z_max)

            interfacial_water_O_indices = [i for i in analyser.aqua_O_indices if i in interface_indices]

            interfacial_voronoi_dict = {i: voronoi_dict[i] for i in interfacial_water_O_indices}
            

            frame_OH_indices = []
            frame_H_indices = []
            frame_O_indices = []
            frame_H3O_indices = []

            for i in interfacial_water_O_indices:
                coordination = len(interfacial_voronoi_dict[i])

                if coordination == 0:
                    frame_O_indices.append(i)
                    print('O atom {i} has no H atoms attached in frame {frame}')
                
                if coordination == 1:
                    frame_OH_indices.append(i)
                    OH_count+=1
                    if OH_O_atoms_dict.get(frame_count*sampling_interval) is None:
                        OH_O_atoms_dict[frame_count*sampling_interval] = []
                    OH_O_atoms_dict[frame_count*sampling_interval].append(i)
                    # print('OH found at frame:', frame_count)
                    # print('O index:', i)
                if coordination == 2:
                    H2O_count+=1
                if coordination == 3:
                    frame_H3O_indices.append(i)
                    H3O_count+=1
                    if H3O_O_atoms_dict.get(frame_count*sampling_interval) is None:
                        H3O_O_atoms_dict[frame_count*sampling_interval] = []
                    H3O_O_atoms_dict[frame_count*sampling_interval].append(i)
                    # print('H3O found at frame:', frame_count)
                    # print('O index:', i)

            for i in substrate_indices:
                coordination = len(voronoi_dict[i])
                if coordination == 1:
                    if substrate_H_atoms_dict.get(frame_count*sampling_interval) is None:
                        substrate_H_atoms_dict[frame_count*sampling_interval] = []
                    substrate_H_atoms_dict[frame_count*sampling_interval].append(voronoi_dict[i][0])
                    substrate_H_count += coordination
                if coordination > 1: 
                    if substrate_H_atoms_dict.get(frame_count*sampling_interval) is None:
                        substrate_H_atoms_dict[frame_count*sampling_interval] = []
                    substrate_H_count += coordination
                    substrate_H_atoms_dict[frame_count*sampling_interval].extend(voronoi_dict[i])
                    print(f"WARNING: Substrate atom {i} has {coordination} H atoms on it: {voronoi_dict[i]}")

            if OH_count != substrate_H_count:
                if substrate_H_atoms_dict.get(frame_count*sampling_interval) is None:
                        substrate_H_atoms_dict[frame_count*sampling_interval] = []
                print(f"Frame {frame_count*sampling_interval}: OH- count ({OH_count}) does not match substrate H count ({substrate_H_count}).")
                print(f"H indices: {substrate_H_atoms_dict[frame_count*sampling_interval]}")
                print(f"OH indices:{frame_OH_indices}")
                print("Interface Indices:", interface_indices)
                print()

            if substrate_H_atoms_dict.get(frame_count*sampling_interval) is not None:
                if OH_O_atoms_dict.get(frame_count*sampling_interval) is None:
                    print(
                        f"Frame {frame_count*sampling_interval}: Substrate H atoms found: {substrate_H_atoms_dict[frame_count*sampling_interval]}, but no OH- detected."
                    )


            substrate_H_population.append(substrate_H_count)
            H2O_population.append(H2O_count)
            H3O_population.append(H3O_count)
            OH_population.append(OH_count)
            times.append(frame_count * sampling_interval)  

        if save_results:
            # Save the atom dictionaries as JSON files
            traj_index = trajectory_indices[traj_count]
            
            # Convert integer keys to strings and numpy int64 values to regular ints for JSON compatibility
            H3O_dict_str_keys = {str(k): [int(atom_idx) for atom_idx in v] for k, v in H3O_O_atoms_dict.items()}
            OH_dict_str_keys = {str(k): [int(atom_idx) for atom_idx in v] for k, v in OH_O_atoms_dict.items()}
            substrate_H_dict_str_keys = {str(k): [int(atom_idx) for atom_idx in v] for k, v in substrate_H_atoms_dict.items()}
            
            # Save H3O atoms dictionary
            with open(f"{fragment_data_dir}/{system_name}_traj_{traj_index}_H3O_atoms.json", 'w') as f:
                json.dump(H3O_dict_str_keys, f, indent=2)
            
            # Save OH atoms dictionary
            with open(f"{fragment_data_dir}/{system_name}_traj_{traj_index}_OH_atoms.json", 'w') as f:
                json.dump(OH_dict_str_keys, f, indent=2)
            
            # Save substrate H atoms dictionary
            with open(f"{fragment_data_dir}/{system_name}_traj_{traj_index}_substrate_H_atoms.json", 'w') as f:
                json.dump(substrate_H_dict_str_keys, f, indent=2)
        
        # These lines should be at the trajectory level, not frame level
        all_traj_substrate_H_populations.append(substrate_H_population)
        all_traj_H2O_populations.append(H2O_population)
        all_traj_H3O_populations.append(H3O_population)
        all_traj_OH_populations.append(OH_population)
        all_traj_times.append(times)

        traj_count += 1


    return all_traj_OH_populations, all_traj_H2O_populations, all_traj_H3O_populations, all_traj_substrate_H_populations, all_traj_times





def plot_dissociation_statistics(name,
                                 trajectories,
                                 substrate, 
                                 z_min, 
                                 z_max,
                                 sampling_interval=10,
                                 dt=5,
                                 fragments_plots_dir='./fragments_plots',
                                 save_results=True,
                                 trajectory_indices=None
                                 ):

    if not os.path.exists(fragments_plots_dir):
        os.makedirs(fragments_plots_dir)

    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

    # Check for existing data files and load if they exist
    num_trajectories = len(trajectories)
    all_traj_OH_populations = []
    all_traj_H2O_populations = []
    all_traj_H3O_populations = []
    all_traj_substrate_H_populations = []
    all_traj_times = []
    
    data_exists = True
    for i in trajectory_indices:
        csv_filename = f"{fragments_plots_dir}/{name}_traj_{i}_dissociation_data.csv"
        if os.path.exists(csv_filename):
            print(f"Loading existing data from: {csv_filename}")
            df = pd.read_csv(csv_filename)
            all_traj_times.append(df['Time_frames'].tolist())
            all_traj_OH_populations.append(df['OH_population'].tolist())
            all_traj_H2O_populations.append(df['H2O_population'].tolist())
            all_traj_H3O_populations.append(df['H3O_population'].tolist())
            all_traj_substrate_H_populations.append(df['H_substrate_population'].tolist())
        else:
            data_exists = False
            break
    
    # If data doesn't exist for all trajectories, calculate it
    if not data_exists:
        print("Existing data not found or incomplete. Calculating dissociation statistics...")
        os.makedirs(fragments_plots_dir+'/traj_data', exist_ok=True)
        populations_data = get_dissociation_statistics(trajectories, substrate, z_min, z_max, sampling_interval=sampling_interval, save_results=save_results, system_name=name, fragment_data_dir=fragments_plots_dir+'/traj_data', trajectory_indices=trajectory_indices)
        all_traj_OH_populations, all_traj_H2O_populations, all_traj_H3O_populations, all_traj_substrate_H_populations, all_traj_times = populations_data
        
        # Save the calculated data
        for i in trajectory_indices:
            df = pd.DataFrame({
                'Time_frames': all_traj_times[i-1],
                'OH_population': all_traj_OH_populations[i-1],
                'H2O_population': all_traj_H2O_populations[i-1],
                'H3O_population': all_traj_H3O_populations[i-1],
                'H_substrate_population': all_traj_substrate_H_populations[i-1]
            })

            csv_filename = f"{fragments_plots_dir}/{name}_traj_{i}_dissociation_data.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Saved trajectory {i} data to: {csv_filename}")
        
        # Save combined data
        combined_data = []
        for i in trajectory_indices:
            for j, time in enumerate(all_traj_times[i-1]):
                combined_data.append({
                    'Trajectory': i,
                    'Time_frames': time,
                    'OH_population': all_traj_OH_populations[i-1][j],
                    'H2O_population': all_traj_H2O_populations[i-1][j],
                    'H3O_population': all_traj_H3O_populations[i-1][j],
                    'H_substrate_population': all_traj_substrate_H_populations[i-1][j]
                })
        combined_df = pd.DataFrame(combined_data)
        combined_csv_filename = f"{fragments_plots_dir}/{name}_all_trajectories_dissociation_data.csv"
        combined_df.to_csv(combined_csv_filename, index=False)
        print(f"Saved combined data to: {combined_csv_filename}")

    # Create subplots with extra space for legend on the right
    fig, axes = plt.subplots(num_trajectories, 1, figsize=(14, 4 * num_trajectories), sharex=True)

    # Handle case of single trajectory
    if num_trajectories == 1:
        axes = [axes]
    
    # Store legend elements from first trajectory for single legend
    legend_lines = []
    legend_labels = []
    
    # Plot each trajectory
    for i in trajectory_indices:
        times = np.array(all_traj_times[i-1]) * dt / 1000  # Converting to ps of simulation time
        OH_pop = all_traj_OH_populations[i-1]
        H2O_pop = all_traj_H2O_populations[i-1]
        H3O_pop = all_traj_H3O_populations[i-1]
        H_substrate_pop = all_traj_substrate_H_populations[i-1]

        # Primary y-axis for dissociated species (OH-, H3O+)
        ax1 = axes[i-1]
        line1 = ax1.plot(times, OH_pop, label=r'$\mathrm{OH}^-$', color='red', linewidth=1.5, linestyle='--')
        line2 = ax1.plot(times, H3O_pop, label=r'$\mathrm{OH}_3^+$', color='green', linewidth=1.5, linestyle='-.')
        line3 = ax1.plot(times, H_substrate_pop, label='Substrate H', color='purple', linewidth=1.5, linestyle=':')
        ax1.set_ylabel('Dissociated Species Count', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # Secondary y-axis for H2O
        ax2 = ax1.twinx()
        line4 = ax2.plot(times, H2O_pop, label=r'$\mathrm{H_2O}$', color='blue', linewidth=1.5)
        ax2.set_ylabel(r'$\mathrm{H_2O}$', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Collect legend elements from first trajectory only
        if i == 0:
            legend_lines = line1 + line2 + line3 + line4
            legend_labels = [l.get_label() for l in legend_lines]

        ax1.set_title(f'Trajectory {i}: Water Species Population vs Time')

    # Set x-label only for bottom plot
    axes[-1].set_xlabel('Time (ps)')
    
    # Create single legend to the right of all subplots
    fig.legend(legend_lines, legend_labels, 
               loc='center right', 
               bbox_to_anchor=(0.98, 0.5),
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # Adjust subplot parameters to make room for legend
    plt.subplots_adjust(right=0.85)
    
    plt.suptitle(f'{name}: Water Dissociation Statistics (z: {z_min:.1f} - {z_max:.1f} Å)', fontsize=30)
    plt.tight_layout()
    plt.savefig(fragments_plots_dir + '/' + name + '_dissociation_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()





# def plot_angular_distributions(
#                         name,
#                         substrate,
#                         trajectories,
#                         O_z_min,
#                         O_z_max,
#                         sampling_interval=20,
#                         z_sampling_increment = 0.05,
#                         num_cos_bins=100,
#                         save_results = True,
#                         ):
    

#     pass
