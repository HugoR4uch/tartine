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



def extract_simulation_data(simulation_runs_dir,
                            name,
                            dirs,
                            equilib_end_frame,
                            ):
                           
    
    #Loading trajectories, substrates, logfiles
    substrates = []
    trajectories = []
    logfile_paths = []
    

    for i,traj_name in enumerate(dirs):
        traj_dir = simulation_runs_dir + '/' + traj_name
        print('Loading trajectory: ', traj_dir)
        substrate_filename = '_'.join(traj_name.split('_')[:2])
        substrate = ase.io.read(traj_dir +'/' + substrate_filename + '.pdb')
        substrates.append(substrate)

        traj_path = traj_dir + '/' + traj_name + '.xyz'

        if os.path.isfile(traj_path):
            traj = ase.io.read(traj_dir + '/' + traj_name + '.xyz',index=':')    
        else:
            print(traj_path,' does not exist! Skipping this trajectory.')
            continue
        
        trajectories.append(traj)
        print('Number of trajectories: ',len(trajectories))

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


    return substrates, trajectories, logfile_paths



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
        fig, ax = plt.subplots(1,2,figsize=(8, 8),sharex=True,sharey=False)
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
                         density_vs_aimd_figures_dir='.'):
    

    plt.plot(O_bin_centers,O_average_density,label = current_model_name)
    max_density = np.max(O_average_density) 
    plt.ylim([0, max_density * 1.2])  
    plt.xlabel(r'Distance From Interface z [$\AA$]')
    plt.ylabel(r'Density [$gcm^{-3}$]')
    plt.grid()
    plt.title(name + ' Water Density Profile')


    for density_profile_file in AIMD_density_profiles[name]:
        profile_data = density_profile_file.split('/')[-1].split('.')[0]
        print('Extracting data from:', profile_data)
        temp = profile_data.split('_')[1]
        XC_functional = profile_data.split('_')[2]
        author = profile_data.split('_')[3]
        profile_name = f'{temp}K {XC_functional} {author}'

        data = np.loadtxt(density_profile_file, delimiter=',', skiprows=1)
        z = data[:, 0]
        density = data[:, 1]
        plt.plot(z, density, label=profile_name)
        new_max_density = np.max(density)
        if new_max_density > max_density:
            max_density = new_max_density

        plt.legend()
        plt.xlim([0,20])
        plt.ylim([0,max_density*1.2])
        plt.savefig(density_vs_aimd_figures_dir + '/' + name + '_density_vs_z.png')


    plt.close()



def multi_gen_density_plot(
                    current_O_bin_centers,
                    current_O_average_density,
                    name,
                    name_of_this_gen,
                    older_gen_sim_dirs,
                    multi_gen_density_figures_dir= './multi_gen_density_figures',
                    equilib_end_frame = 4000
                    ):
    
    
    for gen_name in older_gen_sim_dirs.keys():
        if gen_name == name_of_this_gen:
            continue

        gen_sim_dir = older_gen_sim_dirs[gen_name]
        if not os.path.exists(gen_sim_dir):
            print('The',gen_name, 'does not exist:'+gen_sim_dir)
            return

        sim_dirs = os.listdir(gen_sim_dir)

        system_sim_dirs=[dir for dir in sim_dirs if name in dir]

                

        substrates, trajectories, _ = extract_simulation_data(
                                                    gen_sim_dir,
                                                    name,
                                                    system_sim_dirs, # note, dirs for system only
                                                    equilib_end_frame,
                                                    )
        
            
        O_data = interface_analysis_tools.get_z_density_profile(trajectories,
                                                                substrates[0],
                                                                z_min=-1,
                                                                z_max=30,
                                                                plot_all_profiles=False,
                                                                num_layers = None)

        O_bin_centers, O_average_density, _ = O_data

        plt.plot(O_bin_centers,O_average_density,label = gen_name)



    plt.plot(current_O_bin_centers,current_O_average_density,label = name_of_this_gen)

    plt.legend()
    plt.grid()
    plt.xlabel(r'Distance From Interface z [$\AA$]')
    plt.ylabel(r'Water Density (from O atom location)[$gcm^{-3}$]')
    plt.title(name + ' Water Density Profile for Different Generations')
    plt.savefig(multi_gen_density_figures_dir + '/' + name + 'multi_gen_density_vs_z.png')
    plt.close()



def plot_angular_distributions(
                               name,
                               substrate,
                               trajectories,
                               O_z_min,
                               O_z_max,
                               contact_layer_angle_vs_z_plots_dir='./',
                               contact_layer_angular_plots_dir='./',
                               ):

        if not os.path.exists(contact_layer_angular_plots_dir):
            os.makedirs(contact_layer_angular_plots_dir)

        if not os.path.exists(contact_layer_angle_vs_z_plots_dir):
            os.makedirs(contact_layer_angle_vs_z_plots_dir)

        all_z_angles_coords = []

        for traj in trajectories:
            for frame in tqdm(traj[::5], desc=f"Processing traj frames for Angular anlysis {name}"):
                data = interface_analysis_tools.get_interfacial_z_vs_dipole_angles(frame,substrate)
                _ , z , angles = data
                contact_layer_coords = [ [z_val,theta] for z_val,theta in zip(z,angles) if O_z_min < z_val < O_z_max ]
                all_z_angles_coords.extend(contact_layer_coords)


        plt.hist2d(*zip(*all_z_angles_coords), bins=200,density=True, range=[[2.5, 4], [0, 180]], cmap='Blues')
        plt.colorbar(label='Probability Density')
        plt.xlabel('z [Å]')
        plt.ylabel('Water Dipole Angle Perpendicular to Interface [°]')
        plt.title(f'{name} Water Dipole Angle Distribution in Contact Layer')
        plt.savefig(contact_layer_angle_vs_z_plots_dir + '/' + name + '_angle_vs_z.png')
        plt.close()


        plt.hist(np.array(all_z_angles_coords)[:,1],bins=200,density=True)
        plt.xlabel('Water Dipole Angle Perpendicular to Interface [°]')
        plt.ylabel('Probability Density')
        plt.title(f'{name} Water Dipole Angle Distribution in Contact Layer')
        plt.savefig(contact_layer_angular_plots_dir + '/' + name + '_angle_distribution.png')
        plt.close()

def plot_default_density_vs_z(name,
                                O_bin_centers,
                                O_average_density,
                                O_errors,
                                H_bin_centers,
                                H_average_density,
                                H_errors,
                                density_vs_z_figures_dir='./',
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

    for a in ax:
        a.set_ylim([0, maximum_density * 1.2])
        a.set_xlim([0, 20])        
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
                                 ):

    if not os.path.exists(density_vs_z_figures_dir):
        os.makedirs(density_vs_z_figures_dir)


    peaks_in_contact_layer = [z for z in peak_O_z_vals if z < O_contact_layer_end]
    contact_layer_z_mask = [ O_contact_layer_start <= z <= O_contact_layer_end for z in O_bin_centers]
    contact_layer_mean_density = np.mean(np.array(O_average_density)[contact_layer_z_mask]) 
    

    O_max_density = np.max(O_average_density)
    H_max_density = np.max(H_average_density)

    maximum_density = max(O_max_density, H_max_density)

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
        
        a.set_ylim([0, maximum_density * 1.2])
        a.set_xlim([0, 20])

        for i, peak_z in enumerate(peaks_in_contact_layer):
            a.plot([peak_z, peak_z], [0, maximum_density * 1.2], 'r--', label=f'Water O Peak {i+1}: {peak_z:.1f} Å')
        
        a.plot([O_contact_layer_start, O_contact_layer_start], [0, maximum_density * 1.2], 'k--', label=f'Contact Layer Start: {O_contact_layer_start:.1f} Å')
        a.plot([O_contact_layer_end, O_contact_layer_end], [0, maximum_density * 1.2], 'k--', label=f'Contact Layer End: {O_contact_layer_end:.1f} Å')
        a.set_xlabel(r'Distance From Interface z [$\AA$]')
        a.set_ylabel(r'Density [$gcm^{-3}$]')
        
        a.grid()
        
    fig.tight_layout()

    ax[1].legend()

    plt.savefig(density_vs_z_figures_dir + '/' + name + '_density_vs_z.png')

    plt.close()


def plot_H_bonds_vs_z(name,
                      substrate,
                      trajectories,
                      H_bond_vs_z_figures_dir='./H_bonds_vs_z_figures',
                      ):
    
    print('Calculating H bonds vs z for:', name)
    if not os.path.exists(H_bond_vs_z_figures_dir):
        os.makedirs(H_bond_vs_z_figures_dir)


    #Finding the interface z-value
    



    z_vals=[]
    bond_counts=[]

    start_time = time.time()

    for traj in trajectories:
    
        num_layers= None
        substrate_top_layer_indices = interface_analysis_tools.find_top_layer_indices(substrate,num_layers)
        all_top_layer_atom_trajectories = interface_analysis_tools.find_atomic_trajectories(traj,substrate_top_layer_indices,relative_to_COM=False)

        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): # transpose fron (atom index, frame index, coord) to (frame index, atom index, coord index)
            substrate_z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(substrate_z_vals))
    
        for frame_index, frame in tqdm(enumerate(traj[::20])):
            water_O_indices = [atom.index for atom in frame if atom.tag == 1 and atom.symbol == 'O']
            analyser = water_analyser.Analyser(frame)
            connectivity_matrix = analyser.get_H_bond_connectivity(directed=False)
            positions= frame.get_positions()
            frame_z_vals = positions[water_O_indices][:,2]
            frame_z_vals = [z - interface_z_mean_traj[frame_index] for z in frame_z_vals]
            frame_bond_counts = [np.sum(list(connectivity_matrix[i].values())) for i in water_O_indices]
            z_vals.extend(list(frame_z_vals))
            bond_counts.extend(frame_bond_counts)

    print(interface_z_mean_traj)

    z_vals = np.array(z_vals)
    bond_counts = np.array(bond_counts)


    # Choose bins
    bins = np.linspace(z_vals.min(), z_vals.max(), 200)

    # Bin sums (numerator): total bond count per bin
    bond_sum_per_bin, _ = np.histogram(z_vals, bins=bins, weights=bond_counts)

    # Bin counts (denominator): number of molecules per bin
    counts_per_bin, _ = np.histogram(z_vals, bins=bins)

    # Avoid division by zero
    mean_bonds_per_bin = np.divide(bond_sum_per_bin, counts_per_bin, where=counts_per_bin!=0)

    # Compute bin centers for plotting
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    end_time = time.time()
    print(f"Time taken to calculate H bonds vs z: {end_time - start_time:.2f} seconds")


    plt.plot(bin_centers, mean_bonds_per_bin)
    plt.xlabel('z [Å]')
    plt.ylabel('Mean Number of H Bonds (per Water)')
    plt.grid()
    plt.title('Mean H Bond Coordination vs z')

    plt.savefig(H_bond_vs_z_figures_dir + '/' + name+ '_H_bonds_vs_z.png')
    plt.close()

def get_dissociation_statistics(name,trajectories,fragments_plots_dir= './fragments_plots'):

    if not os.path.exists(fragments_plots_dir):
        os.makedirs(fragments_plots_dir)
    print('Calculating dissociation statistics for:', name)

    occupancies = {}

    for traj in trajectories:
        for frame in tqdm(traj[::10]):
            water_O_indices = [atom.index for atom in frame if atom.tag == 1 and atom.symbol == 'O']
            num_water_molecules = len(water_O_indices)
            analyser = water_analyser.Analyser(frame)
            voronoi_dict = analyser.get_voronoi_dict()
            for i in water_O_indices:
                local_H_list = voronoi_dict[i]
                num_H= len(local_H_list) 
                if not num_H in occupancies:
                    occupancies[num_H]=0
                else:
                    occupancies[num_H]+=1
            

    occupancy_levels = sorted(occupancies.keys())
    counts = [occupancies[o] for o in occupancy_levels]
    total = sum(counts)
    # Normalize counts to get occupancy levels
    counts = [count / total for count in counts]

    # Convert occupancy levels to strings to make them categorical
    categories = [str(o) for o in occupancy_levels]

    


    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(categories, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Hydrogens (Categorical)')
    plt.ylabel('Count')
    plt.title('Distribution of Hydrogen Occupancy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fragments_plots_dir + '/' + name + '_hydrogen_occupancy.png')
    plt.close()
    print('Occupancy levels:',occupancy_levels)



