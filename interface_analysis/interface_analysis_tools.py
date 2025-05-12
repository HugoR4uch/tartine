import ase
import ase.io
import numpy as np
import copy
import math
import os
import time
import matplotlib.pyplot as plt
import importlib 
import pandas as pd 

from ase.constraints import FixAtoms
from mace.calculators.foundations_models import mace_mp
from ase.optimize import BFGS
from ase.optimize import FIRE
import ase.data
from ase.constraints import StrainFilter
from scipy.signal import find_peaks





def find_atomic_trajectories(input_trajectory, atom_indices = None,relative_to_COM=False):
    """
    Returns 
    -------
    -all_trajectories: ndarray of trajectories for all atoms whos indices are in <indices>. Element i is the trajectory of the atom who's index is the ith member of <indices>.
    """


    if relative_to_COM:       
        #Find traj of COM
        COM_trajectory = find_COM_trajectory(input_trajectory)

    if atom_indices == None:
        atom_indices = np.arange(0,len(input_trajectory[0]),1)

    frame_indices=np.arange(0,len(input_trajectory),1) #if no frame indices specified, assume whole trajectory
    
    all_trajectories= [] #array populated by trajectories of each atoms
    for atom_index in atom_indices:
        trajectory_of_single_atom = [] #trajectory of atom with index 'atom_index'
        for frame_index in frame_indices:
            if relative_to_COM:
                trajectory_of_single_atom.append(input_trajectory[frame_index][atom_index].position  - COM_trajectory[frame_index] ) 
            else:
                trajectory_of_single_atom.append( input_trajectory[frame_index][atom_index].position )
        all_trajectories.append(np.array(trajectory_of_single_atom))
    all_trajectories = np.array(all_trajectories)
    return all_trajectories




def find_COM_trajectory(input_trajectory):
    """
    Returns 
    -------
    -COM_trajectory: ndarray of trajectories for the center of mass of the system. Element i is the trajectory of the COM at frame i.
    """
    frame_indices=np.arange(0,len(input_trajectory),1) #if no frame indices specified, assume whole trajectory
    
    COM_trajectory= [] #array populated by trajectories of each atoms
    for frame_index in frame_indices:
        COM_trajectory.append( input_trajectory[frame_index].get_center_of_mass() ) 
    COM_trajectory = np.array(COM_trajectory)
    return COM_trajectory



def find_top_layer_indices(substrate,num_layers):
    z_vals = substrate.positions[:,2]
    
    if num_layers == None:
        top_layer_z_val_threshold = np.max(z_vals) - 0.1 # anything 0.1 A below top atom
    else:
        top_layer_z_val_threshold = np.percentile(z_vals, 100*(1-1/num_layers))
    
    top_layer_indices = np.where(z_vals >= top_layer_z_val_threshold)[0]
    return top_layer_indices



def get_z_density_profile(trajectories,substrate,z_min,z_max,plot_all_profiles=False, num_layers = 4,bins=400,species='O'):

    #NOTE: You need to put in equilibrated trajectories (they will not be truncated in this function)

    n_trajectories = len(trajectories)
  
    mass_densities = [] 

    #Finding densities for each independent trajectory of a system
    #Will spit out mean densty(z) and corresponding error bars 
    for slab in trajectories:
        

        input_trajectory = copy.deepcopy(slab) # only sample up to max_T
        #print(max_T_val)

        ##########################################
        #Finding interface - water relative z vals
        ##########################################

        #Finding the interface z-value
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=True)

        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): # transpose fron (atom index, frame index, coord) to (frame index, atom index, coord index)
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 


        #Aggregating Z vals over entire trajectory and for all O atoms (that are not in the substrate)
        all_O_indices = np.where(input_trajectory[0].symbols == species) [0] 
        substrate_0_indices = np.where(substrate.symbols == species) [0]
        O_indices = np.setdiff1d(all_O_indices, substrate_0_indices)
        all_O_trajectories = find_atomic_trajectories(input_trajectory,O_indices,relative_to_COM=True)
        all_O_traj_relative_to_interface = []
        for frame_index, frame_positions in enumerate(all_O_trajectories.transpose(1,0,2)):
            frame_z_vals = frame_positions[:,2]
            frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
            all_O_traj_relative_to_interface.append(frame_z_vals_relative_to_interface)

        all_O_atom_z_val_displacements = np.array(all_O_traj_relative_to_interface).flatten()
        data = all_O_atom_z_val_displacements 


        ##############################
        #Binning z values -> densities 
        ##############################
 

        # Generate histogram data
        counts, bin_edges = np.histogram(data, range = [z_min,z_max], bins=bins)

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

        #Bin width 
        dz = (z_max - z_min)/bins

        T=len(input_trajectory)
        v1=input_trajectory[0].cell[0]
        v2=input_trajectory[0].cell[1]
        box_area = np.linalg.norm(np.cross(v1,v2))

        #Convert counts to density 
        number_density = counts/(T*box_area*dz)
        mass_density =  29.915 *number_density
        mass_densities.append(mass_density)

    #Taking the mean and std of the densities 
    mass_densities = np.array(mass_densities)
    average_density = []
    errors = []
    for i in range(bins):
        average_density.append(np.mean(mass_densities[:,i]))
        errors.append(np.std(mass_densities[:,i])/np.sqrt(n_trajectories))


    #Returning densities
    if plot_all_profiles:
        return bin_centers, mass_densities
    else:
        return bin_centers, average_density, errors 



# def get_z_density_turning_points(z_values, density_profile, prominence=0.3):

#     # This functions is deceptive. Could be called get first peak width, or something like that.
#     # The real get_z_turning_points should actually return: peaks, troughs.


#     # Find the first z value with a non-zero density
#     density_profile = np.array(density_profile)
#     non_zero_indices = np.where(density_profile > 0)[0]
#     first_non_zero_index = non_zero_indices[0]
#     first_non_zero_z = z_values[first_non_zero_index]

#     # Find peaks in the density profile
#     peaks, _ = find_peaks(density_profile, distance=10, prominence=prominence)
    
#     if len(peaks) == 0:
#         first_trough_z= None
#     else:

#         # Get the first peak
#         print('Peaks: ',peaks)
#         first_peak_index = peaks[0]
#         first_peak_z = z_values[first_peak_index]

#         # Find troughs (local minima) in the density profile
#         troughs, _ = find_peaks(-density_profile, distance=10, prominence=prominence)
#         if len(troughs) == 0:
#             first_trough_z= None
#         else:
#         # Find the first trough after the first peak
#             first_trough_index = troughs[troughs > first_peak_index][0]
#             first_trough_z = z_values[first_trough_index]

#         return first_non_zero_z, first_trough_z

        

def get_xy_RDF(trajectories,
               substrate,
               species_indices_1,
               species_indices_2,
               num_layers=None,
               z_min=None,
               z_max=None,
               r_max=10):


    for trajectory in trajectories:
        

            find_top_layer_indices(substrate,num_layers)

            #Finding the interface z-value
            substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
            all_top_layer_atom_trajectories = find_atomic_trajectories(trajectory,
                                                                       substrate_top_layer_indices,
                                                                       relative_to_COM=True)

            interface_z_mean_traj = []
            # transpose fron (atom index, frame index, coord) to (frame index, atom index, coord index)
            for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): 
                z_vals = frame_positions[:,2]
                interface_z_mean_traj.append(np.mean(z_vals)) 


            for frame in trajectory:
        

                #Aggregating Z vals over entire trajectory and for all O atoms (that are not in the substrate)
                all_O_indices = np.where(input_trajectory[0].symbols == species) [0] 
                substrate_0_indices = np.where(substrate.symbols == species) [0]
                O_indices = np.setdiff1d(all_O_indices, substrate_0_indices)
                all_O_trajectories = find_atomic_trajectories(input_trajectory,O_indices,relative_to_COM=True)
                all_O_traj_relative_to_interface = []
                for frame_index, frame_positions in enumerate(all_O_trajectories.transpose(1,0,2)):
                    frame_z_vals = frame_positions[:,2]
                    frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
                    all_O_traj_relative_to_interface.append(frame_z_vals_relative_to_interface)

                all_O_atom_z_val_displacements = np.array(all_O_traj_relative_to_interface).flatten()
                data = all_O_atom_z_val_displacements 

























def get_xy_density_profile(trajectories,species,z_min,z_max,L,substrate,return_all_profiles=False,num_layers=4):

    #NOTE: You need to put in equilibrated trajectories (they will not be truncated in this function)
    #NOTE: THIS ONLY WORKS FOR ORTHORHOMBIC CELLS

    n_trajectories = len(trajectories)
  
    number_densities = [] 

    #Finding densities for each independent trajectory of a system
    #Will spit out mean densty(z) and corresponding error bars 
    for index , slab in enumerate(trajectories):
        

        #Density plot params
        bins= 100
        
        input_trajectory = copy.deepcopy(slab) 


        #Finding the interface z-value (so we can select atoms with z_min < z < z_max)
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        all_top_layer_atom_trajectories = find_atomic_trajectories(substrate_top_layer_indices,input_trajectory,relative_to_COM=True)

        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): # transpose fron (atom index, frame index, coord) to (frame index, atom index, coord index)
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 

        #Aggregating x,y vals over entire trajectory and for all <species> atoms
        all_species_indices = np.where(input_trajectory[0].symbols == species) [0]
        substrate_species_indices = np.where(substrate.symbols == species) [0]
        species_indices = np.setdiff1d(all_species_indices, substrate_species_indices)
        all_species_trajectories = find_atomic_trajectories(input_trajectory,species_indices,relative_to_COM=True)
        all_xy_pairs = []
        for frame_index, frame_positions in enumerate(all_species_trajectories.transpose(1,0,2)):
            frame_z_vals = frame_positions[:,2]
            frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
            valid_indices = np.where((frame_z_vals_relative_to_interface >= z_min) & (frame_z_vals_relative_to_interface <= z_max))[0]
            for idx in valid_indices:
                all_xy_pairs.append(frame_positions[idx][:2])
            

        data = np.array(all_xy_pairs)


        ##############################
        #Binning x,y values -> densities 
        ##############################


        # Generate histogram data

        heatmap, xedges, yedges = np.histogram2d(data[:,0],data[:,1], bins=[bins, bins],range=[[-L/2, L/2], [-L/2, L/2]],density=True)

        # Compute bin centers
        bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
        bin_centers_y = (yedges[:-1] + yedges[1:]) / 2

        #Bin volume 
        bin_width_x = xedges[1] - xedges[0]
        bin_width_y = yedges[1] - yedges[0]
        bin_height = z_max - z_min
        bin_volume = bin_width_x*bin_width_y*bin_height
        
        T=len(input_trajectory)

        #Convert counts to trajectory mean number density 
        number_density = heatmap/(T*bin_volume)
        number_densities.append(number_density)

    #Taking the mean and std of the densities 
    number_densities = np.array(number_densities)
    average_density = []
    errors = []

    average_density = np.zeros((bins,bins))
    errors = np.zeros((bins,bins))
    for x in range(bins):
        for y in range(bins):
            average_density[x][y] = np.mean(number_densities[:,x,y])
            errors[x][y] = np.std(number_densities[:,x,y])/np.sqrt(n_trajectories)

    #Returning densities
    if return_all_profiles:
        return bin_centers_x,bin_centers_y, number_densities
    else:
        return bin_centers_x,bin_centers_y, average_density, errors 



def get_xy_trajectory(trajectory, substrate, L, stride=10, num_layers=4):
    # Find the indices of the atoms in the top layer of the substrate
    top_layer_indices = find_top_layer_indices(substrate, num_layers)
    
    # Find the x, y trajectories of these atoms, relative to the COM of the system
    all_top_layer_atom_trajectories = find_atomic_trajectories(trajectory,top_layer_indices,  relative_to_COM=True)
    
    # Initialize lists to store x and y values
    all_x_values = []
    all_y_values = []

    # Filter x, y values to include only those within a box of width and height L around the origin
    for frame_positions in all_top_layer_atom_trajectories.transpose(1, 0, 2)[::stride]:  # (frame index, atom index, coord index)
        x_vals = frame_positions[:, 0]
        y_vals = frame_positions[:, 1]
        valid_indices = np.where((np.abs(x_vals) <= L / 2) & (np.abs(y_vals) <= L / 2))[0]
        all_x_values.extend(x_vals[valid_indices])
        all_y_values.extend(y_vals[valid_indices])

    return all_x_values, all_y_values




def get_xy_pair_correlations(trajectories,
                                        substrate,
                                        group_1_indices,
                                        group_2_indices,
                                        z_min,z_max,
                                        L,
                                        no_group_1_z_lims=False, no_group_2_z_lims=False,
                                        return_all_profiles=False,
                                        num_layers=4):

    #NOTE: You need to put in equilibrated trajectories (they will not be truncated in this function)
    #NOTE: THIS ONLY WORKS FOR ORTHORHOMBIC CELLS

    n_trajectories = len(trajectories)
  
    number_densities = [] 

    #Finding densities for each independent trajectory of a system
    #Will spit out mean densty(z) and corresponding error bars 
    for index , slab in enumerate(trajectories):
        

        #Density plot params
        bins= 100
        
        input_trajectory = copy.deepcopy(slab) 

        T=len(input_trajectory)


        #Finding the interface z-value (so we can select atoms with z_min < z < z_max)
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=True)

        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): # transpose fron (atom index, frame index, coord) to (frame index, atom index, coord index)
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 


        # Create masks for each frame indicating whether group 1 atoms are in the z region
        if no_group_1_z_lims:
            group_1_masks = np.ones((T, len(group_1_indices)), dtype=bool)
        else:
            group_1_trajectories = find_atomic_trajectories(input_trajectory,group_1_indices,  relative_to_COM=True)
            group_1_masks = []
            for frame_index, frame_positions in enumerate(group_1_trajectories.transpose(1, 0, 2)):
                frame_z_vals = frame_positions[:, 2]
                frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
                mask = (frame_z_vals_relative_to_interface >= z_min) & (frame_z_vals_relative_to_interface <= z_max)
                group_1_masks.append(mask)

        if no_group_2_z_lims:
            group_2_masks = np.ones((T, len(group_2_indices)), dtype=bool)
        else:
            group_2_trajectories = find_atomic_trajectories(input_trajectory,group_2_indices,  relative_to_COM=True)
            group_2_masks = []
            for frame_index, frame_positions in enumerate(group_2_trajectories.transpose(1, 0, 2)):
                frame_z_vals = frame_positions[:, 2]
                frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
                mask = (frame_z_vals_relative_to_interface >= z_min) & (frame_z_vals_relative_to_interface <= z_max)
                group_2_masks.append(mask)


        

        displacements = []

        for frame_index in range(len(input_trajectory)):
            in_z_region_group_1_indices = np.where(group_1_masks[frame_index])[0]
            in_z_region_group_2_indices = np.where(group_2_masks[frame_index])[0]

            for index in in_z_region_group_1_indices:
                other_indices = np.setdiff1d(in_z_region_group_2_indices,index)
                displacements.append( input_trajectory[frame_index].get_distances(index,other_indices,vector=True,mic=True)[:,:2] )
        
        flat_distances =  np.array(displacements).reshape(-1, 2)

         

        distances = flat_distances


        #350K 2d correlation function
        plt.hist2d(distances[:,0],distances[:,1],bins=100)



        #Aggregating x,y vals over entire trajectory and for all <species> atoms
        species_indices = np.where(input_trajectory[0].symbols == species) [0]
        all_species_trajectories = find_atomic_trajectories(input_trajectory,species_indices,relative_to_COM=True)
        all_xy_pairs = []
        for frame_index, frame_positions in enumerate(all_species_trajectories.transpose(1,0,2)):
            frame_z_vals = frame_positions[:,2]
            frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
            valid_indices = np.where((frame_z_vals_relative_to_interface >= z_min) & (frame_z_vals_relative_to_interface <= z_max))[0]
            for idx in valid_indices:
                all_xy_pairs.append(frame_positions[idx][:2])
  
  
            

        data = np.array(all_xy_pairs)


        ##############################
        #Binning x,y values -> densities 
        ##############################


        # Generate histogram data

        heatmap, xedges, yedges = np.histogram2d(data[:,0],data[:,1], bins=[bins, bins],range=[[-L/2, L/2], [-L/2, L/2]],density=True)

        # Compute bin centers
        bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
        bin_centers_y = (yedges[:-1] + yedges[1:]) / 2

        #Bin volume 
        bin_width_x = xedges[1] - xedges[0]
        bin_width_y = yedges[1] - yedges[0]
        bin_height = z_max - z_min
        bin_volume = bin_width_x*bin_width_y*bin_height
        
        

        #Convert counts to trajectory mean number density 
        number_density = heatmap/(T*bin_volume)
        number_densities.append(number_density)

    #Taking the mean and std of the densities 
    number_densities = np.array(number_densities)
    average_density = []
    errors = []

    average_density = np.zeros((bins,bins))
    errors = np.zeros((bins,bins))
    for x in range(bins):
        for y in range(bins):
            average_density[x][y] = np.mean(number_densities[:,x,y])
            errors[x][y] = np.std(number_densities[:,x,y])/np.sqrt(n_trajectories)

    #Returning densities
    if return_all_profiles:
        return bin_centers_x,bin_centers_y, number_densities
    else:
        return bin_centers_x,bin_centers_y, average_density, errors 

