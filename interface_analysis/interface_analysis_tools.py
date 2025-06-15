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
from tqdm import tqdm
import spglib

from ase.constraints import FixAtoms
from mace.calculators.foundations_models import mace_mp
from ase.optimize import BFGS
from ase.optimize import FIRE
import ase.data
from ase.constraints import StrainFilter
from scipy.signal import find_peaks

from . import water_analyser



def find_atomic_trajectories(input_trajectory, atom_indices = None,relative_to_COM=False):
    """
    Returns 
    -------
    -all_trajectories: ndarray of trajectories for all atoms whos indices are in <indices>. 
    Element i is the trajectory of the atom who's index is the ith member of <indices>.
    """


    if relative_to_COM:       
        #Find traj of COM
        COM_trajectory = find_COM_trajectory(input_trajectory)

    if atom_indices is None:
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



def find_water_species_indices_traj(
    trajectory,
    substrate,
    species_criterion_function,
    **kwargs
):
    """
    Returns
    -------
    -water_indices: index i is the 1d array of indices atoms satisfying the specie criterion for frame i.
    
    Parameters
    ----------
    -trajectory: list of ase.Atoms objects, each representing a frame in the trajectory.
    -substrate: ase.Atoms object representing the substrate.
    -species_criterion_function: function that takes an ase.Atoms frame, and substrate
      and returns an array of indices of atoms belonging to desired specie.
    """

    species_indices_traj = []

    for frame in tqdm(trajectory, desc="Finding species indices in trajectory"):

        # Get the indices of atoms that satisfy the species criterion
        specie_indices = species_criterion_function(frame, substrate,**kwargs)
        species_indices_traj.append(specie_indices)

    return species_indices_traj



def interfacial_water_criterion(frame, substrate, z_min, z_max):

    """
    Returns
    -------
    -interfacial_water_indices: list of indices of atoms (O and H) beloinging to 
        interfacial water molecules in the frame.
    Parameters
    ----------
    -frame: ase.Atoms object representing the frame.
    -substrate: ase.Atoms object representing the substrate.
    -z_min: minimum z-coordinate for interfacial water.
    -z_max: maximum z-coordinate for interfacial water.
    """


    substrate_indices = np.arange(len(substrate))

    analyser = water_analyser.Analyser(frame,substrate_indices)
    
    aqua_O_indices = analyser.aqua_O_indices

    #Finding the interface z-value
    substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers=None)

    frame_positions = frame.get_positions()
    z_vals = frame_positions[:,2]
  
    z_interface = np.mean(z_vals[substrate_top_layer_indices])

    interfacial_water_O_indices = [water_index for water_index in aqua_O_indices if z_min <= frame[water_index].position[2] - z_interface <= z_max]
    
    if len(interfacial_water_O_indices) == 0:
        return []

    voronoi_dict = analyser.get_voronoi_dict(aqua_O_indices)

    
    interfacial_water_indices = []
    interfacial_water_indices.extend(interfacial_water_O_indices)

    for O_index in interfacial_water_O_indices:
            interfacial_water_indices.extend(voronoi_dict[O_index])

    return interfacial_water_indices

def chemisorbed_water_criterion(frame, substrate,theta_c = 120/180*np.pi):

    """
    Returns
    -------
    -interfacial_water_indices: tuple: list of indices of water O atoms belonging to 
        the chemisorbed state and another list of O atoms in physisorbed the state in the frame.
    """

    substrate_indices = np.arange(len(substrate))

    analyser = water_analyser.Analyser(frame,substrate_indices)
    aqua_O_indices = analyser.aqua_O_indices

    chemisorbed_water_O_indices = []
    physisorbed_water_O_indices = []
    bridging_water_indices = []

    water_H_vectors = analyser.get_water_H_vectors(aqua_O_indices)


    for O_index in aqua_O_indices:
        H_vectors = water_H_vectors[O_index]

        if len(H_vectors) != 2:
            continue
        
        theta_1 = np.arccos(np.dot(H_vectors[0], [0, 0, 1]) / (np.linalg.norm(H_vectors[0]) * np.linalg.norm([0, 0, 1])))
        theta_2 = np.arccos(np.dot(H_vectors[1], [0, 0, 1]) / (np.linalg.norm(H_vectors[1]) * np.linalg.norm([0, 0, 1])))


        if theta_1 < theta_c and theta_2 < theta_c:
            chemisorbed_water_O_indices.append(O_index)

        elif ( theta_c < theta_1 and theta_2 < theta_c ) or ( theta_c < theta_2 and theta_1 < theta_c ):
            physisorbed_water_O_indices.append(O_index)
            

        elif theta_1 > theta_c and theta_2 > theta_c:
            bridging_water_indices.append(O_index)
        
    return chemisorbed_water_O_indices, physisorbed_water_O_indices, bridging_water_indices


def get_z_density_profile(trajectories,
                          substrate,
                          z_min=None,
                          z_max=None,
                          plot_all_profiles=False,
                          num_layers = None,
                          bins=400,
                          sampling_indices_trajectory = None,
                          species='O'):


    n_trajectories = len(trajectories)
  
    mass_densities = [] 

    for slab in trajectories:

        input_trajectory = copy.deepcopy(slab) # only sample up to max_T

        ##########################################
        #Finding interface - water relative z vals
        ##########################################

        #Finding the interface z-value
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        

        if sampling_indices_trajectory is not None:
                    all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=False)
        else:
            all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=True)


        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): 
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 


        #Aggregating Z vals over entire trajectory and for all O atoms (that are not in the substrate)


        if sampling_indices_trajectory is not None:

            sampling_trajectories = []
            for frame_index, indices in enumerate(sampling_indices_trajectory):
                sample_positions = slab[frame_index][indices].get_positions()
                sampling_trajectories.append(sample_positions)

        else:
            all_X_indices = np.where(input_trajectory[0].symbols == species) [0] 
            substrate_X_indices = np.where(substrate.symbols == species) [0]
            X_indices = np.setdiff1d(all_X_indices, substrate_X_indices)
            all_X_trajectories = find_atomic_trajectories(input_trajectory,X_indices,relative_to_COM=True)
            sampling_trajectories = all_X_trajectories.transpose(1,0,2)


        all_X_atom_z_val_displacements = []


        for frame_index, frame_positions in enumerate(sampling_trajectories):
            frame_z_vals = frame_positions[:,2]
            frame_z_vals_relative_to_interface = frame_z_vals - interface_z_mean_traj[frame_index]
            all_X_atom_z_val_displacements.extend(frame_z_vals_relative_to_interface)

        data = all_X_atom_z_val_displacements 




        ##############################
        #Binning z values -> densities 
        ##############################
 

        if z_min is None:
            z_min = np.min(data) - 0.1 * (np.max(data) - np.min(data))
        if z_max is None:
            z_max = np.max(data) + 0.1 * (np.max(data) - np.min(data))


        counts, bin_edges = np.histogram(data, range = [z_min,z_max], bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

        dz = (z_max - z_min)/bins


        if sampling_indices_trajectory is not None:
            T= len(sampling_indices_trajectory)
        else:
            T=len(input_trajectory)


        v1=input_trajectory[0].cell[0]
        v2=input_trajectory[0].cell[1]
        box_area = np.linalg.norm(np.cross(v1,v2))

        print(T*box_area*dz)

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
    


def get_HH_fingerprint_data(trajectories,
                            substrate,
                            z_min,
                            z_max,
                            bins=100,
                            sampling_indices_trajectory=None,
                            ):
    
    num_layers = None

    n_trajectories = len(trajectories)
  
    mass_densities = [] 

    for slab in trajectories:

        input_trajectory = copy.deepcopy(slab) # only sample up to max_T

        ##########################################
        #Finding interface - water relative z vals
        ##########################################

        #Finding the interface z-value
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=True)

        if sampling_indices_trajectory is not None:
                    all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=False)


        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): 
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 


        #Aggregating fingerprint data over entire trajectory


        if sampling_indices_trajectory is None:

            sampling_indices_trajectory = [ np.arange(len(slab[0])) for i in range(len(slab)) ]


        substrate_indices = np.arange(len(substrate))

        all_HH_fingerprint_data = []

        for frame_index, indices in tqdm(enumerate(sampling_indices_trajectory)):
            
            
            analyser = water_analyser.Analyser(slab[frame_index], substrate_indices)
            H_vectors = analyser.get_water_H_vectors()
            frame_O_indices = [i for i in indices if i in analyser.aqua_O_indices]


            for index in frame_O_indices:  
                
                if len(H_vectors[index]) != 2:
                    continue

                theta_1 = np.arccos(np.dot(H_vectors[index][0], [0, 0, 1]) / (np.linalg.norm(H_vectors[index][0]) * np.linalg.norm([0, 0, 1])))
                theta_2 = np.arccos(np.dot(H_vectors[index][1], [0, 0, 1]) / (np.linalg.norm(H_vectors[index][1]) * np.linalg.norm([0, 0, 1])))

                all_HH_fingerprint_data.append([theta_1, theta_2])

        

        x_vals = np.array(all_HH_fingerprint_data)[:, 0]
        y_vals = np.array(all_HH_fingerprint_data)[:, 1]
        
        return x_vals, y_vals

def get_interfacial_z_vs_dipole_angles(frame,substrate,num_layers=None):

    substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
    substrate_top_layer_z_vals  =frame[substrate_top_layer_indices].positions[:,2] 
    substrate_top_layer_z_val = np.mean(substrate_top_layer_z_vals)
    frame_z_vals = frame.positions[:,2]
    frame_interfacial_z_vals = frame_z_vals - substrate_top_layer_z_val


    num_substrate_atoms = len(substrate) 
    frame_analyser = water_analyser.Analyser(frame)


    # We assume that the normal vector is (0,0,1)
    normal_vector = np.array([0, 0, 1])

    water_O_indices = [ i for i in range(len(frame)) if frame[i].symbol == 'O' and i >= num_substrate_atoms]

    voronoi_dict = frame_analyser.get_voronoi_dict(water_O_indices)
    
    indices = []
    interfacial_angles = []
    interfacial_z_vals = []

    for atom_index in water_O_indices:
        
        indices.append(atom_index)

        z = frame_interfacial_z_vals[atom_index]
        interfacial_z_vals.append( z ) 
 
        H_indices = voronoi_dict[atom_index]
        if len(H_indices) != 2:
            continue

        
        water_dipole = frame_analyser.get_water_dipole_moment(atom_index)
        
        # We normalise just to make sure
        dot_product = np.dot(water_dipole, normal_vector) / (np.linalg.norm(water_dipole) * np.linalg.norm(normal_vector))

        angle = np.arccos(dot_product) * 180 / np.pi

        interfacial_angles.append( angle) 


    return indices, interfacial_z_vals, interfacial_angles



def get_dipole_vs_interface_angles(frame,substrate,):

    num_substrate_atoms = len(substrate) 
    frame_analyser = water_analyser.Analyser(frame)


    # We assume that the normal vector is (0,0,1)
    normal_vector = np.array([0, 0, 1])

    water_O_indices = [ i for i in range(len(frame)) if frame[i].symbol == 'O' and i >= num_substrate_atoms]

    voronoi_dict = frame_analyser.get_voronoi_dict(water_O_indices)

    interface_dipole_angles = {}

    for atom_index in water_O_indices:

        H_indices = voronoi_dict[atom_index]
        if len(H_indices) != 2:
            continue

        
        water_dipole = frame_analyser.get_water_dipole_moment(atom_index)
        
        # We normalise just to make sure
        dot_product = np.dot(water_dipole, normal_vector) / (np.linalg.norm(water_dipole) * np.linalg.norm(normal_vector))

        angle = np.arccos(dot_product) * 180 / np.pi

        interface_dipole_angles[atom_index] = angle
    

    return interface_dipole_angles




def get_substrate_primitive_unit_cell(substrate,symprec=1e-3):

    copy_substrate = copy.deepcopy(substrate)
    copy_substrate.cell[2]= np.array([0, 0, 100.0])
    lattice = copy_substrate.get_cell()

    positions = copy_substrate.get_scaled_positions()
    numbers = copy_substrate.get_atomic_numbers()
    cell = (lattice, positions, numbers)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    primitive_cell = dataset.primitive_lattice

    origin_shift = dataset.origin_shift @ lattice

    return primitive_cell, origin_shift




def get_xy_density_profile(trajectories,
                           substrate, 
                           z_min=None,
                           z_max=None,
                           return_all_traj_data=False,
                           num_layers=None,
                           N_bins=50,
                           sampling_indices_trajectory = None,
                           species='O',
                           symprec=1e-2):

    primitive_cell, origin_shift = get_substrate_primitive_unit_cell(substrate,symprec=symprec)
    print(primitive_cell)

    original_unit_cell = substrate.cell.copy()

    data_vs_traj = {}

    for index, slab in enumerate(trajectories):

        for frame in slab:
            frame.cell = primitive_cell
            frame.wrap()


        input_trajectory = copy.deepcopy(slab) 


        #Finding the interface z-value (so we can select atoms with z_min < z < z_max)
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        # all_top_layer_atom_trajectories = find_atomic_trajectories(substrate_top_layer_indices,input_trajectory,relative_to_COM=True)


        all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,
                                                                substrate_top_layer_indices,
                                                                relative_to_COM=False)


        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): 
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 

        

        if sampling_indices_trajectory is not None:

            sampling_trajectories = []
            for frame_index, indices in enumerate(sampling_indices_trajectory):
                species_indices = [i for i in indices if slab[frame_index][i].symbol == species]

                sample_positions = slab[frame_index][species_indices].get_positions()
                sampling_trajectories.append(sample_positions)

        else:
            raise NotImplementedError("You need to provide sampling indices trajectory or implement the sampling indices selection logic.")
            # all_X_indices = np.where(input_trajectory[0].symbols == species) [0] 
            # substrate_X_indices = np.where(substrate.symbols == species) [0]
            # X_indices = np.setdiff1d(all_X_indices, substrate_X_indices)
            # all_X_trajectories = find_atomic_trajectories(input_trajectory,X_indices,relative_to_COM=True)
            # sampling_trajectories = all_X_trajectories.transpose(1,0,2)


        all_x_vals = []
        all_y_vals = []
        all_z_vals = []

        for frame_index, frame_positions in enumerate(sampling_trajectories):
            print(len(frame_positions))
            all_x_vals.extend(frame_positions[:,0])
            all_y_vals.extend(frame_positions[:,1])
            all_z_vals.extend(frame_positions[:,2])




        ##############################
        #Binning x,y values -> densities 
        ##############################


        
        # number_density = counts/(T*box_area*dz)
        # mass_density =  29.915 *number_density



        if z_min is None:
            z_min = np.min(all_z_vals) - 0.1 * (np.max(all_z_vals) - np.min(all_z_vals))
        if z_max is None:
            z_max = np.max(all_z_vals) + 0.1 * (np.max(all_z_vals) - np.min(all_z_vals))

        L_x = np.max(all_x_vals) - np.min(all_x_vals)
        L_y = np.max(all_y_vals) - np.min(all_y_vals) 
        dz = (z_max - z_min)


        print(L_x)
        print(L_y, np.max(all_y_vals), np.min(all_y_vals))
        print(z_min, z_max)
        print(np.min(all_z_vals), np.max(all_z_vals))
        print(dz)


        volume = L_x / N_bins * L_y / N_bins * dz

        T = len(sampling_trajectories) 
        print("T", T)
        # N_water = ( len(trajectories[0][0]) - len(substrate) ) /3
        normalization_factor =  29.915 / (  T * volume)  # Convert from PROB DENSITY to g/cm^3
        print(normalization_factor)

        data = [all_x_vals,all_y_vals,all_z_vals]
        data_vs_traj[index] = {'xy':data, 'norm':normalization_factor}

    if return_all_traj_data:
        return data_vs_traj
    else:
        all_x = []
        all_y = []
        all_z = []
        all_norms= []
        for traj_data in data_vs_traj.values():
            data = traj_data['xy']
            all_x.extend(data[0])
            all_y.extend(data[1])
            all_z.extend(data[2])
            normalization_factor = traj_data['norm']
            all_norms.extend([normalization_factor])

        return np.array(all_x), np.array(all_y), np.array(all_z), np.sum(all_norms), slab

















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
        N_bins= 100
        
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

        heatmap, xedges, yedges = np.histogram2d(data[:,0],data[:,1], bins=[N_bins, N_bins],range=[[-L/2, L/2], [-L/2, L/2]],density=True)

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

    average_density = np.zeros((N_bins,N_bins))
    errors = np.zeros((N_bins,N_bins))
    for x in range(N_bins):
        for y in range(N_bins):
            average_density[x][y] = np.mean(number_densities[:,x,y])
            errors[x][y] = np.std(number_densities[:,x,y])/np.sqrt(n_trajectories)

    #Returning densities
    if return_all_profiles:
        return bin_centers_x,bin_centers_y, number_densities
    else:
        return bin_centers_x,bin_centers_y, average_density, errors 


