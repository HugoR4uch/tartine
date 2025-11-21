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

from . import interface_analysis_tools

from ..interfaces_building import water_analyser 


def get_averaged_trajectory(trajectory, averaging_length):
    # Returns trajectory with time averaged coordinates.
    # averaging_length is number of frames to average over
    # If add_tag , adds info to Atoms metadata
    
    
    average_trajectory = copy.deepcopy(trajectory)

    traj_length = len(trajectory)

    atomic_trajectories = interface_analysis_tools.find_atomic_trajectories(trajectory)

    averaged_atomic_trajectories = {}

    for atom_index , atom_trajectory in enumerate(atomic_trajectories): 

        average_atomic_trajectory = []


        for frame_index in range(  traj_length - averaging_length ):
            
            average_pos = np.mean(atom_trajectory[frame_index:frame_index+averaging_length], axis=0)

            average_atomic_trajectory.append(average_pos)

        # print(len(average_atomic_trajectory))
            
        averaged_atomic_trajectories[atom_index] = np.array(average_atomic_trajectory)


    # print(atomic_trajectories[0])
    # print('')
    # print(averaged_atomic_trajectories[0])

    for frame_index , frame in enumerate(average_trajectory):
        
        if frame_index >= traj_length - averaging_length:
            continue


        frame_average_positions = np.array([averaged_atomic_trajectories[key][frame_index] for key in averaged_atomic_trajectories.keys() ])
        
        frame.set_positions(frame_average_positions)

    average_trajectory = average_trajectory[:traj_length - averaging_length]

    return average_trajectory
    # Add averaged atomic trajectories to the trajectory 

    


def get_layer_trajectory(trajectory):

    # Finds all peak widths. For each peak, outputs a trajectory of only peak + substrate. 


    analyser = water_analyser.Analyser(trajectory)

    voronoi_dict = analyser.get_voronoi_dict(1)

    print(voronoi_dict)












def get_contact_layer_only_trajectory(trajectory,species,name,substrate,z_min,z_max=20,num_layers=4):
    
    #NOTE: You need to put in equilibrated trajectories (they will not be truncated in this function)
    #NOTE: THIS ONLY WORKS FOR ORTHORHOMBIC CELLS

  
    number_densities = [] 


    # Get the z density profile
    z_values, density_profile, _ = get_z_density_profile([trajectory], substrate, z_min, z_max)

    # Get the turning points in the z density profile
    turning_points = get_z_density_turning_points(z_values, density_profile)

    print('Turning points: ',turning_points)

    if None in turning_points:
        return None
    else:
        first_non_zero_z, first_trough_z = turning_points
    
    input_trajectory = copy.deepcopy(trajectory) 


    #Finding the interface z-value (so we can select atoms with z_min < z < z_max)
    substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
    all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=True)

    interface_z_mean_traj = []
    for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): # transpose fron (atom index, frame index, coord) to (frame index, atom index, coord index)
        z_vals = frame_positions[:,2]
        interface_z_mean_traj.append(np.mean(z_vals)) 


    # Finding trajectory of contact layer
    new_trajectory = []

    com_traj = find_COM_trajectory(input_trajectory)

    for frame_index, frame in enumerate(input_trajectory):

        frame_z_vals = frame.positions[:, 2]
        frame_z_vals_relative_to_COM = np.array(frame_z_vals) - com_traj[frame_index][2]
        frame_z_vals_relative_to_interface = frame_z_vals_relative_to_COM - interface_z_mean_traj[frame_index]
        valid_indices = np.where((frame_z_vals_relative_to_interface >= z_min) & (frame_z_vals_relative_to_interface <= first_trough_z))[0]
        new_frame = input_trajectory[frame_index][valid_indices]
        new_trajectory.append(new_frame)

    # Write the new trajectory to a file
    return new_trajectory

