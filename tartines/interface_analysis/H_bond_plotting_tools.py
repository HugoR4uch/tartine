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
    


def H_bonds_vs_layer_data_generator(name,
                                   layer_bounds,
                                   substrate,
                                   trajectories,
                                   trajectory_indices=None,
                                   sampling_interval=200,
                                   num_layers=None):
    """
    Generate H-bond vs layer data and save/load H-bond connectivity matrices.
    
    This function processes H-bond trajectories to calculate inter- and intra-layer
    bonding statistics, layer occupancies, water densities, and other metrics.
    It also saves H-bond connectivity matrices to cache for reuse by plot_H_bonds_vs_z.
    
    Parameters:
    -----------
    name : str
        Name identifier for the system
    layer_bounds : dict
        Dictionary mapping layer indices to (z_min, z_max) tuples
    substrate : ase.Atoms
        Substrate structure for calculating xy area
    trajectories : list
        List of trajectory objects to analyze
    trajectory_indices : list, optional
        Specific trajectory indices to include. If None, uses all trajectories
    sampling_interval : int, optional
        Sampling interval for trajectory analysis (default: 200)
    num_layers : int, optional
        Number of layers (currently unused)
        
    Returns:
    --------
    dict
        Dictionary containing all processed data including:
        - Layer bonding matrices
        - Occupancy and density statistics
        - Layer geometry information
        - Combined statistics across trajectories
    """
    import time
    import os
    import json
    
    print('Generating H-bond vs layer data for:', name)
    
    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")
    
    # Create H_bond_trajectories_data directory
    H_bond_trajectories_data_dir = os.path.join("./", "H_bond_trajectories_data")
    if not os.path.exists(H_bond_trajectories_data_dir):
        os.makedirs(H_bond_trajectories_data_dir)
    
    # Data file path for H-bond trajectories
    h_bond_trajectories_data_path = f"{H_bond_trajectories_data_dir}/{name}_H_bond_trajectories_data.json"
    
    start_time = time.time()
    
    # Load existing H_bond_trajectories_data if available
    H_bond_trajectories_data = {}
    if os.path.exists(h_bond_trajectories_data_path):
        try:
            with open(h_bond_trajectories_data_path, 'r') as f:
                H_bond_trajectories_data = json.load(f)
            print(f"Loaded existing H-bond trajectories data from {h_bond_trajectories_data_path}")
        except json.JSONDecodeError:
            print(f"Corrupted H-bond trajectories data file found. Starting fresh.")
            H_bond_trajectories_data = {}
    
    # Initialize new data dictionary for this run
    new_H_bond_trajectories_data = {}
    
    # Initialize data structures
    interlayer_bonding_data = {}
    layer_occupancies_data = {}

    for traj_idx, traj in enumerate(trajectories):
        print(f"Processing trajectory {trajectory_indices[traj_idx]}/{len(trajectories)}")
        
        # Find interface z-values
        substrate_top_layer_indices = interface_analysis_tools.find_top_layer_indices(substrate, num_layers)
        all_top_layer_atom_trajectories = interface_analysis_tools.find_atomic_trajectories(traj, substrate_top_layer_indices, relative_to_COM=False)

        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2):
            substrate_z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(substrate_z_vals))
        
        # Initialize bonding matrix for this trajectory
        interlayer_H_bond_connectivity_directed_matrix = {layer_index: {other_layer_index: 0 for other_layer_index in layer_bounds} for layer_index in layer_bounds}
        
        # Initialize layer occupancies for this trajectory
        actual_traj_idx = trajectory_indices[traj_idx]
        layer_occupancies_data[actual_traj_idx] = {layer_index: {'frame_indices': [], 'occupancies': []} for layer_index in layer_bounds}

        for frame_index, frame in tqdm(enumerate(traj[::sampling_interval]), desc=f"Processing traj {trajectory_indices[traj_idx]} frames"):
            actual_frame_index = frame_index * sampling_interval
            traj_key = str(trajectory_indices[traj_idx])  # Use actual trajectory index, not traj_idx
            frame_key = str(actual_frame_index)
            
            water_O_indices = [atom.index for atom in frame if atom.tag == 1 and atom.symbol == 'O']
            
            # Check if connectivity matrix exists in cached data
            connectivity_matrix_directed = None
            matrix_source = "cached"
            if (traj_key in H_bond_trajectories_data and 
                frame_key in H_bond_trajectories_data[traj_key] and
                'directed_H_bond_connectivity' in H_bond_trajectories_data[traj_key][frame_key]):
                connectivity_matrix_directed = H_bond_trajectories_data[traj_key][frame_key]['directed_H_bond_connectivity']
                if frame_index == 0:  # First frame
                    print(f"🔄 Trajectory {trajectory_indices[traj_idx]}: Using CACHED H-bond matrices")
            else:
                # Compute connectivity matrix
                matrix_source = "computed"
                if frame_index == 0:  # First frame
                    print(f"⚡ Trajectory {trajectory_indices[traj_idx]}: COMPUTING new H-bond matrices")
                analyser = water_analyser.Analyser(frame)
                connectivity_matrix_directed = analyser.get_H_bond_connectivity(directed=True)
                
                # Store in new data dictionary
                if traj_key not in new_H_bond_trajectories_data:
                    new_H_bond_trajectories_data[traj_key] = {}
                new_H_bond_trajectories_data[traj_key][frame_key] = {
                    'frame_indices': actual_frame_index,
                    'directed_H_bond_connectivity': connectivity_matrix_directed
                }
            
            # Debug connectivity matrix every 50 frames
            if frame_index % 50 == 0:
                num_donors = len(connectivity_matrix_directed) if connectivity_matrix_directed else 0
                total_hbonds = sum(sum(acceptors.values()) for acceptors in connectivity_matrix_directed.values()) if connectivity_matrix_directed else 0
                print(f"Frame {actual_frame_index}: Connectivity matrix {matrix_source}, {num_donors} donors, {len(water_O_indices)} waters, {total_hbonds} H-bonds")
            
            positions = frame.get_positions()
            frame_z_vals = positions[water_O_indices][:,2]
            frame_z_vals = [z - interface_z_mean_traj[frame_index] for z in frame_z_vals]

            # Map water O indices to layers for this frame
            O_index_to_layer = {}
            layer_occupancies_frame = {layer_index: 0 for layer_index in layer_bounds}
            
            for i, water_O_idx in enumerate(water_O_indices):
                z_val = frame_z_vals[i]
                for layer_index, layer_range in layer_bounds.items():
                    if layer_range[0] <= z_val < layer_range[1]:
                        O_index_to_layer[water_O_idx] = layer_index
                        layer_occupancies_frame[layer_index] += 1
                        break
            
            # Store layer occupancies for this frame
            for layer_index in layer_bounds:
                layer_occupancies_data[actual_traj_idx][layer_index]['frame_indices'].append(actual_frame_index)
                layer_occupancies_data[actual_traj_idx][layer_index]['occupancies'].append(layer_occupancies_frame[layer_index])
            
            # Count H-bonds between layers
            for water_O_idx in water_O_indices:
                if water_O_idx not in O_index_to_layer:
                    continue  # Skip if water is not in any defined layer
                
                # Handle both integer and string keys (in case data was loaded from JSON)
                donor_key = str(water_O_idx) if str(water_O_idx) in connectivity_matrix_directed else water_O_idx
                if donor_key not in connectivity_matrix_directed:
                    continue  # Skip if water doesn't donate any H-bonds
                
                donor_layer = O_index_to_layer[water_O_idx]
                for acceptor_O_index in connectivity_matrix_directed[donor_key]:
                    hbond_count = connectivity_matrix_directed[donor_key][acceptor_O_index]
                    if hbond_count <= 0:
                        continue  # Skip if no H-bond exists
                    
                    # Handle both integer and string keys for acceptor indices (from JSON loading)
                    acceptor_O_idx_int = int(acceptor_O_index) if isinstance(acceptor_O_index, str) else acceptor_O_index
                    if acceptor_O_idx_int not in O_index_to_layer:
                        continue  # Skip if acceptor is not in any defined layer
                        
                    acceptor_layer = O_index_to_layer[acceptor_O_idx_int]
                    interlayer_H_bond_connectivity_directed_matrix[donor_layer][acceptor_layer] += hbond_count

        # Calculate number of frames processed for this trajectory
        num_frames_this_traj = len([frame for frame in traj[::sampling_interval]])
        
        # Convert to average H-bonds per frame for this trajectory
        for donor_layer in layer_bounds:
            for acceptor_layer in layer_bounds:
                if num_frames_this_traj > 0:
                    interlayer_H_bond_connectivity_directed_matrix[donor_layer][acceptor_layer] /= num_frames_this_traj
        
        interlayer_bonding_data[actual_traj_idx] = interlayer_H_bond_connectivity_directed_matrix
    
    # Update and save H_bond_trajectories_data with any new connectivity matrices
    if new_H_bond_trajectories_data:
        # Merge new data with existing data
        for traj_key, frames_data in new_H_bond_trajectories_data.items():
            if traj_key not in H_bond_trajectories_data:
                H_bond_trajectories_data[traj_key] = {}
            H_bond_trajectories_data[traj_key].update(frames_data)
        
        # Save updated trajectories data (convert numpy types to native Python types for JSON)
        with open(h_bond_trajectories_data_path, 'w') as f:
            json.dump(convert_numpy_types_for_json(H_bond_trajectories_data), f, indent=2)
        
        new_matrices_count = sum(len(frames) for frames in new_H_bond_trajectories_data.values())
        print(f"💾 Updated H-bond trajectories data with {new_matrices_count} new connectivity matrices")
        print(f"📁 Saved H-bond connectivity matrices to {h_bond_trajectories_data_path}")
    else:
        print(f"📁 All H-bond connectivity matrices were loaded from cache")
    
    end_time = time.time()
    print(f"Time taken to generate H bonds vs layer data: {end_time - start_time:.2f} seconds")
    
    # Calculate substrate xy area for density calculations
    substrate_cell = substrate.get_cell()
    xy_area_angstrom2 = np.linalg.norm(np.cross(substrate_cell[0], substrate_cell[1]))
    xy_area_nm2 = xy_area_angstrom2 / 100.0  # Convert Ų to nm²
    
    print(f"Substrate xy area: {xy_area_angstrom2:.2f} Ų = {xy_area_nm2:.4f} nm²")
    
    # Calculate layer center positions (midpoint of layer bounds)
    layer_centers = {}
    for layer_index in sorted(layer_bounds.keys()):
        layer_centers[layer_index] = (layer_bounds[layer_index][0] + layer_bounds[layer_index][1]) / 2
    
    # Create combined bonding matrix (average across trajectories)
    combined_bonding_matrix = {layer_index: {other_layer_index: 0.0 for other_layer_index in layer_bounds} for layer_index in layer_bounds}
    layer_indices = sorted(layer_bounds.keys())
    
    for traj_idx in trajectory_indices:
        bonding_matrix = interlayer_bonding_data[traj_idx]
        for donor_layer in layer_indices:
            for acceptor_layer in layer_indices:
                combined_bonding_matrix[donor_layer][acceptor_layer] += bonding_matrix[donor_layer][acceptor_layer]
    
    # Average across trajectories
    num_trajectories = len(trajectory_indices)
    for donor_layer in layer_indices:
        for acceptor_layer in layer_indices:
            combined_bonding_matrix[donor_layer][acceptor_layer] /= num_trajectories
    
    # Calculate occupancy and density statistics
    occupancy_stats = {}
    density_stats = {}
    for layer_index in sorted(layer_bounds.keys()):
        all_occupancies = []
        all_densities = []
        for traj_idx in trajectory_indices:
            traj_data = layer_occupancies_data[traj_idx]
            occupancies = np.array(traj_data[layer_index]['occupancies'])
            densities = occupancies / xy_area_nm2
            all_occupancies.extend(occupancies.tolist())
            all_densities.extend(densities.tolist())
        
        occupancy_stats[layer_index] = {
            'mean': np.mean(all_occupancies),
            'std': np.std(all_occupancies),
            'min': np.min(all_occupancies),
            'max': np.max(all_occupancies)
        }
        
        density_stats[layer_index] = {
            'mean': np.mean(all_densities),
            'std': np.std(all_densities),
            'min': np.min(all_densities),
            'max': np.max(all_densities)
        }
    
    # Calculate total samples
    total_samples = 0
    for traj_idx in trajectory_indices:
        traj_data = layer_occupancies_data[traj_idx]
        first_layer_key = list(layer_bounds.keys())[0]
        num_frames_this_traj = len(traj_data[first_layer_key]['frame_indices'])
        total_samples += num_frames_this_traj
    
    # Create combined matrix array
    combined_matrix_array = np.zeros((len(layer_indices), len(layer_indices)))
    for row_idx, donor_layer in enumerate(layer_indices):
        for col_idx, acceptor_layer in enumerate(layer_indices):
            combined_matrix_array[row_idx, col_idx] = combined_bonding_matrix[donor_layer][acceptor_layer]
    
    # Note: combined_matrix_array is already averaged, so average_matrix_array is the same
    average_matrix_array = combined_matrix_array.copy()
    
    # Return all necessary data
    return {
        'name': name,
        'layer_bounds': layer_bounds,
        'substrate': substrate,
        'trajectory_indices': trajectory_indices,
        'interlayer_bonding_data': interlayer_bonding_data,
        'layer_occupancies_data': layer_occupancies_data,
        'combined_bonding_matrix': combined_bonding_matrix,
        'combined_matrix_array': combined_matrix_array,
        'average_matrix_array': average_matrix_array,
        'total_samples': total_samples,
        'layer_centers': layer_centers,
        'layer_indices': layer_indices,
        'xy_area_nm2': xy_area_nm2,
        'xy_area_angstrom2': xy_area_angstrom2,
        'occupancy_stats': occupancy_stats,
        'density_stats': density_stats,
    }


def load_H_bonds_vs_layer_data(name, layer_bounds, substrate, trajectory_indices, H_bond_vs_layer_figures_dir='./H_bonds_vs_layer_figures'):
    """
    Load existing H-bond vs layer data from saved files and reconstruct full data dictionary.
    
    Parameters:
    -----------
    name : str
        Name identifier for the system
    layer_bounds : dict
        Dictionary mapping layer indices to (z_min, z_max) tuples
    substrate : ase.Atoms
        Substrate structure for calculating xy area
    trajectory_indices : list
        List of trajectory indices that were analyzed
    H_bond_vs_layer_figures_dir : str, optional
        Base directory for data files (default: './H_bonds_vs_layer_figures')
        
    Returns:
    --------
    dict or False
        Complete data dictionary compatible with plot_H_bonds_vs_layer if all required files exist,
        False if any required files are missing
    """
    
    # Create system-specific subdirectories
    system_plots_dir = os.path.join(H_bond_vs_layer_figures_dir, f"{name}_H-bond_layer_analysis_plots")
    system_data_dir = os.path.join(H_bond_vs_layer_figures_dir, f"{name}_H-bond_layer_analysis_data")
    
    # Data file paths
    layer_data_path = f"{system_data_dir}/{name}_interlayer_H_bond_data.json"
    layer_occupancies_data_path = f"{system_data_dir}/{name}_layer_occupancies_data.json"
    stats_path = f"{system_data_dir}/{name}_layer_density_statistics.json"
    
    # Check if required files exist
    if not os.path.exists(layer_data_path):
        print(f"Layer data file not found: {layer_data_path}")
        return False
    
    if not os.path.exists(layer_occupancies_data_path):
        print(f"Layer occupancies data file not found: {layer_occupancies_data_path}")
        return False
    
    try:
        # Load layer bonding data
        with open(layer_data_path, 'r') as f:
            interlayer_bonding_data = json.load(f)
        
        # Convert string keys back to integers for trajectory indices
        interlayer_bonding_data = {int(k): v for k, v in interlayer_bonding_data.items()}
        
        # Load layer occupancies data
        with open(layer_occupancies_data_path, 'r') as f:
            layer_occupancies_data = json.load(f)
        
        # Convert string keys back to integers for trajectory indices
        layer_occupancies_data = {int(k): v for k, v in layer_occupancies_data.items()}
        
        # Load statistics if available
        occupancy_stats = {}
        density_stats = {}
        xy_area_angstrom2 = None
        xy_area_nm2 = None
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
            occupancy_stats = stats_data.get('occupancy_stats', {})
            density_stats = stats_data.get('density_stats', {})
            xy_area_angstrom2 = stats_data.get('xy_area_angstrom2')
            xy_area_nm2 = stats_data.get('xy_area_nm2')
        
        # If statistics weren't loaded, calculate them from substrate
        if xy_area_angstrom2 is None or xy_area_nm2 is None:
            substrate_cell = substrate.get_cell()
            xy_area_angstrom2 = np.linalg.norm(np.cross(substrate_cell[0], substrate_cell[1]))
            xy_area_nm2 = xy_area_angstrom2 / 100.0
        
        # Calculate layer indices and centers
        layer_indices = sorted(layer_bounds.keys())
        layer_centers = {}
        for layer_index in layer_indices:
            layer_centers[layer_index] = (layer_bounds[layer_index][0] + layer_bounds[layer_index][1]) / 2
        
        # Create combined bonding matrix (average across trajectories)
        combined_bonding_matrix = {layer_index: {other_layer_index: 0.0 for other_layer_index in layer_bounds} for layer_index in layer_bounds}
        
        for traj_idx in trajectory_indices:
            if traj_idx in interlayer_bonding_data:
                bonding_matrix = interlayer_bonding_data[traj_idx]
                for donor_layer in layer_indices:
                    for acceptor_layer in layer_indices:
                        # Handle both integer and string keys
                        donor_key = str(donor_layer) if str(donor_layer) in bonding_matrix else donor_layer
                        acceptor_key = str(acceptor_layer) if str(acceptor_layer) in bonding_matrix[donor_key] else acceptor_layer
                        combined_bonding_matrix[donor_layer][acceptor_layer] += bonding_matrix[donor_key][acceptor_key]
        
        # Average across trajectories
        num_trajectories = len([idx for idx in trajectory_indices if idx in interlayer_bonding_data])
        if num_trajectories > 0:
            for donor_layer in layer_indices:
                for acceptor_layer in layer_indices:
                    combined_bonding_matrix[donor_layer][acceptor_layer] /= num_trajectories
        
        # Calculate total samples
        total_samples = 0
        for traj_idx in trajectory_indices:
            if traj_idx in layer_occupancies_data:
                traj_data = layer_occupancies_data[traj_idx]
                first_layer_key = str(list(layer_bounds.keys())[0]) if str(list(layer_bounds.keys())[0]) in traj_data else list(layer_bounds.keys())[0]
                if first_layer_key in traj_data:
                    num_frames_this_traj = len(traj_data[first_layer_key]['frame_indices'])
                    total_samples += num_frames_this_traj
        
        # Create combined matrix array
        combined_matrix_array = np.zeros((len(layer_indices), len(layer_indices)))
        for row_idx, donor_layer in enumerate(layer_indices):
            for col_idx, acceptor_layer in enumerate(layer_indices):
                combined_matrix_array[row_idx, col_idx] = combined_bonding_matrix[donor_layer][acceptor_layer]
        
        average_matrix_array = combined_matrix_array.copy()
        
        # Calculate occupancy and density statistics if not loaded
        if not occupancy_stats or not density_stats:
            occupancy_stats = {}
            density_stats = {}
            for layer_index in sorted(layer_bounds.keys()):
                all_occupancies = []
                all_densities = []
                for traj_idx in trajectory_indices:
                    if traj_idx in layer_occupancies_data:
                        traj_data = layer_occupancies_data[traj_idx]
                        layer_key = str(layer_index) if str(layer_index) in traj_data else layer_index
                        if layer_key in traj_data:
                            occupancies = np.array(traj_data[layer_key]['occupancies'])
                            densities = occupancies / xy_area_nm2
                            all_occupancies.extend(occupancies.tolist())
                            all_densities.extend(densities.tolist())
                
                if all_occupancies:
                    occupancy_stats[layer_index] = {
                        'mean': np.mean(all_occupancies),
                        'std': np.std(all_occupancies),
                        'min': np.min(all_occupancies),
                        'max': np.max(all_occupancies)
                    }
                    
                    density_stats[layer_index] = {
                        'mean': np.mean(all_densities),
                        'std': np.std(all_densities),
                        'min': np.min(all_densities),
                        'max': np.max(all_densities)
                    }
        
        print(f"Successfully loaded H-bond layer data for {name}")
        
        # Return complete data dictionary compatible with plot_H_bonds_vs_layer
        return {
            'name': name,
            'layer_bounds': layer_bounds,
            'substrate': substrate,
            'trajectory_indices': trajectory_indices,
            'interlayer_bonding_data': interlayer_bonding_data,
            'layer_occupancies_data': layer_occupancies_data,
            'combined_bonding_matrix': combined_bonding_matrix,
            'combined_matrix_array': combined_matrix_array,
            'average_matrix_array': average_matrix_array,
            'total_samples': total_samples,
            'layer_centers': layer_centers,
            'layer_indices': layer_indices,
            'xy_area_nm2': xy_area_nm2,
            'xy_area_angstrom2': xy_area_angstrom2,
            'occupancy_stats': occupancy_stats,
            'density_stats': density_stats,
            'system_plots_dir': system_plots_dir,
            'system_data_dir': system_data_dir,
            'save_data': True
        }
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading H-bond layer data: {e}")
        return False


def save_H_bonds_vs_layer_data(data_dict, H_bond_vs_layer_figures_dir='./H_bonds_vs_layer_figures'):
    """
    Save H-bond vs layer data to files.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data from H_bonds_vs_layer_data_generator()
    H_bond_vs_layer_figures_dir : str, optional
        Base directory for output files (default: './H_bonds_vs_layer_figures')
    """
    
    name = data_dict['name']
    layer_bounds = data_dict['layer_bounds']
    substrate = data_dict['substrate']
    trajectory_indices = data_dict['trajectory_indices']
    interlayer_bonding_data = data_dict['interlayer_bonding_data']
    layer_occupancies_data = data_dict['layer_occupancies_data']
    combined_bonding_matrix = data_dict['combined_bonding_matrix']
    combined_matrix_array = data_dict['combined_matrix_array']
    layer_indices = data_dict['layer_indices']
    xy_area_nm2 = data_dict['xy_area_nm2']
    xy_area_angstrom2 = data_dict['xy_area_angstrom2']
    occupancy_stats = data_dict['occupancy_stats']
    density_stats = data_dict.get('density_stats', {})  # Make optional since plotting functions calculate their own
    total_samples = data_dict['total_samples']
    
    # Create system-specific subdirectories
    system_plots_dir = os.path.join(H_bond_vs_layer_figures_dir, f"{name}_H-bond_layer_analysis_plots")
    system_data_dir = os.path.join(H_bond_vs_layer_figures_dir, f"{name}_H-bond_layer_analysis_data")
    
    if not os.path.exists(system_plots_dir):
        os.makedirs(system_plots_dir)
    
    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir)
    
    # Data file paths
    layer_data_path = f"{system_data_dir}/{name}_interlayer_H_bond_data.json"
    layer_occupancies_data_path = f"{system_data_dir}/{name}_layer_occupancies_data.json"
    
    # Save interlayer bonding data
    with open(layer_data_path, 'w') as f:
        json.dump(convert_numpy_types_for_json(interlayer_bonding_data), f, indent=2)
    print(f"Saved interlayer bonding data to {layer_data_path}")
    
    # Save layer occupancies data
    with open(layer_occupancies_data_path, 'w') as f:
        json.dump(convert_numpy_types_for_json(layer_occupancies_data), f, indent=2)
    print(f"Saved layer occupancies data to {layer_occupancies_data_path}")
    
    # Save individual trajectory layer occupancy CSV files
    for traj_idx in trajectory_indices:
        traj_data = layer_occupancies_data[traj_idx]
        
        # Prepare data for CSV - get all frame indices (should be same for all layers)
        first_layer_key = list(layer_bounds.keys())[0]
        frame_indices = traj_data[first_layer_key]['frame_indices']
        
        # Create CSV data array
        csv_data = [frame_indices]
        csv_header = ['frame_index']
        
        for layer_index in sorted(layer_bounds.keys()):
            csv_data.append(traj_data[layer_index]['occupancies'])
            csv_header.append(f'layer_{layer_index}_occupancy')
        
        # Transpose to get rows as frames, columns as [frame_index, layer_0_occ, layer_1_occ, ...]
        csv_data = np.array(csv_data).T
        
        # Save CSV
        csv_path = f"{system_data_dir}/{name}_traj_{traj_idx}_layer_occupancies.csv"
        np.savetxt(csv_path, csv_data, delimiter=',', fmt='%d', 
                  header=','.join(csv_header), comments='')
        print(f"Saved layer occupancies CSV for trajectory {traj_idx} to {csv_path}")
    
    # Save density and occupancy statistics
    stats_path = f"{system_data_dir}/{name}_layer_density_statistics.json"
    stats_data = {
        'xy_area_angstrom2': xy_area_angstrom2,
        'xy_area_nm2': xy_area_nm2,
        'density_stats': density_stats,
        'occupancy_stats': occupancy_stats
    }
    with open(stats_path, 'w') as f:
        json.dump(convert_numpy_types_for_json(stats_data), f, indent=2)
    print(f"Saved density and occupancy statistics to {stats_path}")
    
    # Save inter/intra bonding statistics
    layer_centers = data_dict['layer_centers']
    inter_intra_stats = {}
    for layer_index in sorted(layer_bounds.keys()):
        layer_center_z = layer_centers[layer_index]
        total_donated = sum(combined_bonding_matrix[layer_index][target_layer] for target_layer in layer_bounds.keys())
        intra_layer_bonds = combined_bonding_matrix[layer_index][layer_index]
        
        # Adjacent layer bonds
        left_layer_bonds = 0
        right_layer_bonds = 0
        sorted_layers = sorted(layer_bounds.keys())
        layer_position = sorted_layers.index(layer_index)
        
        if layer_position > 0:
            left_layer_index = sorted_layers[layer_position - 1]
            left_layer_bonds = combined_bonding_matrix[layer_index][left_layer_index]
            
        if layer_position < len(sorted_layers) - 1:
            right_layer_index = sorted_layers[layer_position + 1]
            right_layer_bonds = combined_bonding_matrix[layer_index][right_layer_index]
        
        inter_intra_stats[layer_index] = {
            'layer_center_z': layer_center_z,
            'layer_bounds': layer_bounds[layer_index],
            'total_donated': total_donated,
            'intra_layer_bonds': intra_layer_bonds,
            'left_layer_bonds': left_layer_bonds,
            'right_layer_bonds': right_layer_bonds,
            'inter_layer_bonds': total_donated - intra_layer_bonds,
            'intra_fraction': intra_layer_bonds / total_donated if total_donated > 0 else 0,
            'inter_fraction': (total_donated - intra_layer_bonds) / total_donated if total_donated > 0 else 0
        }
    
    inter_intra_stats_path = f"{system_data_dir}/{name}_inter_intra_layer_H_bond_statistics.json"
    with open(inter_intra_stats_path, 'w') as f:
        json.dump(convert_numpy_types_for_json(inter_intra_stats), f, indent=2)
    print(f"Saved inter/intra layer H-bond statistics to {inter_intra_stats_path}")
    
    # Save individual trajectory density CSV files
    for traj_idx in trajectory_indices:
        traj_data = layer_occupancies_data[traj_idx]
        
        # Prepare data for CSV - get all frame indices (should be same for all layers)
        first_layer_key = list(layer_bounds.keys())[0]
        frame_indices = traj_data[first_layer_key]['frame_indices']
        
        # Create CSV data array
        csv_data = [frame_indices]
        csv_header = ['frame_index']
        
        for layer_index in sorted(layer_bounds.keys()):
            occupancies = np.array(traj_data[layer_index]['occupancies'])
            densities = occupancies / xy_area_nm2  # Convert to molecules/nm²
            csv_data.append(densities.tolist())
            csv_header.append(f'layer_{layer_index}_density_per_nm2')
        
        # Transpose to get rows as frames, columns as [frame_index, layer_0_density, layer_1_density, ...]
        csv_data = np.array(csv_data).T
        
        # Save CSV
        csv_path = f"{system_data_dir}/{name}_traj_{traj_idx}_layer_water_densities.csv"
        np.savetxt(csv_path, csv_data, delimiter=',', fmt='%g', 
                  header=','.join(csv_header), comments='')
        print(f"Saved layer water densities CSV for trajectory {traj_idx} to {csv_path}")
    
    # Save substrate area info
    area_info_path = f"{system_data_dir}/{name}_substrate_area_info.txt"
    substrate_cell = substrate.get_cell()
    with open(area_info_path, 'w') as f:
        f.write(f"Substrate xy area information for {name}:\n")
        f.write(f"Area in Ų: {xy_area_angstrom2:.6f}\n")
        f.write(f"Area in nm²: {xy_area_nm2:.6f}\n")
        f.write(f"Lattice vectors:\n")
        f.write(f"a: {substrate_cell[0]}\n")
        f.write(f"b: {substrate_cell[1]}\n")
        f.write(f"c: {substrate_cell[2]}\n")
    print(f"Saved substrate area information to {area_info_path}")
    
    # Save matrices as CSV files (average H-bonds per frame)
    for traj_idx in trajectory_indices:
        bonding_matrix = interlayer_bonding_data[traj_idx]
        matrix_array = np.zeros((len(layer_indices), len(layer_indices)))
        for row_idx, donor_layer in enumerate(layer_indices):
            for col_idx, acceptor_layer in enumerate(layer_indices):
                matrix_array[row_idx, col_idx] = bonding_matrix[donor_layer][acceptor_layer]
        
        np.savetxt(f"{system_data_dir}/{name}_traj_{traj_idx}_interlayer_H_bond_matrix.csv", 
                  matrix_array, delimiter=',', fmt='%.6f',
                  header=f"Average inter-interfacial layer H-bond matrix (per frame) for trajectory {traj_idx}. Rows=Donor interfacial layers, Cols=Acceptor interfacial layers. Layer order: {layer_indices}")
    
    # Save combined matrix
    np.savetxt(f"{system_data_dir}/{name}_combined_interlayer_H_bond_matrix.csv", 
              combined_matrix_array, delimiter=',', fmt='%.6f',
              header=f"Combined average inter-interfacial layer H-bond matrix (per frame). Total samples: {total_samples} from {len(trajectory_indices)} trajectories. Rows=Donor interfacial layers, Cols=Acceptor interfacial layers. Layer order: {layer_indices}")
    
    # Save average matrix (alias for backwards compatibility)
    np.savetxt(f"{system_data_dir}/{name}_average_interlayer_H_bond_matrix.csv", 
              combined_matrix_array, delimiter=',', fmt='%.6f',
              header=f"Average inter-interfacial layer H-bond matrix (per frame). Total samples: {total_samples} from {len(trajectory_indices)} trajectories. Rows=Donor interfacial layers, Cols=Acceptor interfacial layers. Layer order: {layer_indices}")
    
    # Save sample count information
    sample_info = {
        'total_samples': total_samples,
        'num_trajectories': len(trajectory_indices),
        'trajectory_indices': trajectory_indices,
        'samples_per_trajectory': {}
    }
    
    for traj_idx in trajectory_indices:
        traj_data = layer_occupancies_data[traj_idx]
        first_layer_key = list(layer_bounds.keys())[0]
        num_frames_this_traj = len(traj_data[first_layer_key]['frame_indices'])
        sample_info['samples_per_trajectory'][traj_idx] = num_frames_this_traj
        
    with open(f"{system_data_dir}/{name}_sampling_info.json", 'w') as f:
        json.dump(sample_info, f, indent=2)
    print(f"Saved sampling information: {total_samples} total samples from {len(trajectory_indices)} trajectories")
    
    # Add system directories to return data
    return {
        'system_plots_dir': system_plots_dir,
        'system_data_dir': system_data_dir,
        'save_data': True
    }


def get_H_bonds_vs_layer_data_and_plot(name, layer_bounds, substrate, trajectories, 
                                      trajectory_indices=None, H_bond_vs_layer_figures_dir='./H_bonds_vs_layer_figures',
                                      sampling_interval=200, num_layers=None, 
                                      save_data=True, load_data=True):
    """
    Complete workflow function that handles loading, generating, saving, and plotting H-bond layer data.
    
    This function combines all the separate functions into a single convenient interface that:
    1. Tries to load existing data
    2. Generates new data if loading fails or load_data=False
    3. Saves the data if save_data=True
    4. Creates all the plots
    
    Parameters:
    -----------
    name : str
        Name identifier for the system
    layer_bounds : dict
        Dictionary mapping layer indices to (z_min, z_max) tuples
    substrate : ase.Atoms
        Substrate structure for calculating xy area
    trajectories : list
        List of trajectory objects to analyze
    trajectory_indices : list, optional
        Specific trajectory indices to include. If None, uses all trajectories
    H_bond_vs_layer_figures_dir : str, optional
        Base directory for output files (default: './H_bonds_vs_layer_figures')
    sampling_interval : int, optional
        Sampling interval for trajectory analysis (default: 200)  
    num_layers : int, optional
        Number of layers (currently unused)
    save_data : bool, optional
        Whether to save processed data to files (default: True)
    load_data : bool, optional
        Whether to try loading existing data first (default: True)
        
    Returns:
    --------
    dict
        Complete data dictionary with all processed results and file paths
    """
    
    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")
    
    # Create output directory
    if not os.path.isdir(H_bond_vs_layer_figures_dir):
        os.makedirs(H_bond_vs_layer_figures_dir)

    data_dict = None
    
    # Try to load existing data first
    if load_data:
        print("Attempting to load existing H-bond layer data...")
        data_dict = load_H_bonds_vs_layer_data(name, layer_bounds, substrate, trajectory_indices, H_bond_vs_layer_figures_dir)
        
        if data_dict is not False:
            print("Successfully loaded existing data!")
        else:
            print("Loading failed or no existing data found.")
    
    # Generate new data if loading failed or was disabled
    if data_dict is False or data_dict is None:
        print("Generating new H-bond layer data...")
        data_dict = H_bonds_vs_layer_data_generator(name, layer_bounds, substrate, trajectories, 
                                                   trajectory_indices, sampling_interval, num_layers)
        
        # Save the generated data if requested
        if save_data:
            print("Saving generated data...")
            save_info = save_H_bonds_vs_layer_data(data_dict, H_bond_vs_layer_figures_dir)
            # Add save info to data dict for plotting
            data_dict.update(save_info)
        else:
            # Still need directory info for plotting even if not saving
            system_plots_dir = os.path.join(H_bond_vs_layer_figures_dir, f"{name}_H-bond_layer_analysis_plots")
            system_data_dir = os.path.join(H_bond_vs_layer_figures_dir, f"{name}_H-bond_layer_analysis_data")
            if not os.path.exists(system_plots_dir):
                os.makedirs(system_plots_dir)
            data_dict.update({
                'system_plots_dir': system_plots_dir,
                'system_data_dir': system_data_dir,
                'save_data': save_data
            })
    
    # Create the plots
    print("Creating H-bond layer plots...")
    plot_H_bonds_vs_layer(data_dict)
    
    print(f"H-bond layer analysis complete! Plots saved to: {data_dict['system_plots_dir']}")
    return data_dict


def plot_molecules_per_area_vs_time(data_dict):
    """
    Plot water molecules per area (density) vs time for each layer.
    Creates both individual trajectory subplots and combined plots.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data
    """
    # Extract data from dictionary
    name = data_dict['name']
    layer_bounds = data_dict['layer_bounds']
    substrate = data_dict['substrate']
    trajectory_indices = data_dict['trajectory_indices']
    layer_occupancies_data = data_dict['layer_occupancies_data']
    xy_area_nm2 = data_dict['xy_area_nm2']
    xy_area_angstrom2 = data_dict['xy_area_angstrom2']
    system_plots_dir = data_dict['system_plots_dir']
    system_data_dir = data_dict['system_data_dir']
    save_data = data_dict['save_data']
    
    num_trajectories = len(trajectory_indices)
    
    
    
    # Calculate density statistics for the bar plot
    calculated_density_stats = {}
    for layer_index in sorted(layer_bounds.keys()):
        all_densities = []
        for traj_idx in trajectory_indices:
            traj_data = layer_occupancies_data[traj_idx]
            # Handle both integer and string keys for layer indices
            layer_key = str(layer_index) if str(layer_index) in traj_data else layer_index
            occupancies = np.array(traj_data[layer_key]['occupancies'])
            densities = occupancies / xy_area_nm2
            all_densities.extend(densities.tolist())
        
        calculated_density_stats[layer_index] = {
            'mean': np.mean(all_densities),
            'std': np.std(all_densities),
            'min': np.min(all_densities),
            'max': np.max(all_densities)
        }
    print("Calculated density statistics:")
    for layer_index, stats in calculated_density_stats.items():
        print(f"Layer {layer_index}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}, Min = {stats['min']:.4f}, Max = {stats['max']:.4f}")

    print(f"Substrate xy area: {xy_area_angstrom2:.2f} Ų = {xy_area_nm2:.4f} nm²")
    
    # Plot 1: Layer water density (molecules/nm²) vs time for each trajectory (subplots)
    fig, axes = plt.subplots(1, num_trajectories, figsize=(6*num_trajectories, 5))
    if num_trajectories == 1:
        axes = [axes]
    
    for i, traj_idx in enumerate(trajectory_indices):
        traj_data = layer_occupancies_data[traj_idx]
        
        for layer_index in sorted(layer_bounds.keys()):
            # Handle both integer and string keys for layer indices
            layer_key = str(layer_index) if str(layer_index) in traj_data else layer_index
            frame_indices = np.array(traj_data[layer_key]['frame_indices'])
            occupancies = np.array(traj_data[layer_key]['occupancies'])
            densities = occupancies / xy_area_nm2  # Convert to molecules/nm²
            
            axes[i].plot(frame_indices, densities, 'o-', 
                        label=f'Layer {layer_index} Density:{calculated_density_stats[layer_index]["mean"]:.2f}±{calculated_density_stats[layer_index]["std"]:.2f} Water/nm²', 
                        markersize=2, linewidth=1)
        
        axes[i].set_xlabel('Frame Index')
        axes[i].set_ylabel('Water Density (molecules/nm²)')
        axes[i].set_title(f'Traj {traj_idx}: Interfacial Layer Water Densities vs Time')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_layer_water_densities_by_traj.png', dpi=300)
    plt.close()
    


    # Save density data as CSV files
    if save_data:
        # Save individual trajectory density CSV files
        for traj_idx in trajectory_indices:
            traj_data = layer_occupancies_data[traj_idx]
            
            # Prepare data for CSV - get all frame indices (should be same for all layers)
            first_layer_key = str(list(layer_bounds.keys())[0]) if str(list(layer_bounds.keys())[0]) in traj_data else list(layer_bounds.keys())[0]
            frame_indices = traj_data[first_layer_key]['frame_indices']
            
            # Create CSV data array
            csv_data = [frame_indices]
            csv_header = ['frame_index']
            
            for layer_index in sorted(layer_bounds.keys()):
                # Handle both integer and string keys for layer indices
                layer_key = str(layer_index) if str(layer_index) in traj_data else layer_index
                occupancies = np.array(traj_data[layer_key]['occupancies'])
                densities = occupancies / xy_area_nm2  # Convert to molecules/nm²
                csv_data.append(densities.tolist())
                csv_header.append(f'layer_{layer_index}_density_per_nm2')
            
            # Transpose to get rows as frames, columns as [frame_index, layer_0_density, layer_1_density, ...]
            csv_data = np.array(csv_data).T
            
            # Save CSV
            csv_path = f"{system_data_dir}/{name}_traj_{traj_idx}_layer_water_densities.csv"
            np.savetxt(csv_path, csv_data, delimiter=',', fmt='%g', 
                      header=','.join(csv_header), comments='')
            print(f"Saved layer water densities CSV for trajectory {traj_idx} to {csv_path}")
        
        # Save substrate area info
        area_info_path = f"{system_data_dir}/{name}_substrate_area_info.txt"
        substrate_cell = substrate.get_cell()
        with open(area_info_path, 'w') as f:
            f.write(f"Substrate xy area information for {name}:\n")
            f.write(f"Area in Ų: {xy_area_angstrom2:.6f}\n")
            f.write(f"Area in nm²: {xy_area_nm2:.6f}\n")
            f.write(f"Lattice vectors:\n")
            f.write(f"a: {substrate_cell[0]}\n")
            f.write(f"b: {substrate_cell[1]}\n")
            f.write(f"c: {substrate_cell[2]}\n")
        print(f"Saved substrate area information to {area_info_path}")
        
        # Save density statistics
        stats_path = f"{system_data_dir}/{name}_layer_density_statistics.json"
        stats_data = {
            'xy_area_angstrom2': xy_area_angstrom2,
            'xy_area_nm2': xy_area_nm2,
            'density_stats': calculated_density_stats,
        }
        with open(stats_path, 'w') as f:
            json.dump(convert_numpy_types_for_json(stats_data), f, indent=2)
        print(f"Saved density statistics to {stats_path}")


def plot_h_bond_breakdown_per_layer(data_dict):
    """
    Plot H-bond breakdown per layer using improved styling.
    Shows H-bonds per water molecule with neighboring layers and intra-layer bonds.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data
    """
    # Extract data from dictionary
    name = data_dict['name']
    layer_bounds = data_dict['layer_bounds']
    combined_bonding_matrix = data_dict['combined_bonding_matrix']
    occupancy_stats = data_dict['occupancy_stats']
    trajectory_indices = data_dict['trajectory_indices']
    system_plots_dir = data_dict['system_plots_dir']

    # The combined_bonding_matrix is already averaged, so we can use it directly
    averaged_bonding_matrix = combined_bonding_matrix

    # IMPROVED VERSION: Better styling and automatic sizing
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get actual layer numbers and process interface layers (all except last which is bulk)
    sorted_layers = sorted(layer_bounds.keys())
    interface_layers = sorted_layers[:-1] if len(sorted_layers) > 1 else sorted_layers
    x_positions = np.arange(len(interface_layers))
    bar_width = 0.2  # Slightly narrower bars for better proportions

    # Improved colors with better contrast
    color_left = '#FF6B6B'      # Modern red for bonds with previous layer
    color_center = '#4ECDC4'    # Modern teal for intra-layer bonds 
    color_right = '#45B7D1'     # Modern blue for bonds with next layer

    print('DEBUGGING')
    print('average bonding:', averaged_bonding_matrix)

    for i, layer_index in enumerate(interface_layers):
        # Get mean occupancy for this layer
        print()
        print('Layer: ', layer_index)
        mean_occupancy = occupancy_stats[layer_index]['mean']
        print('Mean occupancy: ', mean_occupancy)
        
        # Calculate H-bonds per water molecule using new algorithm
        # Intra-layer bonds: (connectivity_matrix[i][i] + connectivity_matrix[i][i]) / n_i
        intra_bonds = (2 * averaged_bonding_matrix[layer_index][layer_index]) / mean_occupancy
        print('Intra-layer bonds per molecule: ', intra_bonds)
        
        # Bonds with neighboring layers
        left_bonds = 0
        right_bonds = 0
        layer_position = sorted_layers.index(layer_index)
        
        if layer_position > 0:  # Has previous layer
            prev_layer_index = sorted_layers[layer_position - 1]
            # Shared bonds: (connectivity_matrix[i][j] + connectivity_matrix[j][i]) / n_i
            left_bonds = (averaged_bonding_matrix[layer_index][prev_layer_index] + 
                         averaged_bonding_matrix[prev_layer_index][layer_index]) / mean_occupancy
            print('Left bonds per molecule: ', left_bonds)
        
        if layer_position < len(sorted_layers) - 1:  # Has next layer
            next_layer_index = sorted_layers[layer_position + 1]
            # Shared bonds: (connectivity_matrix[i][j] + connectivity_matrix[j][i]) / n_i
            right_bonds = (averaged_bonding_matrix[layer_index][next_layer_index] + 
                          averaged_bonding_matrix[next_layer_index][layer_index]) / mean_occupancy
            print('Right bonds per molecule: ', right_bonds)
        
        # Plot bars with improved styling
        x_center = x_positions[i]
        
        # Left bar (bonds with previous layer) - only if not first layer
        if layer_position > 0:
            ax.bar(x_center - bar_width, left_bonds, bar_width, 
                   color=color_left, alpha=0.8, edgecolor='white', linewidth=1,
                   label='Bonds with Previous Layer' if i == 1 else "")
        
        # Center bar (intra-layer bonds)
        ax.bar(x_center, intra_bonds, bar_width, 
               color=color_center, alpha=0.8, edgecolor='white', linewidth=1,
               label='Intra-layer H-bonds' if i == 0 else "")
        
        # Right bar (bonds with next layer)
        if layer_position < len(sorted_layers) - 1:  # Only if not the last layer
            ax.bar(x_center + bar_width, right_bonds, bar_width, 
                   color=color_right, alpha=0.8, edgecolor='white', linewidth=1,
                   label='Bonds with Next Layer' if i == 0 else "")
        
        # Add value labels with better positioning and styling
        label_offset = max(intra_bonds, left_bonds, right_bonds) * 0.02  # Dynamic offset
        
        if layer_position > 0:  # Left bar label
            ax.text(x_center - bar_width, left_bonds + label_offset, f'{left_bonds:.2f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_left)
        
        # Center bar label
        ax.text(x_center, intra_bonds + label_offset, f'{intra_bonds:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_center)
        
        # Right bar label
        if layer_position < len(sorted_layers) - 1:
            ax.text(x_center + bar_width, right_bonds + label_offset, f'{right_bonds:.2f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_right)

    # Add horizontal dashed line for bulk intra-layer H-bonds per molecule (if bulk layer exists)
    if len(sorted_layers) > len(interface_layers):
        bulk_layer_index = sorted_layers[-1]
        bulk_occupancy = occupancy_stats[bulk_layer_index]['mean']
        bulk_intra_per_molecule = (2 * averaged_bonding_matrix[bulk_layer_index][bulk_layer_index]) / bulk_occupancy

        ax.axhline(y=bulk_intra_per_molecule, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Bulk Region: {bulk_intra_per_molecule:.2f} H-bonds/molecule')

    # Customize plot with better styling
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('H-bonds per Water Molecule', fontsize=12, fontweight='bold')
    ax.set_title(f'{name}: H-Bond Distribution per Water Molecule by Layer', fontsize=14, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xticks(x_positions)
    layer_labels = [f'Layer {idx}' if idx != sorted_layers[-1] else 'Bulk Region' for idx in interface_layers]
    ax.set_xticklabels(layer_labels, fontsize=11)

    # Improve grid and styling
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind bars

    # Better legend positioning and styling
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                       fontsize=10, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Let matplotlib handle the y-axis limits automatically with some padding
    ax.margins(y=0.1)

    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_H_bond_breakdown_per_layer.png', dpi=300)
    plt.close()

    # Print summary
    print(f"\nH-Bond Distribution Summary for {name}:")
    print("=" * 50)
    for layer_index in sorted_layers:
        mean_occupancy = occupancy_stats[layer_index]['mean']
        intra_bonds = (2 * averaged_bonding_matrix[layer_index][layer_index]) / mean_occupancy
        
        layer_name = f"Layer {layer_index}" if layer_index != sorted_layers[-1] else "Bulk Region"
        print(f"{layer_name}:")
        print(f"  Average occupancy: {mean_occupancy:.1f} molecules")
        print(f"  Intra-layer H-bonds per molecule: {intra_bonds:.2f}")
        
        # Check bonds with other layers
        for target_layer in sorted_layers:
            if layer_index != target_layer:  # Skip self-interactions
                shared_bonds = (averaged_bonding_matrix[layer_index][target_layer] + 
                               averaged_bonding_matrix[target_layer][layer_index]) / mean_occupancy
                if shared_bonds > 0.01:  # Only show significant bonds
                    target_name = f"Layer {target_layer}" if target_layer != sorted_layers[-1] else "Bulk Region"
                    print(f"  Shared bonds with {target_name} per molecule: {shared_bonds:.2f}")
        print()




def plot_bonding_network_graph(data_dict):
    """
    Plot bidirectional H-bond network graph.
    Creates both individual trajectory subplots and combined plots.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data
    """
    # Extract data from dictionary
    name = data_dict['name']
    layer_bounds = data_dict['layer_bounds']
    trajectory_indices = data_dict['trajectory_indices']
    interlayer_bonding_data = data_dict['interlayer_bonding_data']
    combined_bonding_matrix = data_dict['combined_bonding_matrix']
    layer_centers = data_dict['layer_centers']
    layer_indices = data_dict['layer_indices']
    system_plots_dir = data_dict['system_plots_dir']
    save_data = data_dict['save_data']
    
    # Bidirectional H-Bond Network Graph
    try:
        import networkx as nx
        
        # Plot for each trajectory
        for traj_idx in trajectory_indices:
            bonding_matrix = interlayer_bonding_data[traj_idx]
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes (layers)
            for layer_index in sorted(layer_bounds.keys()):
                layer_center_z = layer_centers[layer_index]
                G.add_node(layer_index, 
                          pos=(0, layer_center_z),
                          label=f"Layer {layer_index}\n({layer_bounds[layer_index][0]:.1f}-{layer_bounds[layer_index][1]:.1f} Å)")
            
            # Add edges (H-bonds) with weights
            for donor_layer in layer_indices:
                for acceptor_layer in layer_indices:
                    if donor_layer != acceptor_layer:  # No self-loops for clarity
                        # Handle both integer and string keys
                        donor_key = str(donor_layer) if str(donor_layer) in bonding_matrix else donor_layer
                        acceptor_key = str(acceptor_layer) if str(acceptor_layer) in bonding_matrix[donor_key] else acceptor_layer
                        weight = bonding_matrix[donor_key][acceptor_key]
                        if weight > 0.001:  # Only show significant bonds
                            G.add_edge(donor_layer, acceptor_layer, weight=weight)
            
            # Create plot
            plt.figure(figsize=(10, 12))
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=3000, alpha=0.7)
            
            # Draw node labels
            labels = {node: f"Layer {node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
            
            # Draw edges with varying thickness based on bond strength
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            
            # Normalize weights for edge thickness
            edge_widths = [5 * (w / max_weight) for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                                  edge_color='red', arrows=True, arrowsize=20,
                                  connectionstyle="arc3,rad=0.1")
            
            # Add edge labels with bond strengths
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
            plt.title(f'{name} - Traj {traj_idx}: H-Bond Network Between Interfacial Layers\n'
                     f'(Edge thickness ∝ H-bond strength, Numbers show average H-bonds per frame)')
            plt.ylabel('Z-position (Å)')
            plt.xlabel('H-bond Network')
            
            # Remove x-axis as it's not meaningful
            plt.gca().set_xlim(-1, 1)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().set_xticks([])
            
            plt.tight_layout()
            plt.savefig(system_plots_dir + '/' + name + f'_traj_{traj_idx}_H_bond_network.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Combined network graph
        G_combined = nx.DiGraph()
        
        # Add nodes (layers)
        for layer_index in sorted(layer_bounds.keys()):
            layer_center_z = layer_centers[layer_index]
            G_combined.add_node(layer_index, 
                      pos=(0, layer_center_z),
                      label=f"Layer {layer_index}\n({layer_bounds[layer_index][0]:.1f}-{layer_bounds[layer_index][1]:.1f} Å)")
        
        # Add edges (H-bonds) with weights from combined matrix
        for donor_layer in layer_indices:
            for acceptor_layer in layer_indices:
                if donor_layer != acceptor_layer:  # No self-loops for clarity
                    weight = combined_bonding_matrix[donor_layer][acceptor_layer]
                    if weight > 0.001:  # Only show significant bonds
                        G_combined.add_edge(donor_layer, acceptor_layer, weight=weight)
        
        # Create combined plot
        plt.figure(figsize=(10, 12))
        pos = nx.get_node_attributes(G_combined, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(G_combined, pos, node_color='lightgreen', 
                              node_size=3000, alpha=0.7)
        
        # Draw node labels
        labels = {node: f"Layer {node}" for node in G_combined.nodes()}
        nx.draw_networkx_labels(G_combined, pos, labels, font_size=10, font_weight='bold')
        
        # Draw edges with varying thickness based on bond strength
        edges = G_combined.edges()
        weights = [G_combined[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        # Normalize weights for edge thickness
        edge_widths = [5 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G_combined, pos, width=edge_widths, alpha=0.6, 
                              edge_color='darkred', arrows=True, arrowsize=20,
                              connectionstyle="arc3,rad=0.1")
        
        # Add edge labels with bond strengths
        edge_labels = {(u, v): f"{G_combined[u][v]['weight']:.2f}" for u, v in G_combined.edges()}
        nx.draw_networkx_edge_labels(G_combined, pos, edge_labels, font_size=8)
        
        plt.title(f'{name}: Combined H-Bond Network Between Interfacial Layers\n'
                 f'(Edge thickness ∝ H-bond strength, Numbers show average H-bonds per frame)')
        plt.ylabel('Z-position (Å)')
        plt.xlabel('H-bond Network')
        
        # Remove x-axis as it's not meaningful
        plt.gca().set_xlim(-1, 1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().set_xticks([])
        
        plt.tight_layout()
        plt.savefig(system_plots_dir + '/' + name + '_combined_H_bond_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("NetworkX not available. Skipping H-bond network graph.")


def plot_H_bonds_vs_layer(data_dict):
    """
    Wrapper function that creates all H-bond vs layer visualizations using processed data.
    
    This function calls all the individual plotting functions to create a comprehensive
    set of H-bond layer analysis plots.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data from get_H_bonds_vs_layer_data()
    """
    
    print("Creating H-bond layer analysis plots...")
    
    # Plot 1: Water molecules per area vs time
    print("  - Plotting water density vs time...")
    plot_molecules_per_area_vs_time(data_dict)
    
    # Plot 2: H-bond breakdown per layer
    print("  - Plotting H-bond breakdown per layer...")
    plot_h_bond_breakdown_per_layer(data_dict)
    
    # # Plot 3: Bonding network graph
    # print("  - Plotting bonding network graphs...")
    # plot_bonding_network_graph(data_dict)
    
    print("All H-bond layer analysis plots completed!")








def plot_H_bonds_vs_z(name,
                      substrate,
                      trajectories,
                      trajectory_indices=None,
                      H_bond_vs_z_figures_dir='./H_bonds_vs_z_figures',
                      sampling_interval=200,
                      z_sampling_min=0,
                      z_sampling_max=30,
                      z_interfacial_sampling_max=10,  
                      z_sampling_increment=0.05,
                      save_data=True,
                      load_data=True,
                      ):
    
    print('Calculating H bonds vs z for:', name)
    
    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")
    
    # Create system-specific subdirectories
    system_plots_dir = os.path.join(H_bond_vs_z_figures_dir, f"{name}_H-bond_analysis_plots")
    system_data_dir = os.path.join(H_bond_vs_z_figures_dir, f"{name}_H-bond_analysis_data")
    
    # Create H_bond_trajectories_data directory at the same level as H_bond_vs_z_figures_dir
    H_bond_trajectories_data_dir = os.path.join("./", "H_bond_trajectories_data")
    
    if not os.path.exists(system_plots_dir):
        os.makedirs(system_plots_dir)
    
    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir)
        
    if not os.path.exists(H_bond_trajectories_data_dir):
        os.makedirs(H_bond_trajectories_data_dir)

    # Data file paths
    h_bond_data_path = f"{system_data_dir}/{name}_H_bond_raw_data.json"
    h_bond_trajectories_data_path = f"{H_bond_trajectories_data_dir}/{name}_H_bond_trajectories_data.json"
    
    # Define z bins for analysis
    z_bins = np.arange(z_sampling_min, z_sampling_max + z_sampling_increment, z_sampling_increment)
    z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2
    
    start_time = time.time()
    
    # Load existing H_bond_trajectories_data if available
    H_bond_trajectories_data = {}
    if os.path.exists(h_bond_trajectories_data_path):
        try:
            with open(h_bond_trajectories_data_path, 'r') as f:
                H_bond_trajectories_data = json.load(f)
            print(f"Loaded existing H-bond trajectories data from {h_bond_trajectories_data_path}")
        except json.JSONDecodeError:
            print(f"Corrupted H-bond trajectories data file found. Starting fresh.")
            H_bond_trajectories_data = {}
    
    # Initialize new data dictionary for this run
    new_H_bond_trajectories_data = {}
    
    # Load existing data if requested and available
    if load_data and os.path.exists(h_bond_data_path):
        with open(h_bond_data_path, 'r') as f:
            traj_H_bond_data = json.load(f)
        print(f"Loaded existing H-bond data from {h_bond_data_path}")
    else:
        # Calculate H-bond data for each trajectory
        traj_H_bond_data = {}
        
        for traj_idx, traj in enumerate(trajectories):
            print(f"Processing trajectory {trajectory_indices[traj_idx]}/{len(trajectories)}")
            
            # Find interface z-values
            num_layers = None
            substrate_top_layer_indices = interface_analysis_tools.find_top_layer_indices(substrate, num_layers)
            all_top_layer_atom_trajectories = interface_analysis_tools.find_atomic_trajectories(traj, substrate_top_layer_indices, relative_to_COM=False)

            interface_z_mean_traj = []
            for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2):
                substrate_z_vals = frame_positions[:,2]
                interface_z_mean_traj.append(np.mean(substrate_z_vals))
            
            # Collect H-bond data for this trajectory
            z_vals = []
            total_bonds = []
            accepted_bonds = []
            donated_bonds = []
            
            for frame_index, frame in tqdm(enumerate(traj[::sampling_interval]), desc=f"Processing traj {trajectory_indices[traj_idx]} frames"):
                actual_frame_index = frame_index * sampling_interval
                traj_key = str(trajectory_indices[traj_idx])  # Use actual trajectory index, not traj_idx
                frame_key = str(actual_frame_index)
                
                water_O_indices = [atom.index for atom in frame if atom.tag == 1 and atom.symbol == 'O']
                
                # Check if connectivity matrix exists in cached data
                connectivity_matrix_directed = None
                matrix_source = "cached"
                if (traj_key in H_bond_trajectories_data and 
                    frame_key in H_bond_trajectories_data[traj_key] and
                    'directed_H_bond_connectivity' in H_bond_trajectories_data[traj_key][frame_key]):
                    connectivity_matrix_directed = H_bond_trajectories_data[traj_key][frame_key]['directed_H_bond_connectivity']
                    if frame_index == 0:  # First frame
                        print(f"📁 Trajectory {trajectory_indices[traj_idx]}: Using CACHED H-bond matrices (likely from H_bonds_vs_layer_data_generator)")
                else:
                    # Compute connectivity matrix
                    matrix_source = "computed"
                    if frame_index == 0:  # First frame
                        print(f"⚡ Trajectory {trajectory_indices[traj_idx]}: COMPUTING new H-bond matrices")
                    analyser = water_analyser.Analyser(frame)
                    connectivity_matrix_directed = analyser.get_H_bond_connectivity(directed=True)
                    
                    # Store in new data dictionary
                    if traj_key not in new_H_bond_trajectories_data:
                        new_H_bond_trajectories_data[traj_key] = {}
                    new_H_bond_trajectories_data[traj_key][frame_key] = {
                        'frame_indices': actual_frame_index,
                        'directed_H_bond_connectivity': connectivity_matrix_directed
                    }
                
                # Debug: Check connectivity matrix
                if frame_index % 50 == 0:
                    print(f"  Frame {actual_frame_index}: Connectivity matrix {matrix_source}, {len(connectivity_matrix_directed)} donors, {len(water_O_indices)} waters")
                
                positions = frame.get_positions()
                frame_z_vals = positions[water_O_indices][:,2]
                frame_z_vals = [z - interface_z_mean_traj[frame_index] for z in frame_z_vals]

                # Debug: Track H-bond statistics for this frame
                frame_total_hbonds = 0
                frame_waters_with_hbonds = 0
                
                for i, water_O_idx in enumerate(water_O_indices):
                    z_val = frame_z_vals[i]
                    
                    # Calculate donated bonds: bonds that water_O_idx donates to others
                    donated_bond_count = 0
                    # Handle both integer and string keys (in case data was loaded from JSON)
                    donor_key = str(water_O_idx) if str(water_O_idx) in connectivity_matrix_directed else water_O_idx
                    if donor_key in connectivity_matrix_directed:
                        donated_bond_count = np.sum(list(connectivity_matrix_directed[donor_key].values()))
                    
                    # Calculate accepted bonds: bonds that water_O_idx accepts from others
                    accepted_bond_count = 0
                    for donor_O_idx in water_O_indices:
                        # Handle both integer and string keys for both donor and acceptor
                        donor_key_check = str(donor_O_idx) if str(donor_O_idx) in connectivity_matrix_directed else donor_O_idx
                        if donor_key_check in connectivity_matrix_directed:
                            acceptor_key = str(water_O_idx) if str(water_O_idx) in connectivity_matrix_directed[donor_key_check] else water_O_idx
                            if acceptor_key in connectivity_matrix_directed[donor_key_check]:
                                accepted_bond_count += connectivity_matrix_directed[donor_key_check][acceptor_key]
                    
                    total_bond_count = accepted_bond_count + donated_bond_count
                    
                    # Debug tracking
                    if total_bond_count > 0:
                        frame_waters_with_hbonds += 1
                    frame_total_hbonds += total_bond_count

                    z_vals.append(z_val)
                    total_bonds.append(total_bond_count)
                    accepted_bonds.append(accepted_bond_count)
                    donated_bonds.append(donated_bond_count)
                
                # Debug print every 50 frames
                if frame_index % 50 == 0:
                    print(f"  Frame {actual_frame_index}: {len(water_O_indices)} waters, {frame_waters_with_hbonds} with H-bonds, {frame_total_hbonds} total H-bonds")
                    print(f"    Connectivity matrix has {len(connectivity_matrix_directed)} donors")
                    if len(connectivity_matrix_directed) > 0:
                        sample_donor = list(connectivity_matrix_directed.keys())[0]
                        sample_acceptors = connectivity_matrix_directed[sample_donor]
                        print(f"    Sample donor {sample_donor} (type: {type(sample_donor)}) has {len(sample_acceptors)} acceptors")
                        if len(sample_acceptors) > 0:
                            sample_acceptor = list(sample_acceptors.keys())[0]
                            print(f"    Sample acceptor {sample_acceptor} (type: {type(sample_acceptor)}) with bond count: {sample_acceptors[sample_acceptor]}")
            
            # Store trajectory data (convert numpy types to Python types for JSON compatibility)
            traj_H_bond_data[str(traj_idx)] = {
                'z_vals': [float(z) for z in z_vals],
                'total_bonds': [int(b) for b in total_bonds],
                'accepted_bonds': [int(b) for b in accepted_bonds],
                'donated_bonds': [int(b) for b in donated_bonds]
            }
        
        # Save data if requested
        if save_data:
            with open(h_bond_data_path, 'w') as f:
                json.dump(traj_H_bond_data, f, indent=2)
            print(f"Saved H-bond data to {h_bond_data_path}")
        
        # Update and save H_bond_trajectories_data with new data
        if new_H_bond_trajectories_data:
            # Merge new data with existing data
            for traj_key, frames_data in new_H_bond_trajectories_data.items():
                if traj_key not in H_bond_trajectories_data:
                    H_bond_trajectories_data[traj_key] = {}
                H_bond_trajectories_data[traj_key].update(frames_data)
            
            # Save updated trajectories data
            with open(h_bond_trajectories_data_path, 'w') as f:
                json.dump(convert_numpy_types_for_json(H_bond_trajectories_data), f, indent=2)
            
            new_matrices_count = sum(len(frames) for frames in new_H_bond_trajectories_data.values())
            print(f"💾 plot_H_bonds_vs_z: Updated H-bond trajectories data with {new_matrices_count} new connectivity matrices")
            print(f"📁 Saved H-bond connectivity matrices to {h_bond_trajectories_data_path}")
        else:
            print(f"📁 plot_H_bonds_vs_z: All H-bond connectivity matrices were loaded from cache (likely saved by H_bonds_vs_layer_data_generator)")
    
    # Process and bin the data for each trajectory
    num_trajectories = len(trajectories)
    traj_binned_data = []
    
    for traj_idx in range(num_trajectories):
        traj_data = traj_H_bond_data[str(traj_idx)]
        z_vals = np.array(traj_data['z_vals'])
        total_bonds = np.array(traj_data['total_bonds'])
        accepted_bonds = np.array(traj_data['accepted_bonds'])
        donated_bonds = np.array(traj_data['donated_bonds'])
        
        # Bin the data
        total_bonds_binned, _ = np.histogram(z_vals, bins=z_bins, weights=total_bonds)
        accepted_bonds_binned, _ = np.histogram(z_vals, bins=z_bins, weights=accepted_bonds)
        donated_bonds_binned, _ = np.histogram(z_vals, bins=z_bins, weights=donated_bonds)
        counts_per_bin, _ = np.histogram(z_vals, bins=z_bins)
        
        # Calculate probability density for z values (normalize by bin width and total count)
        bin_width = z_bins[1] - z_bins[0]
        density_z = counts_per_bin / (len(z_vals) * bin_width)
        
        # Calculate mean bonds per bin (avoid division by zero)
        mean_total_bonds = np.divide(total_bonds_binned, counts_per_bin, out=np.zeros_like(total_bonds_binned, dtype=float), where=counts_per_bin!=0)
        mean_accepted_bonds = np.divide(accepted_bonds_binned, counts_per_bin, out=np.zeros_like(accepted_bonds_binned, dtype=float), where=counts_per_bin!=0)
        mean_donated_bonds = np.divide(donated_bonds_binned, counts_per_bin, out=np.zeros_like(donated_bonds_binned, dtype=float), where=counts_per_bin!=0)
        
        # Store binned data
        binned_data = {
            'z_centers': z_bin_centers,
            'total_H_bonds': mean_total_bonds,
            'accepted_H_bonds': mean_accepted_bonds,
            'donated_H_bonds': mean_donated_bonds,
            'density_z': density_z
        }
        traj_binned_data.append(binned_data)
        
        # Save individual trajectory CSV
        if save_data:
            csv_data = np.column_stack([z_bin_centers, mean_total_bonds, mean_accepted_bonds, mean_donated_bonds, density_z])
            np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_H_bond_binned_data.csv", 
                      csv_data, delimiter=',', 
                      header='z_center,total_H_bonds,accepted_H_bonds,donated_H_bonds,density_z', comments='')
    
    # Combine data from all trajectories
    combined_z_vals = []
    combined_total_bonds = []
    combined_accepted_bonds = []
    combined_donated_bonds = []
    
    for traj_idx in range(num_trajectories):
        traj_data = traj_H_bond_data[str(traj_idx)]
        combined_z_vals.extend(traj_data['z_vals'])
        combined_total_bonds.extend(traj_data['total_bonds'])
        combined_accepted_bonds.extend(traj_data['accepted_bonds'])
        combined_donated_bonds.extend(traj_data['donated_bonds'])
    
    combined_z_vals = np.array(combined_z_vals)
    combined_total_bonds = np.array(combined_total_bonds)
    combined_accepted_bonds = np.array(combined_accepted_bonds)
    combined_donated_bonds = np.array(combined_donated_bonds)
    
    # Bin combined data
    total_bonds_combined, _ = np.histogram(combined_z_vals, bins=z_bins, weights=combined_total_bonds)
    accepted_bonds_combined, _ = np.histogram(combined_z_vals, bins=z_bins, weights=combined_accepted_bonds)
    donated_bonds_combined, _ = np.histogram(combined_z_vals, bins=z_bins, weights=combined_donated_bonds)
    counts_combined, _ = np.histogram(combined_z_vals, bins=z_bins)
    
    # Calculate probability density for combined z values (normalize by bin width and total count)
    bin_width = z_bins[1] - z_bins[0]
    density_combined = counts_combined / (len(combined_z_vals) * bin_width)
    
    # Calculate mean bonds per bin for combined data
    mean_total_combined = np.divide(total_bonds_combined, counts_combined, out=np.zeros_like(total_bonds_combined, dtype=float), where=counts_combined!=0)
    mean_accepted_combined = np.divide(accepted_bonds_combined, counts_combined, out=np.zeros_like(accepted_bonds_combined, dtype=float), where=counts_combined!=0)
    mean_donated_combined = np.divide(donated_bonds_combined, counts_combined, out=np.zeros_like(donated_bonds_combined, dtype=float), where=counts_combined!=0)
    
    # Save combined data CSV
    if save_data:
        combined_csv_data = np.column_stack([z_bin_centers, mean_total_combined, mean_accepted_combined, mean_donated_combined, density_combined])
        np.savetxt(f"{system_data_dir}/{name}_combined_H_bond_binned_data.csv", 
                  combined_csv_data, delimiter=',', 
                  header='z_center,total_H_bonds,accepted_H_bonds,donated_H_bonds,density_z', comments='')
    
    end_time = time.time()
    print(f"Time taken to calculate H bonds vs z: {end_time - start_time:.2f} seconds")
    
    # Update and save H_bond_trajectories_data with any new connectivity matrices
    if new_H_bond_trajectories_data:
        # Merge new data with existing data
        for traj_key, frames_data in new_H_bond_trajectories_data.items():
            if traj_key not in H_bond_trajectories_data:
                H_bond_trajectories_data[traj_key] = {}
            H_bond_trajectories_data[traj_key].update(frames_data)
        
        # Save updated trajectories data
        with open(h_bond_trajectories_data_path, 'w') as f:
            json.dump(convert_numpy_types_for_json(H_bond_trajectories_data), f, indent=2)
        print(f"Updated H-bond trajectories data with {sum(len(frames) for frames in new_H_bond_trajectories_data.values())} new connectivity matrices")
    
    # Plot 1: Individual trajectory subplots
    fig, axes = plt.subplots(1, num_trajectories, figsize=(6*num_trajectories, 5))
    if num_trajectories == 1:
        axes = [axes]
    
    for traj_idx in range(num_trajectories):
        data = traj_binned_data[traj_idx]
        
        # Scale density to have maximum at 4 H-bonds/water height
        max_density = np.max(data['density_z'])
        if max_density > 0:
            density_scaled = data['density_z'] * (4.0 / max_density)
        else:
            density_scaled = data['density_z']
        
        # Plot density as filled area (no line, only shaded area)
        axes[traj_idx].fill_between(data['z_centers'], 0, density_scaled, 
                                   alpha=0.3, color='gray', label='Water density')
        
        # Plot H-bond data
        axes[traj_idx].plot(data['z_centers'], data['total_H_bonds'], label='Total H-bonds', linewidth=2, markersize=4)
        axes[traj_idx].plot(data['z_centers'], data['accepted_H_bonds'], label='Accepted H-bonds', linewidth=2, markersize=4)
        axes[traj_idx].plot(data['z_centers'], data['donated_H_bonds'], label='Donated H-bonds', linewidth=2, markersize=4)
        
        # Create twin axis for density label
        ax2 = axes[traj_idx].twinx()
        ax2.set_ylabel('Water density (a.u.)', color='gray')
        ax2.set_yticks([])  # Remove tick numbers, keep only label
        ax2.tick_params(axis='y', colors='gray')
        
        axes[traj_idx].set_xlabel('z [Å]')
        axes[traj_idx].set_ylabel('Mean H Bonds (per Water)')
        axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: H-Bond Coordination vs z')
        axes[traj_idx].grid(True, alpha=0.3)
        axes[traj_idx].legend(loc='upper right')
        axes[traj_idx].set_xlim(z_sampling_min, z_sampling_max)
    
    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_H_bonds_vs_z_by_traj.png', dpi=300)
    plt.close()
    
    # Plot 2: Combined data from all trajectories
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Scale density to have maximum at 4 H-bonds/water height
    max_density_combined = np.max(density_combined)
    if max_density_combined > 0:
        density_combined_scaled = density_combined * (4.0 / max_density_combined)
    else:
        density_combined_scaled = density_combined
    
    # Plot density as filled area (no line, only shaded area)
    ax1.fill_between(z_bin_centers, 0, density_combined_scaled, 
                     alpha=0.3, color='gray', label='Water density')
    
    # Plot H-bond data
    ax1.plot(z_bin_centers, mean_total_combined,  label='Total H-bonds', linewidth=2, markersize=0)
    ax1.plot(z_bin_centers, mean_accepted_combined, label='Accepted H-bonds', linewidth=2, markersize=0)
    ax1.plot(z_bin_centers, mean_donated_combined, label='Donated H-bonds', linewidth=2, markersize=0)
    
    # Create twin axis for density label
    ax2 = ax1.twinx()
    ax2.set_ylabel('Water density (a.u.)', color='gray')
    ax2.set_yticks([])  # Remove tick numbers, keep only label
    ax2.tick_params(axis='y', colors='gray')
    
    ax1.set_xlabel('z [Å]')
    ax1.set_ylabel('Mean Number of H Bonds (per Water)')
    ax1.set_title(f'{name}: H-Bond Coordination vs z')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(z_sampling_min, z_sampling_max)
    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_H_bonds_vs_z_combined.png', dpi=300)
    plt.close()
    
    # Plot 3: Combined data from all trajectories - Interfacial region focus
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Scale density to have maximum at 4 H-bonds/water height (same as Plot 2)
    max_density_combined = np.max(density_combined)
    if max_density_combined > 0:
        density_combined_scaled = density_combined * (4.0 / max_density_combined)
    else:
        density_combined_scaled = density_combined
    
    # Plot density as filled area (no line, only shaded area)
    ax1.fill_between(z_bin_centers, 0, density_combined_scaled, 
                     alpha=0.3, color='gray', label='Water density')
    
    # Plot H-bond data
    ax1.plot(z_bin_centers, mean_total_combined,  label='Total H-bonds', linewidth=2, markersize=0)
    ax1.plot(z_bin_centers, mean_accepted_combined, label='Accepted H-bonds', linewidth=2, markersize=0)
    ax1.plot(z_bin_centers, mean_donated_combined, label='Donated H-bonds', linewidth=2, markersize=0)
    
    # Create twin axis for density label
    ax2 = ax1.twinx()
    ax2.set_ylabel('Water density (a.u.)', color='gray')
    ax2.set_yticks([])  # Remove tick numbers, keep only label
    ax2.tick_params(axis='y', colors='gray')
    
    ax1.set_xlabel('z [Å]')
    ax1.set_ylabel('Mean Number of H Bonds (per Water)')
    ax1.set_title(f'{name}: H-Bond Coordination vs z (Interfacial Region)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(z_sampling_min, z_interfacial_sampling_max)
    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_H_bonds_vs_z_combined_interfacial.png', dpi=300)
    plt.close()
    
    # Plot 4: Mean H-bonds vs z plots (following the same binning logic as plot_angular_distributions)
    print("Calculating mean H-bonds vs z with proper binning...")
    
    # Calculate mean H-bonds for each z bin for each trajectory
    fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
    if num_trajectories == 1:
        axes = [axes]
    
    all_traj_mean_H_bonds = []
    
    for traj_idx in range(num_trajectories):
        traj_data = traj_H_bond_data[str(traj_idx)]
        z_vals = np.array(traj_data['z_vals'])
        total_bonds = np.array(traj_data['total_bonds'])
        accepted_bonds = np.array(traj_data['accepted_bonds'])
        donated_bonds = np.array(traj_data['donated_bonds'])
        
        # Calculate weighted mean H-bonds for each z bin
        mean_total_bonds_z = []
        mean_accepted_bonds_z = []
        mean_donated_bonds_z = []
        
        for z_idx in range(len(z_bin_centers)):
            z_min = z_bins[z_idx]
            z_max = z_bins[z_idx + 1]
            
            # Find data points in this z bin
            mask = (z_vals >= z_min) & (z_vals < z_max)
            
            if np.sum(mask) > 0:
                mean_total = np.mean(total_bonds[mask])
                mean_accepted = np.mean(accepted_bonds[mask])
                mean_donated = np.mean(donated_bonds[mask])
            else:
                mean_total = np.nan
                mean_accepted = np.nan
                mean_donated = np.nan
                
            mean_total_bonds_z.append(mean_total)
            mean_accepted_bonds_z.append(mean_accepted)
            mean_donated_bonds_z.append(mean_donated)
        
        all_traj_mean_H_bonds.append({
            'total': mean_total_bonds_z,
            'accepted': mean_accepted_bonds_z,
            'donated': mean_donated_bonds_z
        })
        
        # Determine the y-axis range for the mean values (excluding NaN)
        valid_mean_values = [v for v in mean_total_bonds_z if not np.isnan(v)]
        if valid_mean_values:
            max_mean_value = max(valid_mean_values)
            min_mean_value = min(valid_mean_values)
            
            # Scale density to fit 10% above the max mean value
            if np.max(density_combined) > 0:
                density_scaled = (density_combined / np.max(density_combined)) * (max_mean_value * 1.1)
            else:
                density_scaled = density_combined
            
            # Plot background density as filled area (subtle grey)
            axes[traj_idx].fill_between(z_bin_centers, 0, density_scaled, 
                                       alpha=0.7, color='lightgrey', zorder=0)
        
        # Plot mean H-bond curves
        axes[traj_idx].plot(z_bin_centers, mean_total_bonds_z, 'o-', markersize=3, zorder=2, 
                           label='Total H-bonds', linewidth=2)
        axes[traj_idx].plot(z_bin_centers, mean_accepted_bonds_z, 's-', markersize=3, zorder=2, 
                           label='Accepted H-bonds', linewidth=2)
        axes[traj_idx].plot(z_bin_centers, mean_donated_bonds_z, '^-', markersize=3, zorder=2, 
                           label='Donated H-bonds', linewidth=2)
        
        axes[traj_idx].set_xlabel('z [Å]')
        axes[traj_idx].set_ylabel('Mean H-bonds per Water')
        axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: Mean H-bonds vs Z')
        axes[traj_idx].grid(True, alpha=0.3)
        axes[traj_idx].legend()
        axes[traj_idx].set_xlim(z_sampling_min, z_sampling_max)
        
        # Create a twin axis for density label (no ticks)
        ax_density = axes[traj_idx].twinx()
        ax_density.set_ylabel('ρ(z)', color='grey', alpha=0.9)
        ax_density.set_yticks([])  # Remove ticks
        ax_density.spines['right'].set_color('grey')
        ax_density.spines['right'].set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_mean_H_bonds_vs_z_by_traj.png', dpi=300)
    plt.close()
    
    # Plot 5: Combined mean H-bonds vs z from all trajectories
    combined_mean_total_bonds_z = []
    combined_mean_accepted_bonds_z = []
    combined_mean_donated_bonds_z = []
    
    for z_idx in range(len(z_bin_centers)):
        z_min = z_bins[z_idx]
        z_max = z_bins[z_idx + 1]
        
        # Find data points in this z bin from all trajectories
        mask = (combined_z_vals >= z_min) & (combined_z_vals < z_max)
        
        if np.sum(mask) > 0:
            mean_total = np.mean(combined_total_bonds[mask])
            mean_accepted = np.mean(combined_accepted_bonds[mask])
            mean_donated = np.mean(combined_donated_bonds[mask])
        else:
            mean_total = np.nan
            mean_accepted = np.nan
            mean_donated = np.nan
            
        combined_mean_total_bonds_z.append(mean_total)
        combined_mean_accepted_bonds_z.append(mean_accepted)
        combined_mean_donated_bonds_z.append(mean_donated)
    
    plt.figure(figsize=(8, 6))
    
    # Determine the y-axis range for the combined mean values (excluding NaN)
    valid_combined_mean_values = [v for v in combined_mean_total_bonds_z if not np.isnan(v)]
    if valid_combined_mean_values:
        max_combined_mean_value = max(valid_combined_mean_values)
        min_combined_mean_value = min(valid_combined_mean_values)
        
        # Scale density to fit 10% above the max mean value
        if np.max(density_combined) > 0:
            density_scaled_combined = (density_combined / np.max(density_combined)) * (max_combined_mean_value * 1.1)
        else:
            density_scaled_combined = density_combined
        
        # Plot background density as filled area (subtle grey)
        plt.fill_between(z_bin_centers, 0, density_scaled_combined, 
                       alpha=0.7, color='lightgrey', zorder=0)
    
    # Plot combined mean H-bond curves
    plt.plot(z_bin_centers, combined_mean_total_bonds_z, 'o-', markersize=4, linewidth=2, zorder=2,
             label='Total H-bonds')
    plt.plot(z_bin_centers, combined_mean_accepted_bonds_z, 's-', markersize=4, linewidth=2, zorder=2,
             label='Accepted H-bonds')
    plt.plot(z_bin_centers, combined_mean_donated_bonds_z, '^-', markersize=4, linewidth=2, zorder=2,
             label='Donated H-bonds')
    
    plt.xlabel('z [Å]')
    plt.ylabel('Mean H-bonds per Water')
    plt.title(f'{name}: Mean H-bond Coordination vs z')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(z_sampling_min, z_sampling_max)
    
    # Create a twin axis for density label (no ticks)
    ax_density_combined = plt.gca().twinx()
    ax_density_combined.set_ylabel(f'ρ(z)', color='grey', alpha=0.9)
    ax_density_combined.set_yticks([])  # Remove ticks
    ax_density_combined.spines['right'].set_color('grey')
    ax_density_combined.spines['right'].set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(system_plots_dir + '/' + name + '_mean_H_bonds_vs_z_combined.png', dpi=300)
    plt.close()
    
    # Save mean H-bond data as CSV files
    if save_data:
        # Individual trajectory mean H-bond data
        for traj_idx in range(num_trajectories):
            mean_data = all_traj_mean_H_bonds[traj_idx]
            csv_data = np.column_stack([z_bin_centers, mean_data['total'], mean_data['accepted'], mean_data['donated']])
            np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_mean_H_bonds_vs_z.csv", 
                      csv_data, delimiter=',', 
                      header='z_center,mean_total_H_bonds,mean_accepted_H_bonds,mean_donated_H_bonds', comments='')
        
        # Combined mean H-bond data
        combined_mean_csv_data = np.column_stack([z_bin_centers, combined_mean_total_bonds_z, 
                                                 combined_mean_accepted_bonds_z, combined_mean_donated_bonds_z])
        np.savetxt(f"{system_data_dir}/{name}_combined_mean_H_bonds_vs_z.csv", 
                  combined_mean_csv_data, delimiter=',', 
                  header='z_center,mean_total_H_bonds,mean_accepted_H_bonds,mean_donated_H_bonds', comments='')
        
        print(f"Saved mean H-bond vs z data to CSV files")
    
    print(f"Mean H-bond vs z analysis complete for {name}")


    