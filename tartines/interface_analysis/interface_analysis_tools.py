import ase
import ase.io
import numpy as np
import copy
import json
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



def find_top_layer_indices(substrate,num_layers=None,tolerance=0.5):
    z_vals = substrate.positions[:,2]
    
    if num_layers == None:
        top_layer_z_val_threshold = np.max(z_vals) - tolerance # anything <tolerance> Angstrom below top atom
    else:
        top_layer_z_val_threshold = np.percentile(z_vals, 100*(1-1/num_layers))
    
    top_layer_indices = np.where(z_vals >= top_layer_z_val_threshold)[0]
    return top_layer_indices


def _get_2d_kabsch_rotation(moving_xy, reference_xy):
    """
    Return the proper 2D rotation matrix that best aligns moving_xy to reference_xy.
    Input coordinates are expected to already be centered consistently.
    """
    covariance = moving_xy.T @ reference_xy
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    return rotation


def _wrap_positions_xy_to_cell(positions, cell):
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)

    fractional_positions = positions @ np.linalg.inv(cell)
    fractional_positions[:, :2] = fractional_positions[:, :2] % 1.0

    return fractional_positions @ cell


def xy_to_primitive_fractional(xy, primitive_cell, origin_xy=None):
    xy = np.asarray(xy, dtype=float)
    primitive_cell = np.asarray(primitive_cell, dtype=float)

    in_plane_cell = primitive_cell[:2, :2]

    if origin_xy is None:
        origin_xy = np.zeros(2)
    else:
        origin_xy = np.asarray(origin_xy, dtype=float)

    return (xy - origin_xy) @ np.linalg.inv(in_plane_cell)


def primitive_fractional_to_xy(frac, primitive_cell, origin_xy=None):
    frac = np.asarray(frac, dtype=float)
    primitive_cell = np.asarray(primitive_cell, dtype=float)

    in_plane_cell = primitive_cell[:2, :2]

    if origin_xy is None:
        origin_xy = np.zeros(2)
    else:
        origin_xy = np.asarray(origin_xy, dtype=float)

    return frac @ in_plane_cell + origin_xy


def _align_sample_positions_to_top_layer_xy(
    frame,
    sample_indices,
    substrate_top_layer_indices,
    reference_top_layer_xy,
):
    """
    Align sampled coordinates to the initial substrate top-layer xy frame.

    The transform removes top-layer COM drift in xy, then applies the 2D Kabsch
    rotation fitted from the current top layer to the reference top layer.
    """
    sample_indices = np.asarray(sample_indices, dtype=int)

    current_top_layer_xy = frame.positions[substrate_top_layer_indices, :2]
    reference_top_layer_xy = np.asarray(reference_top_layer_xy, dtype=float)

    current_com_xy = np.mean(current_top_layer_xy, axis=0)
    reference_com_xy = np.mean(reference_top_layer_xy, axis=0)

    moving_centered_xy = current_top_layer_xy - current_com_xy
    reference_centered_xy = reference_top_layer_xy - reference_com_xy
    rotation = _get_2d_kabsch_rotation(moving_centered_xy, reference_centered_xy)

    sample_xy = frame.positions[sample_indices, :2]

    aligned_positions = frame.positions[sample_indices].copy()
    aligned_positions[:, :2] = (
        (sample_xy - current_com_xy) @ rotation
        + reference_com_xy
    )

    return aligned_positions



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



def get_interfacial_H(frame,substrate):
    
    # Needs to be implemnented. 
    # Gives indices of H atoms that are closer to substrate atoms than a water O atom.
    pass



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



def chemisorbed_water_criterion(frame, substrate):

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
    bridging_water_indices = []
    H_up_water_indices = []
    H_down_water_indices = []

    water_H_vectors = analyser.get_water_H_vectors(aqua_O_indices)

    for O_index in aqua_O_indices:
        H_vectors = water_H_vectors[O_index]

        if len(H_vectors) != 2:
            continue
        

        dipole_moment = ( H_vectors[0] + H_vectors[1] ) / 2
        delta = H_vectors[0] - dipole_moment


        cos_pitch = np.dot(dipole_moment, [0, 0, 1]) / (np.linalg.norm(dipole_moment))
        
        
        x_vec = np.cross(np.array([0,0,1]) , dipole_moment) / np.linalg.norm(dipole_moment) 

        cos_roll = np.dot ( 
            x_vec / np.linalg.norm(x_vec) ,
            delta / np.linalg.norm(delta)
        ) 

        # Doing abs should mean that angle alwats < 90
        pitch_angle = np.arccos(cos_pitch)
        roll_angle = np.arccos(abs(cos_roll))

        if pitch_angle < np.pi/2 and roll_angle < np.pi/4:
            chemisorbed_water_O_indices.append(O_index)
        if pitch_angle > np.pi/2 and roll_angle < np.pi/4:
            bridging_water_indices.append(O_index)
        if pitch_angle < np.pi/2 and roll_angle > np.pi/4:
            H_up_water_indices.append(O_index)
        if pitch_angle > np.pi/2 and roll_angle > np.pi/4:
            H_down_water_indices.append(O_index)


    return chemisorbed_water_O_indices, H_up_water_indices, bridging_water_indices, H_down_water_indices


def _convert_numpy_types_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _convert_numpy_types_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_types_for_json(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def load_euler_species_partitions(json_path):
    """Load custom Euler species partition definitions from JSON."""

    with open(json_path, "r") as f:
        partitions = json.load(f)

    if not isinstance(partitions, dict):
        raise ValueError("Euler species partition file must contain a dictionary.")

    for partition_name, species_definitions in partitions.items():
        if not isinstance(species_definitions, dict):
            raise ValueError(
                f"Partition {partition_name!r} must contain a species dictionary."
            )

        for species_name, species_spec in species_definitions.items():
            if not isinstance(species_spec, dict):
                raise ValueError(
                    f"Species {species_name!r} in {partition_name!r} must be a dictionary."
                )

            for bounds_key in ["z_bounds", "pitch_bounds", "roll_bounds"]:
                if bounds_key not in species_spec:
                    raise ValueError(
                        f"Species {species_name!r} in {partition_name!r} is missing "
                        f"{bounds_key!r}."
                    )

                bounds = species_spec[bounds_key]
                if bounds is None:
                    continue

                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError(
                        f"{bounds_key!r} for species {species_name!r} in "
                        f"{partition_name!r} must be null or a length-2 list."
                    )

                if bounds[0] >= bounds[1]:
                    raise ValueError(
                        f"{bounds_key!r} for species {species_name!r} in "
                        f"{partition_name!r} must be increasing."
                    )

    return partitions


def set_euler_species_z_bounds(species_definitions, z_bounds):
    """Return a copy of species_definitions with all z_bounds overwritten."""

    if z_bounds is None:
        runtime_z_bounds = None
    else:
        if len(z_bounds) != 2 or z_bounds[0] >= z_bounds[1]:
            raise ValueError("z_bounds must be None or an increasing length-2 sequence.")
        runtime_z_bounds = [float(z_bounds[0]), float(z_bounds[1])]

    species_definitions = copy.deepcopy(species_definitions)

    for species_spec in species_definitions.values():
        species_spec["z_bounds"] = runtime_z_bounds

    return species_definitions


def get_first_trough_after(z_values, z_ref, fallback=None):
    """Return the first z value greater than z_ref."""

    later_troughs = sorted([float(z) for z in z_values if float(z) > z_ref])

    if later_troughs:
        return later_troughs[0]

    if fallback is not None:
        return fallback

    return z_ref


def _value_in_bounds(value, bounds):
    if bounds is None:
        return True

    lower, upper = bounds
    return lower <= value < upper


def _water_pitch_roll_z(frame, substrate, top_layer_indices, O_index, H_vectors):
    if len(H_vectors) != 2:
        return None

    frame_positions = frame.get_positions()
    z_interface = np.mean(frame_positions[top_layer_indices, 2])
    z = frame[O_index].position[2] - z_interface

    dipole_moment = (H_vectors[0] + H_vectors[1]) / 2
    delta = H_vectors[0] - dipole_moment

    dipole_norm = np.linalg.norm(dipole_moment)
    delta_norm = np.linalg.norm(delta)

    if dipole_norm == 0 or delta_norm == 0:
        return None

    cos_pitch = np.dot(dipole_moment, [0, 0, 1]) / dipole_norm
    cos_pitch = np.clip(cos_pitch, -1.0, 1.0)

    x_vec = np.cross(np.array([0, 0, 1]), dipole_moment)
    x_norm = np.linalg.norm(x_vec)

    if x_norm == 0:
        return None

    cos_roll = np.dot(
        x_vec / x_norm,
        delta / delta_norm,
    )
    cos_roll = np.clip(abs(cos_roll), -1.0, 1.0)

    pitch = np.degrees(np.arccos(cos_pitch))
    roll = np.degrees(np.arccos(cos_roll))

    if not np.isfinite([z, pitch, roll]).all():
        return None

    return z, pitch, roll


def get_custom_euler_species_indices_traj(
    trajectory,
    substrate,
    species_definitions,
    allow_overlapping_species=False,
):
    """
    Classify water O atoms using custom z/pitch/roll species definitions.

    Bounds are interpreted as half-open intervals: lower <= value < upper.
    Angles in species_definitions are degrees.
    """

    species_indices_traj = {
        species_name: []
        for species_name in species_definitions.keys()
    }

    substrate_indices = np.arange(len(substrate))
    top_layer_indices = find_top_layer_indices(substrate, num_layers=None)

    for frame in tqdm(trajectory, desc="Finding custom Euler species indices"):
        analyser = water_analyser.Analyser(frame, substrate_indices)
        aqua_O_indices = analyser.aqua_O_indices
        water_H_vectors = analyser.get_water_H_vectors(aqua_O_indices)

        frame_species_indices = {
            species_name: []
            for species_name in species_definitions.keys()
        }

        for O_index in aqua_O_indices:
            data = _water_pitch_roll_z(
                frame,
                substrate,
                top_layer_indices,
                O_index,
                water_H_vectors[O_index],
            )

            if data is None:
                continue

            z, pitch, roll = data
            matching_species = []

            for species_name, species_spec in species_definitions.items():
                if not _value_in_bounds(z, species_spec["z_bounds"]):
                    continue
                if not _value_in_bounds(pitch, species_spec["pitch_bounds"]):
                    continue
                if not _value_in_bounds(roll, species_spec["roll_bounds"]):
                    continue

                matching_species.append(species_name)

            if len(matching_species) > 1 and not allow_overlapping_species:
                raise ValueError(
                    f"Water O index {O_index} matched multiple Euler species: "
                    f"{matching_species}"
                )

            for species_name in matching_species:
                frame_species_indices[species_name].append(O_index)

        for species_name in species_definitions.keys():
            species_indices_traj[species_name].append(frame_species_indices[species_name])

    return species_indices_traj


def _empty_density_profile(z_min, z_max, bins):
    bin_edges = np.linspace(z_min, z_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    zeros = np.zeros(bins)
    return bin_centers, zeros


def _mean_and_error(density_profiles):
    density_profiles = np.asarray(density_profiles, dtype=float)

    if density_profiles.ndim == 1:
        density_profiles = density_profiles[None, :]

    mean_density = np.mean(density_profiles, axis=0)

    if density_profiles.shape[0] > 1:
        errors = np.std(density_profiles, axis=0) / np.sqrt(density_profiles.shape[0])
    else:
        errors = np.zeros_like(mean_density)

    return mean_density, errors


def get_species_resolved_euler_z_density_profiles(
    trajectories,
    substrate,
    species_definitions,
    z_min=0,
    z_max=10,
    bins=400,
    sampling_interval=1,
    include_species_sum=True,
    allow_overlapping_species=False,
):
    """
    Compute default and custom Euler-species resolved water O density profiles.
    """

    sampled_trajectories = [
        traj[::sampling_interval]
        for traj in trajectories
    ]

    all_z, all_density, all_error = get_z_density_profile(
        sampled_trajectories,
        substrate,
        z_min=z_min,
        z_max=z_max,
        bins=bins,
        plot_all_profiles=False,
        species="O",
    )

    bin_centers, zero_density = _empty_density_profile(z_min, z_max, bins)

    species_density_by_traj = {
        species_name: []
        for species_name in species_definitions.keys()
    }
    species_sum_by_traj = []

    for traj in sampled_trajectories:
        species_indices_traj = get_custom_euler_species_indices_traj(
            traj,
            substrate,
            species_definitions,
            allow_overlapping_species=allow_overlapping_species,
        )

        this_traj_species_profiles = []

        for species_name, sampling_indices_trajectory in species_indices_traj.items():
            n_selected = sum(len(indices) for indices in sampling_indices_trajectory)

            if n_selected == 0:
                density = zero_density.copy()
            else:
                _, density, _ = get_z_density_profile(
                    [traj],
                    substrate,
                    z_min=z_min,
                    z_max=z_max,
                    bins=bins,
                    plot_all_profiles=False,
                    sampling_indices_trajectory=sampling_indices_trajectory,
                    species="O",
                )
                density = np.asarray(density, dtype=float)

            species_density_by_traj[species_name].append(density)
            this_traj_species_profiles.append(density)

        if include_species_sum:
            if len(this_traj_species_profiles) == 0:
                species_sum_by_traj.append(zero_density.copy())
            else:
                species_sum_by_traj.append(np.sum(this_traj_species_profiles, axis=0))

    profiles = {
        "all": {
            "z": np.asarray(all_z, dtype=float),
            "density": np.asarray(all_density, dtype=float),
            "error": np.asarray(all_error, dtype=float),
        },
        "species": {},
    }

    for species_name, density_profiles in species_density_by_traj.items():
        mean_density, errors = _mean_and_error(density_profiles)
        profiles["species"][species_name] = {
            "z": bin_centers.copy(),
            "density": mean_density,
            "error": errors,
        }

    if include_species_sum:
        mean_density, errors = _mean_and_error(species_sum_by_traj)
        profiles["species_sum"] = {
            "z": bin_centers.copy(),
            "density": mean_density,
            "error": errors,
        }

    return profiles


def save_species_resolved_density_profiles(csv_path, profiles, metadata=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    columns = {
        "z": profiles["all"]["z"],
        "all_density": profiles["all"]["density"],
        "all_error": profiles["all"]["error"],
    }

    for species_name, species_profile in profiles["species"].items():
        columns[f"{species_name}_density"] = species_profile["density"]
        columns[f"{species_name}_error"] = species_profile["error"]

    if "species_sum" in profiles:
        columns["species_sum_density"] = profiles["species_sum"]["density"]
        columns["species_sum_error"] = profiles["species_sum"]["error"]

    pd.DataFrame(columns).to_csv(csv_path, index=False)

    if metadata is not None:
        with open(csv_path + ".metadata.json", "w") as f:
            json.dump(_convert_numpy_types_for_json(metadata), f, indent=2)


def load_species_resolved_density_profiles(csv_path):
    df = pd.read_csv(csv_path)
    z = df["z"].to_numpy()

    profiles = {
        "all": {
            "z": z,
            "density": df["all_density"].to_numpy(),
            "error": df["all_error"].to_numpy(),
        },
        "species": {},
    }

    for column in df.columns:
        if not column.endswith("_density"):
            continue
        if column in ["all_density", "species_sum_density"]:
            continue

        species_name = column[: -len("_density")]
        error_column = f"{species_name}_error"

        profiles["species"][species_name] = {
            "z": z,
            "density": df[column].to_numpy(),
            "error": (
                df[error_column].to_numpy()
                if error_column in df.columns
                else np.zeros_like(z)
            ),
        }

    if "species_sum_density" in df.columns:
        profiles["species_sum"] = {
            "z": z,
            "density": df["species_sum_density"].to_numpy(),
            "error": (
                df["species_sum_error"].to_numpy()
                if "species_sum_error" in df.columns
                else np.zeros_like(z)
            ),
        }

    return profiles


def get_z_density_profile(trajectories,
                          substrate,
                          z_min=None,
                          z_max=None,
                          plot_all_profiles=False,
                          num_layers = None,
                          bins=400,
                          sampling_indices_trajectory = None,
                          species='O',
                          z_substrate_criterion=None,):


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
    



def get_layer_occupancies_from_traj(
    trajectories,
    z_bounds,
    substrate_indices,
    sampling_interval=1,
):
    """
    Count O atoms in each interface-relative z layer for each frame.

    Accepts either:
      - one trajectory: traj
      - multiple trajectories: [traj1, traj2, ...]

    Returns
    -------
    occupancies : np.ndarray
        Single traj: shape (n_frames, n_layers)
        Multi traj:  shape (n_trajs, n_frames, n_layers)
                     if all trajectories have the same length.
    """

    z_bounds = np.asarray(z_bounds, dtype=float)

    if z_bounds.ndim == 1:
        z_bounds = z_bounds[None, :]

    # Detect single trajectory: first element is an ASE Atoms frame
    if len(trajectories) > 0 and hasattr(trajectories[0], "get_positions"):
        trajectories = [trajectories]
        single_input = True
    else:
        single_input = False

    n_layers = len(z_bounds)
    all_occupancies = []

    for traj_idx, traj in enumerate(trajectories):
        traj = traj[::sampling_interval]

        n_frames = len(traj)
        occupancies = np.zeros((n_frames, n_layers), dtype=int)

        if substrate_indices is not None:
            substrate_indices_arr = np.asarray(substrate_indices, dtype=int)

            substrate_ref = traj[0][substrate_indices_arr]

            top_layer_local_indices = find_top_layer_indices(
                substrate_ref,
            )

            top_layer_global_indices = substrate_indices_arr[top_layer_local_indices]
            substrate_indices_set = set(substrate_indices_arr)
        else:
            top_layer_global_indices = None
            substrate_indices_set = set()

        for frame_idx, frame in enumerate(traj):
            if top_layer_global_indices is None:
                z_interface = 0.0
            else:
                z_interface = np.mean(frame.positions[top_layer_global_indices, 2])

            for atom in frame:
                if atom.index in substrate_indices_set:
                    continue

                if atom.symbol != "O":
                    continue

                z = atom.position[2] - z_interface

                for layer_idx, (z_min, z_max) in enumerate(z_bounds):
                    if z_min <= z < z_max:
                        occupancies[frame_idx, layer_idx] += 1
                        break

        all_occupancies.append(occupancies)

    if single_input:
        return all_occupancies[0]

    frame_counts = [x.shape[0] for x in all_occupancies]

    if len(set(frame_counts)) == 1:
        return np.stack(all_occupancies, axis=0)

    return np.array(all_occupancies, dtype=object)


def plot_layer_occupancies(
    layer_occupancies,
    z_bounds=None,
    filename=None,
    title="Layer occupancies",
    alpha_traj=0.25,
):
    """
    Plot layer occupancies vs sampled frame index.

    Accepts:
      (n_frames, n_layers)
      (n_trajs, n_frames, n_layers)

    Saves plot if filename is given.
    Saves plot data to .npz if data_filename is given.
    """

    occ = np.asarray(layer_occupancies, dtype=float)

    if occ.ndim == 2:
        occ = occ[None, :, :]

    if occ.ndim != 3:
        raise ValueError(
            "layer_occupancies must have shape "
            "(n_frames, n_layers) or (n_trajs, n_frames, n_layers)"
        )

    n_trajs, n_frames, n_layers = occ.shape
    frame_indices = np.arange(n_frames)

    mean_occ = np.nanmean(occ, axis=0)
    std_occ = np.nanstd(occ, axis=0)

    if z_bounds is not None:
        z_bounds_arr = np.asarray(z_bounds, dtype=float)
        if z_bounds_arr.ndim == 1:
            z_bounds_arr = z_bounds_arr[None, :]
        layer_labels = [
            rf"{z0:.2f}–{z1:.2f} Å"
            for z0, z1 in z_bounds_arr
        ]
    else:
        z_bounds_arr = None
        layer_labels = [f"Layer {i + 1}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(8, 6))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for layer_idx in range(n_layers):
        color = default_colors[layer_idx % len(default_colors)]

        for traj_idx in range(n_trajs):
            ax.plot(
                frame_indices,
                occ[traj_idx, :, layer_idx],
                color=color,
                alpha=alpha_traj,
                linewidth=1.0,
            )

        ax.plot(
            frame_indices,
            mean_occ[:, layer_idx],
            color=color,
            linewidth=2.5,
            label=layer_labels[layer_idx],
        )

        ax.fill_between(
            frame_indices,
            mean_occ[:, layer_idx] - std_occ[:, layer_idx],
            mean_occ[:, layer_idx] + std_occ[:, layer_idx],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Sampled frame index")
    ax.set_ylabel("Number of O atoms")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="Layer")
    fig.tight_layout()

  

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

        plt.close(fig)






def save_layer_occupancies(filename, occupancies, z_bounds):
    """
    Save single- or multi-trajectory layer occupancies and z bounds.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    occupancies = np.asarray(occupancies)
    z_bounds = np.asarray(z_bounds, dtype=float)

    if occupancies.dtype == object:
        np.savez_compressed(
            filename,
            z_bounds=z_bounds,
            is_object_array=True,
            n_trajs=len(occupancies),
            **{
                f"occupancies_traj_{i}": occupancies[i]
                for i in range(len(occupancies))
            },
        )
    else:
        np.savez_compressed(
            filename,
            occupancies=occupancies.astype(int),
            z_bounds=z_bounds,
            is_object_array=False,
        )

def load_layer_occupancies(filename):
    """
    Load single- or multi-trajectory layer occupancies and z bounds.
    """

    data = np.load(filename, allow_pickle=True)

    z_bounds = data["z_bounds"]
    is_object_array = bool(data["is_object_array"])

    if is_object_array:
        n_trajs = int(data["n_trajs"])
        occupancies = np.array(
            [data[f"occupancies_traj_{i}"] for i in range(n_trajs)],
            dtype=object,
        )
    else:
        occupancies = data["occupancies"]

    return occupancies, z_bounds





    
def get_euler_angle_data(trajectories,
                            substrate,
                            sampling_indices_trajectory=None,
                            ):

    
    num_layers = None


    for traj in trajectories:

        input_trajectory = copy.deepcopy(traj) # only sample up to max_T

        ##########################################
        #Finding interface - water relative z vals
        ##########################################

        #Finding the interface z-value
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
        all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=True)

        if sampling_indices_trajectory is not None:
            # For some reason, if sampling trajectory, we don't do things relative to the COM 
            all_top_layer_atom_trajectories = find_atomic_trajectories(input_trajectory,substrate_top_layer_indices,relative_to_COM=False)


        interface_z_mean_traj = []
        for frame_positions in all_top_layer_atom_trajectories.transpose(1,0,2): 
            z_vals = frame_positions[:,2]
            interface_z_mean_traj.append(np.mean(z_vals)) 


        #Aggregating fingerprint data over entire trajectory


        if sampling_indices_trajectory is None:

            sampling_indices_trajectory = [ np.arange(len(traj[0])) for i in range(len(traj)) ]


        substrate_indices = np.arange(len(substrate))

        all_euler_data = []

        for frame_index, indices in tqdm(enumerate(sampling_indices_trajectory)):


            analyser = water_analyser.Analyser(traj[frame_index], substrate_indices)
            H_vectors = analyser.get_water_H_vectors()

            frame_O_indices = [i for i in indices if i in analyser.aqua_O_indices]

            for index in frame_O_indices:  
                
                if len(H_vectors[index]) != 2:
                    continue

                z = traj[frame_index][index].position[2] - interface_z_mean_traj[frame_index]
                
                dipole_moment = ( H_vectors[index][0] + H_vectors[index][1] ) / 2
                delta = H_vectors[index][0] - dipole_moment


                cos_pitch = np.dot(dipole_moment, [0, 0, 1]) / (np.linalg.norm(dipole_moment))
                
                x_vec = np.cross(np.array([0,0,1]) , dipole_moment) / np.linalg.norm(dipole_moment) 

                cos_roll = np.dot ( 
                    x_vec / np.linalg.norm(x_vec) ,
                    delta / np.linalg.norm(delta)
                ) 

                # This should ensure that angle is always < 90
                pitch_angle = np.arccos(cos_pitch)
                roll_angle = np.arccos(abs(cos_roll))


                all_euler_data.append([pitch_angle, roll_angle, z])

        if len(all_euler_data) == 0:
            return np.array([]), np.array([]), np.array([])

        x_vals = np.array(all_euler_data)[:, 0]
        y_vals = np.array(all_euler_data)[:, 1]
        z_vals = np.array(all_euler_data)[:, 2]
        
        return x_vals, y_vals, z_vals



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
                
                z = slab[frame_index][index].position[2] - interface_z_mean_traj[frame_index]
                theta_1 = np.arccos(np.dot(H_vectors[index][0], [0, 0, 1]) / (np.linalg.norm(H_vectors[index][0]) * np.linalg.norm([0, 0, 1])))
                theta_2 = np.arccos(np.dot(H_vectors[index][1], [0, 0, 1]) / (np.linalg.norm(H_vectors[index][1]) * np.linalg.norm([0, 0, 1])))

                all_HH_fingerprint_data.append([theta_1, theta_2, z])

        x_vals = np.array(all_HH_fingerprint_data)[:, 0]
        y_vals = np.array(all_HH_fingerprint_data)[:, 1]
        z_vals = np.array(all_HH_fingerprint_data)[:, 2]
        
        return x_vals, y_vals, z_vals



def get_interfacial_z_vs_dipole_angles(frame,substrate,num_layers=None):

    """
    For a given frame, finds the z-coordinate and angle between normal and water dipole of each water O 
    atom relative to the mean substrate top layer z-value.
    """

    substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers)
    substrate_top_layer_z_vals  =frame[substrate_top_layer_indices].positions[:,2] 
    substrate_top_layer_z_val = np.mean(substrate_top_layer_z_vals)
    frame_z_vals = frame.positions[:,2]
    frame_interfacial_z_vals = frame_z_vals - substrate_top_layer_z_val


    num_substrate_atoms = len(substrate) 
    substrate_indices = np.arange(len(substrate))
    frame_analyser = water_analyser.Analyser(frame,substrate_indices=substrate_indices)


    # We assume that the normal vector is (0,0,1)
    normal_vector = np.array([0, 0, 1])

    water_O_indices = [ i for i in range(len(frame)) if frame[i].symbol == 'O' and i >= num_substrate_atoms]

    voronoi_dict = frame_analyser.get_voronoi_dict(water_O_indices)
    
    indices = []
    interfacial_angles = []
    interfacial_z_vals = []

    for atom_index in water_O_indices:
        
        H_indices = voronoi_dict[atom_index]
        if len(H_indices) != 2:
            # No data added if not water (not two OH bonds)
            continue

        else:
            # If water, (two OH bonds), add data
            indices.append(atom_index)
            z = frame_interfacial_z_vals[atom_index]
            interfacial_z_vals.append( z ) 
            
            water_dipole = frame_analyser.get_water_dipole_moment(atom_index)

            if water_dipole is None:
                pass
            else:
                # We normalise just to make sure
                dot_product = np.dot(water_dipole, normal_vector) / (np.linalg.norm(water_dipole) * np.linalg.norm(normal_vector))

                angle = np.arccos(dot_product) * 180 / np.pi

                interfacial_angles.append(angle) 


    return indices, interfacial_z_vals, interfacial_angles




def get_interfacial_z_vs_OH_angles(frame,substrate,num_layers=None):

    """
    For a given frame, finds the z-coordinate and angle between normal and OH vectors of each water O 
    atom relative to the mean substrate top layer z-value.
    Returns angles for both OH vectors in each water molecule.
    """

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

    # Get OH vectors for all water molecules at once (like the dipole function does)
    water_OH_vectors_dict = frame_analyser.get_water_H_vectors(water_O_indices)
    
    for atom_index in water_O_indices:
        
        z = frame_interfacial_z_vals[atom_index]
 
        H_indices = voronoi_dict[atom_index]
        if len(H_indices) != 2:
            continue

        # Get OH vectors for this water molecule from the pre-computed dictionary
        OH_vectors = water_OH_vectors_dict[atom_index]
        
        # Calculate angle for each OH vector with the normal
        for OH_vector in OH_vectors:
            indices.append(atom_index)
            interfacial_z_vals.append(z)
            
            # We normalise just to make sure
            dot_product = np.dot(OH_vector, normal_vector) / (np.linalg.norm(OH_vector) * np.linalg.norm(normal_vector))

            angle = np.arccos(dot_product) * 180 / np.pi

            interfacial_angles.append(angle) 


    return indices, interfacial_z_vals, interfacial_angles






def get_substrate_primitive_cell_data(substrate,symprec=1e-3):

    copy_substrate = copy.deepcopy(substrate)
    copy_substrate.cell[2]= np.array([0, 0, 100.0])
    lattice = copy_substrate.get_cell()

    positions = copy_substrate.get_scaled_positions()
    numbers = copy_substrate.get_atomic_numbers()
    cell = (lattice, positions, numbers)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    # NOTE!! standardize_cell gives info in new transformed coordinate system.
    # Thus, the primitive cell is defined according to a new set of coorinates, not the original one.
    # The dataset.primitive_lattice gives the primtive lattice in the original coordinate system.

    primitive_lattice = dataset.primitive_lattice
    origin_shift = dataset.origin_shift @ lattice

    return primitive_lattice, origin_shift,




def get_xy_density_profile(trajectories,
                           substrate, 
                           z_min=None,
                           z_max=None,
                           return_all_traj_data=False,
                           num_layers=None,
                           N_bins=50,
                           sampling_indices_trajectory = None,
                           species='O',
                           symprec=1e-2,
                           account_for_substrate_xy_motion=False,
                           tolerance=0.1):

    primitive_cell, origin_shift = get_substrate_primitive_cell_data(substrate,symprec=symprec)
    print(primitive_cell)

    original_unit_cell = substrate.cell.copy()

    data_vs_traj = {}

    for index, slab in enumerate(trajectories):

        if not account_for_substrate_xy_motion:
            for frame in slab:
                frame.cell = primitive_cell
                frame.wrap()


        input_trajectory = copy.deepcopy(slab) 


        #Finding the interface z-value (so we can select atoms with z_min < z < z_max)
        substrate_top_layer_indices = find_top_layer_indices(substrate,num_layers,tolerance=tolerance)
        reference_top_layer_xy = slab[0].positions[substrate_top_layer_indices, :2]
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

                if account_for_substrate_xy_motion:
                    sample_positions = _align_sample_positions_to_top_layer_xy(
                        slab[frame_index],
                        species_indices,
                        substrate_top_layer_indices,
                        reference_top_layer_xy,
                    )
                else:
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
            # print(len(frame_positions))
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



def get_xy_free_energy_profile(
    trajectories,
    substrate,
    z_bounds,
    reference_z_bounds=(8.0, 10.0),
    temperature=330.0,
    n_xy_bins=80,
    n_z_bins=60,
    z_min=0.0,
    z_max=30.0,
    sampling_interval=10,
    tolerance=0.1,
    num_layers=None,
    species="O",
    symprec=1e-2,
    return_by_traj=True,
    account_for_substrate_xy_motion=False,
):
    """
    Build a contact-layer F(u, v) map from primitive fractional u/v/z counts.

    The returned raw counts have shape (n_trajs, n_u_bins, n_v_bins, n_z_bins).
    Free energies are computed from density ratios. The plotted F(u, v) is
    -k_B T ln(rho_contact_uv / rho_ref), where rho_contact_uv is the
    contact-layer z-averaged density in each u/v bin and rho_ref is the mean
    density over all u/v/z bins inside reference_z_bounds.

    If account_for_substrate_xy_motion is True, sampled x/y coordinates are
    first translated and rotated into the initial substrate top-layer frame.
    Cartesian x/y coordinates are then converted to primitive fractional u/v
    coordinates and only those fractional coordinates are wrapped.
    """

    if z_bounds is None:
        raise ValueError("z_bounds must be provided as (z_min, z_max)")

    k_b_ev_per_k = 8.617333262145e-5
    kbt = k_b_ev_per_k * temperature

    primitive_cell, origin_shift = get_substrate_primitive_cell_data(
        substrate,
        symprec=symprec,
    )

    u_bins = np.linspace(0.0, 1.0, n_xy_bins + 1)
    v_bins = np.linspace(0.0, 1.0, n_xy_bins + 1)
    z_bins = np.linspace(float(z_min), float(z_max), n_z_bins + 1)

    traj_counts = np.zeros(
        (len(trajectories), n_xy_bins, n_xy_bins, n_z_bins),
        dtype=np.int64,
    )

    substrate_top_layer_indices = find_top_layer_indices(
        substrate,
        num_layers=num_layers,
        tolerance=tolerance,
    )
    substrate_indices = np.arange(len(substrate))

    for traj_idx, trajectory in enumerate(trajectories):
        sampled_frames = trajectory[::sampling_interval]
        reference_top_layer_xy = trajectory[0].positions[substrate_top_layer_indices, :2]

        for frame in tqdm(sampled_frames, desc=f"Processing trajectory {traj_idx + 1} F(u,v)"):
            z_interface = np.mean(frame.positions[substrate_top_layer_indices, 2])

            if species is None:
                sample_indices = np.arange(len(frame))
            elif species == "O":
                analyser = water_analyser.Analyser(frame, substrate_indices=substrate_indices)
                sample_indices = np.asarray(sorted(analyser.aqua_O_indices), dtype=int)
            else:
                all_species_indices = np.where(frame.symbols == species)[0]
                substrate_species_indices = np.where(substrate.symbols == species)[0]
                sample_indices = np.setdiff1d(all_species_indices, substrate_species_indices)

            if sample_indices.size == 0:
                continue

            if account_for_substrate_xy_motion:
                positions = _align_sample_positions_to_top_layer_xy(
                    frame,
                    sample_indices,
                    substrate_top_layer_indices,
                    reference_top_layer_xy,
                )
            else:
                positions = frame.positions[sample_indices].copy()

            original_positions = frame.positions[sample_indices]
            z_relative = original_positions[:, 2] - z_interface
            uv = xy_to_primitive_fractional(
                positions[:, :2],
                primitive_cell,
                origin_xy=origin_shift[:2],
            )
            # Only sampled water/species coordinates are folded into the primitive cell.
            # Substrate top-layer coordinates used for COM/Kabsch alignment stay continuous.
            uv_wrapped = uv % 1.0

            data = np.column_stack([
                uv_wrapped[:, 0],
                uv_wrapped[:, 1],
                z_relative,
            ])

            hist, _ = np.histogramdd(
                data,
                bins=[u_bins, v_bins, z_bins],
            )
            traj_counts[traj_idx] += hist.astype(np.int64)

    counts_for_free_energy = traj_counts.sum(axis=0)
    free_energy_xy, free_energy_xyz, reference_density = xy_histogram_to_relative_free_energy(
        counts_for_free_energy,
        u_bins,
        v_bins,
        z_bins,
        z_bounds=z_bounds,
        temperature=temperature,
        reference_z_bounds=reference_z_bounds,
        primitive_cell=primitive_cell,
    )

    result = {
        "u_bins": u_bins,
        "v_bins": v_bins,
        "x_bins": u_bins,
        "y_bins": v_bins,
        "z_bins": z_bins,
        "counts": traj_counts if return_by_traj else counts_for_free_energy,
        "free_energy_xy": free_energy_xy,
        "free_energy_uv": free_energy_xy,
        "free_energy_xyz": free_energy_xyz,
        "reference_density": reference_density,
        "z_bounds": np.asarray(z_bounds, dtype=float),
        "reference_z_bounds": np.asarray(reference_z_bounds, dtype=float),
        "temperature": float(temperature),
        "primitive_cell": primitive_cell,
        "origin_shift": origin_shift,
        "coordinate_system": "primitive_fractional_uv",
        "sampling_interval": int(sampling_interval),
        "top_layer_tolerance": float(tolerance),
        "account_for_substrate_xy_motion": bool(account_for_substrate_xy_motion),
    }

    return result


def xy_histogram_to_relative_free_energy(
    counts,
    x_bins,
    y_bins,
    z_bins,
    z_bounds,
    temperature=330.0,
    reference_z_bounds=(8.0, 10.0),
    primitive_cell=None,
):
    """
    Convert raw x/y/z or u/v/z counts to F relative to a bulk-like z reference.

    F(x, y) = -k_B T ln(rho_contact_xy / rho_ref)

    where rho_contact_xy is the z-averaged density in the requested contact
    layer for each x/y bin, and rho_ref is the mean density over all x/y/z
    bins inside reference_z_bounds.
    """

    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 4:
        counts = counts.sum(axis=0)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (n_x, n_y, n_z) or (n_traj, n_x, n_y, n_z)")

    total_counts = counts.sum()
    if total_counts <= 0:
        raise ValueError("Total counts are zero")

    x_bins = np.asarray(x_bins, dtype=float)
    y_bins = np.asarray(y_bins, dtype=float)
    z_bins = np.asarray(z_bins, dtype=float)

    dz = np.diff(z_bins)

    if primitive_cell is None:
        dx = np.diff(x_bins)
        dy = np.diff(y_bins)
        xy_area = dx[:, None] * dy[None, :]
    else:
        primitive_cell = np.asarray(primitive_cell, dtype=float)
        primitive_area = np.linalg.norm(np.cross(primitive_cell[0], primitive_cell[1]))
        du = np.diff(x_bins)
        dv = np.diff(y_bins)
        xy_area = primitive_area * du[:, None] * dv[None, :]

    bin_volume = xy_area[:, :, None] * dz[None, None, :]

    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    contact_z_mask = (z_centers >= z_bounds[0]) & (z_centers <= z_bounds[1])
    reference_z_mask = (
        (z_centers >= reference_z_bounds[0])
        & (z_centers <= reference_z_bounds[1])
    )

    if not np.any(contact_z_mask):
        raise ValueError("No z bins found inside z_bounds")
    if not np.any(reference_z_mask):
        raise ValueError("No z bins found inside reference_z_bounds")

    contact_counts_xy = counts[:, :, contact_z_mask].sum(axis=2)
    contact_thickness = dz[contact_z_mask].sum()
    contact_density_xy = contact_counts_xy / (total_counts * xy_area * contact_thickness)
    contact_density_xy = np.ma.masked_where(contact_density_xy <= 0, contact_density_xy)

    reference_counts = counts[:, :, reference_z_mask].sum()
    reference_volume = np.sum(bin_volume[:, :, reference_z_mask])
    if reference_counts <= 0 or reference_volume <= 0:
        raise ValueError("No counts found inside reference_z_bounds")
    reference_density = float(reference_counts / (total_counts * reference_volume))

    probability_density_xyz = counts / (total_counts * bin_volume)
    probability_density_xyz = np.ma.masked_where(probability_density_xyz <= 0, probability_density_xyz)

    k_b_ev_per_k = 8.617333262145e-5
    kbt = k_b_ev_per_k * temperature

    free_energy_xy = -kbt * np.ma.log(contact_density_xy / reference_density)
    free_energy_xyz = -kbt * np.ma.log(probability_density_xyz / reference_density)

    return free_energy_xy, free_energy_xyz, reference_density


def xy_histogram_to_free_energy(
    counts,
    x_bins,
    y_bins,
    z_bins,
    temperature=330.0,
    reference_z_bounds=(8.0, 10.0),
):
    """
    Convert raw x/y/z counts to a shifted F(x, y, z) masked array.

    Deprecated for contact-layer F(x, y): use
    xy_histogram_to_relative_free_energy, which computes F from density ratios
    rather than averaging free energies.
    """

    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 4:
        counts = counts.sum(axis=0)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (n_x, n_y, n_z) or (n_traj, n_x, n_y, n_z)")

    total_counts = counts.sum()
    if total_counts <= 0:
        raise ValueError("Total counts are zero")

    dx = np.diff(np.asarray(x_bins, dtype=float))
    dy = np.diff(np.asarray(y_bins, dtype=float))
    dz = np.diff(np.asarray(z_bins, dtype=float))
    bin_volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]

    probability_density = counts / (total_counts * bin_volume)
    probability_density = np.ma.masked_where(probability_density <= 0, probability_density)

    k_b_ev_per_k = 8.617333262145e-5
    free_energy = -k_b_ev_per_k * temperature * np.ma.log(probability_density)

    z_centers = 0.5 * (np.asarray(z_bins[:-1]) + np.asarray(z_bins[1:]))
    reference_z_mask = (
        (z_centers >= reference_z_bounds[0])
        & (z_centers <= reference_z_bounds[1])
    )
    if not np.any(reference_z_mask):
        raise ValueError("No z bins found inside reference_z_bounds")

    reference_values = free_energy[:, :, reference_z_mask]
    if np.ma.count(reference_values) == 0:
        raise ValueError("No finite free-energy values found inside reference_z_bounds")

    free_energy_reference = float(np.ma.mean(reference_values))
    return free_energy - free_energy_reference, free_energy_reference


def save_xy_free_energy_histogram(
    filename,
    x_bins,
    y_bins,
    z_bins,
    counts,
    z_bounds,
    reference_z_bounds=(8.0, 10.0),
    temperature=330.0,
    primitive_cell=None,
    origin_shift=None,
    sampling_interval=None,
    top_layer_tolerance=None,
    account_for_substrate_xy_motion=False,
):
    """
    Save raw u/v/z histogram counts and metadata for F(u, v) reconstruction.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    payload = {
        "x_bins": np.asarray(x_bins, dtype=float),
        "y_bins": np.asarray(y_bins, dtype=float),
        "u_bins": np.asarray(x_bins, dtype=float),
        "v_bins": np.asarray(y_bins, dtype=float),
        "z_bins": np.asarray(z_bins, dtype=float),
        "counts": np.asarray(counts, dtype=np.int64),
        "z_bounds": np.asarray(z_bounds, dtype=float),
        "reference_z_bounds": np.asarray(reference_z_bounds, dtype=float),
        "temperature": float(temperature),
        "account_for_substrate_xy_motion": bool(account_for_substrate_xy_motion),
        "coordinate_system": "primitive_fractional_uv",
    }

    if primitive_cell is not None:
        payload["primitive_cell"] = np.asarray(primitive_cell, dtype=float)
    if origin_shift is not None:
        payload["origin_shift"] = np.asarray(origin_shift, dtype=float)
    if sampling_interval is not None:
        payload["sampling_interval"] = int(sampling_interval)
    if top_layer_tolerance is not None:
        payload["top_layer_tolerance"] = float(top_layer_tolerance)

    np.savez_compressed(filename, **payload)


def load_xy_free_energy_histogram(filename):
    """
    Load raw x/y/z or u/v/z histogram counts and reconstruct shifted F.
    """

    data = np.load(filename, allow_pickle=False)

    has_fractional_uv_bins = "u_bins" in data.files and "v_bins" in data.files
    x_bins = data["u_bins"] if has_fractional_uv_bins else data["x_bins"]
    y_bins = data["v_bins"] if has_fractional_uv_bins else data["y_bins"]
    z_bins = data["z_bins"]
    counts = data["counts"]
    z_bounds = data["z_bounds"]
    reference_z_bounds = data["reference_z_bounds"]
    temperature = float(data["temperature"])

    free_energy_xy, free_energy_xyz, reference_density = xy_histogram_to_relative_free_energy(
        counts,
        x_bins,
        y_bins,
        z_bins,
        z_bounds=z_bounds,
        temperature=temperature,
        reference_z_bounds=reference_z_bounds,
        primitive_cell=(
            data["primitive_cell"]
            if has_fractional_uv_bins and "primitive_cell" in data.files
            else None
        ),
    )

    result = {
        "u_bins": x_bins,
        "v_bins": y_bins,
        "x_bins": x_bins,
        "y_bins": y_bins,
        "z_bins": z_bins,
        "counts": counts,
        "free_energy_xy": free_energy_xy,
        "free_energy_uv": free_energy_xy,
        "free_energy_xyz": free_energy_xyz,
        "reference_density": reference_density,
        "z_bounds": z_bounds,
        "reference_z_bounds": reference_z_bounds,
        "temperature": temperature,
        "coordinate_system": "primitive_fractional_uv" if has_fractional_uv_bins else "cartesian_xy",
    }

    if "primitive_cell" in data.files:
        result["primitive_cell"] = data["primitive_cell"]
    if "origin_shift" in data.files:
        result["origin_shift"] = data["origin_shift"]
    if "sampling_interval" in data.files:
        result["sampling_interval"] = int(data["sampling_interval"])
    if "top_layer_tolerance" in data.files:
        result["top_layer_tolerance"] = float(data["top_layer_tolerance"])
    if "account_for_substrate_xy_motion" in data.files:
        result["account_for_substrate_xy_motion"] = bool(data["account_for_substrate_xy_motion"])

    return result




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
