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
from matplotlib.patches import Patch



def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _restore_hbond_keys(obj, convert_int_keys=True):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if convert_int_keys:
                try:
                    k_new = int(k)
                except (ValueError, TypeError):
                    k_new = k
            else:
                k_new = k

            out[k_new] = _restore_hbond_keys(
                v,
                convert_int_keys=convert_int_keys,
            )

        return out

    if isinstance(obj, list):
        return [
            _restore_hbond_keys(v, convert_int_keys=convert_int_keys)
            for v in obj
        ]

    return obj


def save_json_data(data, filename):
    with open(filename, "w") as f:
        json.dump(_json_safe(data), f, indent=2)


def load_json_data(filename, convert_int_keys=True):
    with open(filename, "r") as f:
        data = json.load(f)

    return _restore_hbond_keys(
        data,
        convert_int_keys=convert_int_keys,
    )



def get_hbond_donor_acceptor_data(
    trajectories,
    substrate_indices,
    sampling_interval=10,
    angle_cut=30,
    include_substrate=True,
    num_substrate_layers_for_reference=None,
):
    """
    Get H-bond donor/acceptor data for one or more trajectories.

    Returns
    -------
    hbond_stats : dict
        {
            traj_idx: {
                frame_idx: {
                    donor_O: {
                        "donor_position": [x, y, z_relative],
                        "acceptors": {
                            acceptor_O: {
                                "acceptor_position": [x, y, z_relative]
                            }
                        }
                    }
                }
            }
        }

    Notes
    -----
    z_relative = O_z - z_interface(frame)

    where z_interface(frame) is the instantaneous mean z-position of the
    top substrate layer.
    """

    substrate_indices = np.asarray(substrate_indices, dtype=int)

    # Allow both a single trajectory and a list of trajectories
    if len(trajectories) > 0 and hasattr(trajectories[0], "get_positions"):
        trajectories = [trajectories]

    hbond_stats = {}

    for traj_idx, trajectory in enumerate(trajectories):
        substrate_ref = trajectory[0][substrate_indices]

        top_layer_local_indices = interface_analysis_tools.find_top_layer_indices(
            substrate_ref,
            num_layers=num_substrate_layers_for_reference,
        )

        top_layer_global_indices = substrate_indices[top_layer_local_indices]

        traj_hbond_data = {}

        for sampled_frame_idx, frame in tqdm(
            enumerate(trajectory[::sampling_interval]),
            desc=f"Computing H-bond donor/acceptor data: traj {traj_idx + 1}",
        ):
            actual_frame_idx = sampled_frame_idx * sampling_interval

            z_interface = np.mean(frame.positions[top_layer_global_indices, 2])

            analyser = water_analyser.Analyser(
                frame,
                substrate_indices=substrate_indices,
                theta_c=angle_cut,
            )

            water_O = analyser.aqua_O_indices
            substrate_O = analyser.substrate_O_indices

            if include_substrate:
                O_donors_acceptors = water_O | substrate_O
            else:
                O_donors_acceptors = water_O

            directed_connectivity = analyser.get_H_bond_connectivity(
                O_donors_analyse=O_donors_acceptors,
                O_acceptors_analyse=O_donors_acceptors,
                directed=True,
            )

            frame_data = {}

            for donor_O in O_donors_acceptors:
                donor_O = int(donor_O)

                donor_pos = np.array(frame.positions[donor_O], dtype=float)
                donor_pos[2] -= z_interface

                donor_acceptors = {}

                for acceptor_O, is_hbond in directed_connectivity.get(donor_O, {}).items():
                    if not is_hbond:
                        continue

                    acceptor_O = int(acceptor_O)

                    acceptor_pos = np.array(frame.positions[acceptor_O], dtype=float)
                    acceptor_pos[2] -= z_interface

                    donor_acceptors[acceptor_O] = {
                        "acceptor_position": acceptor_pos.tolist(),
                    }

                frame_data[donor_O] = {
                    "donor_position": donor_pos.tolist(),
                    "acceptors": donor_acceptors,
                }

            traj_hbond_data[actual_frame_idx] = frame_data

        hbond_stats[traj_idx] = traj_hbond_data

    return hbond_stats

def save_hbond_donor_acceptor_data(h_bond_data, filename):
    save_json_data(h_bond_data, filename)


def load_hbond_donor_acceptor_data(filename, convert_int_keys=True):
    return load_json_data(filename, convert_int_keys=convert_int_keys)



#################
# H bonds vs z
#################


def get_hbond_z_data(hbond_donor_acceptor_data, water_O_indices=None):
    """
    Convert donor/acceptor H-bond data into H-bonds vs z data.

    Accepts either old form:
        {frame_idx: frame_data}

    or new multi-trajectory form:
        {traj_idx: {frame_idx: frame_data}}

    Returns same structural level as input:
        old  -> {frame_idx: {O_index: {...}}}
        new  -> {traj_idx: {frame_idx: {O_index: {...}}}}
    """

    if water_O_indices is not None:
        water_O_indices = set(int(i) for i in water_O_indices)

    def convert_single_traj(single_traj_hbond_data):
        hbond_z_data = {}

        for frame_idx, frame_data in single_traj_hbond_data.items():
            frame_idx = int(frame_idx)

            O_positions = {}
            donated_counts = {}
            accepted_counts = {}

            for donor_O, donor_data in frame_data.items():
                donor_O = int(donor_O)

                O_positions[donor_O] = donor_data["donor_position"]

                acceptors = donor_data.get("acceptors", {})
                donated_counts[donor_O] = len(acceptors)

                for acceptor_O, acceptor_data in acceptors.items():
                    acceptor_O = int(acceptor_O)

                    O_positions[acceptor_O] = acceptor_data["acceptor_position"]
                    accepted_counts[acceptor_O] = accepted_counts.get(acceptor_O, 0) + 1

            if water_O_indices is None:
                O_indices_to_store = set(O_positions.keys())
            else:
                O_indices_to_store = water_O_indices

            reconstructed_frame_data = {}

            for O_index in O_indices_to_store:
                if O_index not in O_positions:
                    continue

                z_val = O_positions[O_index][2]

                num_donated = donated_counts.get(O_index, 0)
                num_accepted = accepted_counts.get(O_index, 0)

                reconstructed_frame_data[int(O_index)] = {
                    "z": float(z_val),
                    "hbond_counts": (int(num_donated), int(num_accepted)),
                }

            hbond_z_data[frame_idx] = reconstructed_frame_data

        return hbond_z_data

    # Detect new form: {traj_idx: {frame_idx: frame_data}}
    first_key = next(iter(hbond_donor_acceptor_data))
    first_val = hbond_donor_acceptor_data[first_key]

    # New form has one more dict level before frame_data.
    # Old form first_val is already frame_data, whose values contain "donor_position".
    is_new_multi_traj_form = not any(
        isinstance(v, dict) and "donor_position" in v
        for v in first_val.values()
    )

    if is_new_multi_traj_form:
        return {
            int(traj_idx): convert_single_traj(traj_data)
            for traj_idx, traj_data in hbond_donor_acceptor_data.items()
        }

    return convert_single_traj(hbond_donor_acceptor_data)



def plot_hbond_vs_z(
    hbond_z_data,
    z_bin_width=0.5,
    z_min=None,
    z_max=None,
    filename=None,
    alpha_traj=0.25,
):
    """
    Plot mean H-bonds vs z.

    Accepts either:
      old form:
        {frame_idx: {O_index: {"z": ..., "hbond_counts": (...)}}}

      new multi-trajectory form:
        {traj_idx: {frame_idx: {O_index: {"z": ..., "hbond_counts": (...)}}}}

    If multi-trajectory data is supplied:
      - plots each trajectory with low opacity
      - plots the mean across trajectories as a solid line
    """

    def is_single_traj_hbond_z_data(data):
        first_val = next(iter(data.values()))
        return isinstance(first_val, dict) and any(
            isinstance(v, dict) and "z" in v and "hbond_counts" in v
            for v in first_val.values()
        )

    def compute_binned_data(single_traj_hbond_z_data, z_bins):
        z_vals = []
        donated_vals = []
        accepted_vals = []
        total_vals = []

        for frame_idx, frame_data in single_traj_hbond_z_data.items():
            if frame_data is None:
                continue

            for O_index, entry in frame_data.items():
                if entry is None:
                    continue
                if "z" not in entry or "hbond_counts" not in entry:
                    continue

                z = float(entry["z"])
                hbond_counts = entry["hbond_counts"]

                if len(hbond_counts) != 2:
                    continue

                num_donated = int(hbond_counts[0])
                num_accepted = int(hbond_counts[1])
                num_total = num_donated + num_accepted

                z_vals.append(z)
                donated_vals.append(num_donated)
                accepted_vals.append(num_accepted)
                total_vals.append(num_total)

        z_vals = np.asarray(z_vals, dtype=float)
        donated_vals = np.asarray(donated_vals, dtype=float)
        accepted_vals = np.asarray(accepted_vals, dtype=float)
        total_vals = np.asarray(total_vals, dtype=float)

        if len(z_vals) == 0:
            raise ValueError("No valid H-bond z data found.")

        counts_per_bin, _ = np.histogram(z_vals, bins=z_bins)

        sum_donated_per_bin, _ = np.histogram(z_vals, bins=z_bins, weights=donated_vals)
        sum_accepted_per_bin, _ = np.histogram(z_vals, bins=z_bins, weights=accepted_vals)
        sum_total_per_bin, _ = np.histogram(z_vals, bins=z_bins, weights=total_vals)

        mean_donated_per_bin = np.divide(
            sum_donated_per_bin,
            counts_per_bin,
            out=np.full(counts_per_bin.shape, np.nan, dtype=float),
            where=counts_per_bin > 0,
        )

        mean_accepted_per_bin = np.divide(
            sum_accepted_per_bin,
            counts_per_bin,
            out=np.full(counts_per_bin.shape, np.nan, dtype=float),
            where=counts_per_bin > 0,
        )

        mean_total_per_bin = np.divide(
            sum_total_per_bin,
            counts_per_bin,
            out=np.full(counts_per_bin.shape, np.nan, dtype=float),
            where=counts_per_bin > 0,
        )

        return {
            "z_vals": z_vals,
            "donated_vals": donated_vals,
            "accepted_vals": accepted_vals,
            "total_vals": total_vals,
            "counts_per_bin": counts_per_bin,
            "mean_donated_per_bin": mean_donated_per_bin,
            "mean_accepted_per_bin": mean_accepted_per_bin,
            "mean_total_per_bin": mean_total_per_bin,
        }

    # --------------------------------------------------
    # Normalize input into {traj_idx: single_traj_hbond_z_data}
    # --------------------------------------------------
    if is_single_traj_hbond_z_data(hbond_z_data):
        hbond_z_data_by_traj = {0: hbond_z_data}
    else:
        hbond_z_data_by_traj = hbond_z_data

    # --------------------------------------------------
    # Determine global z range
    # --------------------------------------------------
    all_z_vals = []

    for traj_idx, single_traj_data in hbond_z_data_by_traj.items():
        for frame_idx, frame_data in single_traj_data.items():
            for O_index, entry in frame_data.items():
                if entry is None:
                    continue
                if "z" not in entry:
                    continue
                all_z_vals.append(float(entry["z"]))

    all_z_vals = np.asarray(all_z_vals, dtype=float)

    if len(all_z_vals) == 0:
        raise ValueError("No valid H-bond z data found.")

    if z_min is None:
        z_min = float(np.floor(np.nanmin(all_z_vals)))
    if z_max is None:
        z_max = float(np.ceil(np.nanmax(all_z_vals)))

    z_bins = np.arange(z_min, z_max + z_bin_width, z_bin_width)
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    # --------------------------------------------------
    # Compute per-trajectory binned curves
    # --------------------------------------------------
    per_traj_results = {}

    for traj_idx, single_traj_data in hbond_z_data_by_traj.items():
        per_traj_results[traj_idx] = compute_binned_data(single_traj_data, z_bins)

    donated_curves = np.array(
        [d["mean_donated_per_bin"] for d in per_traj_results.values()],
        dtype=float,
    )
    accepted_curves = np.array(
        [d["mean_accepted_per_bin"] for d in per_traj_results.values()],
        dtype=float,
    )
    total_curves = np.array(
        [d["mean_total_per_bin"] for d in per_traj_results.values()],
        dtype=float,
    )

    mean_donated_per_bin = np.nanmean(donated_curves, axis=0)
    mean_accepted_per_bin = np.nanmean(accepted_curves, axis=0)
    mean_total_per_bin = np.nanmean(total_curves, axis=0)

    counts_per_bin_total = np.sum(
        [d["counts_per_bin"] for d in per_traj_results.values()],
        axis=0,
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------
    all_donated = np.concatenate([d["donated_vals"] for d in per_traj_results.values()])
    all_accepted = np.concatenate([d["accepted_vals"] for d in per_traj_results.values()])
    all_total = np.concatenate([d["total_vals"] for d in per_traj_results.values()])

    print(f"Collected {len(all_z_vals)} O samples across {len(per_traj_results)} trajectories")
    print("Mean donated:", np.nanmean(all_donated))
    print("Mean accepted:", np.nanmean(all_accepted))
    print("Mean total:", np.nanmean(all_total))
    print("Max total:", np.nanmax(all_total))
    print("Non-empty z bins:", np.sum(counts_per_bin_total > 0), "/", len(counts_per_bin_total))

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "donated": "C0",
        "accepted": "C1",
        "total": "C2",
    }

    for traj_idx in per_traj_results:
        ax.plot(
            z_bin_centers,
            per_traj_results[traj_idx]["mean_donated_per_bin"],
            color=colors["donated"],
            alpha=alpha_traj,
            linewidth=1.2,
        )
        ax.plot(
            z_bin_centers,
            per_traj_results[traj_idx]["mean_accepted_per_bin"],
            color=colors["accepted"],
            alpha=alpha_traj,
            linewidth=1.2,
        )
        ax.plot(
            z_bin_centers,
            per_traj_results[traj_idx]["mean_total_per_bin"],
            color=colors["total"],
            alpha=alpha_traj,
            linewidth=1.2,
        )

    ax.plot(
        z_bin_centers,
        mean_donated_per_bin,
        color=colors["donated"],
        label="Mean donated H-bonds",
        linewidth=3,
    )
    ax.plot(
        z_bin_centers,
        mean_accepted_per_bin,
        color=colors["accepted"],
        label="Mean accepted H-bonds",
        linewidth=3,
    )
    ax.plot(
        z_bin_centers,
        mean_total_per_bin,
        color=colors["total"],
        label="Mean total H-bonds",
        linewidth=3,
    )

    ax.set_xlabel("z [Å]")
    ax.set_ylabel("Mean H-bonds per O")
    ax.set_title("Mean H-bond coordination vs z")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "z_bins": z_bins,
        "z_bin_centers": z_bin_centers,
        "counts_per_bin": counts_per_bin_total,
        "mean_donated_per_bin": mean_donated_per_bin,
        "mean_accepted_per_bin": mean_accepted_per_bin,
        "mean_total_per_bin": mean_total_per_bin,
        "donated_curves_by_traj": donated_curves,
        "accepted_curves_by_traj": accepted_curves,
        "total_curves_by_traj": total_curves,
    }


def save_hbond_z_data(hbond_z_data, filename):
    save_json_data(hbond_z_data, filename)


def load_hbond_z_data(filename, convert_int_keys=True):
    return load_json_data(filename, convert_int_keys=convert_int_keys)


def save_hbond_vs_z_plot_data_json(filename, hbond_vs_z_data):
    save_json_data(hbond_vs_z_data, filename)


###################
# Stratification
###################

def _normalise_z_bounds(z_bounds):
    """
    Accepts either:
      [z_min, z_max]
    or:
      [[z_min1, z_max1], [z_min2, z_max2], ...]
    """
    z_bounds = np.asarray(z_bounds, dtype=float)

    if z_bounds.ndim == 1:
        if len(z_bounds) != 2:
            raise ValueError("Single z_bound must be [z_min, z_max].")
        z_bounds = z_bounds[None, :]

    if z_bounds.ndim != 2 or z_bounds.shape[1] != 2:
        raise ValueError("z_bounds must be [z_min, z_max] or [[z_min, z_max], ...].")

    # Sort by lower bound
    z_bounds = z_bounds[np.argsort(z_bounds[:, 0])]

    # Basic sanity check
    if np.any(z_bounds[:, 1] <= z_bounds[:, 0]):
        raise ValueError("Each bound must satisfy z_max > z_min.")

    return z_bounds




def hbond_donor_acceptor_data_to_layer_matrices(
    hbond_donor_acceptor_data,
    z_bounds,
    z_cut=3.5,
    return_metadata=True,
):
    """
    Accepts either old single-trajectory form:
        {frame_idx: frame_data}

    or new multi-trajectory form:
        {traj_idx: {frame_idx: frame_data}}

    Returns
    -------
    matrices : np.ndarray
        If single trajectory:
            shape = (n_frames, n_regions, n_regions)

        If multiple trajectories:
            shape = (n_trajs, n_frames, n_regions, n_regions)
            if all trajectories have same sampled frame count.

            If frame counts differ, returns object array unless you adapt padding.
    """

    z_bounds = _normalise_z_bounds(z_bounds)

    n_bounds = len(z_bounds)
    n_regions = n_bounds + 2

    global_z_min = float(np.min(z_bounds[:, 0]))
    global_z_max = float(np.max(z_bounds[:, 1]))

    region_labels = (
        ["substrate"]
        + [rf"{z0:.2f}–{z1:.2f} Å" for z0, z1 in z_bounds]
        + ["bulk"]
    )

    def get_region_index(z):
        z = float(z)

        if (global_z_min - z_cut) <= z < global_z_min:
            return 0

        for bound_idx, (z_low, z_high) in enumerate(z_bounds):
            if z_low <= z < z_high:
                return bound_idx + 1

        if global_z_max <= z < (global_z_max + z_cut):
            return n_bounds + 1

        return None

    def is_single_traj_form(data):
        first_val = next(iter(data.values()))
        return isinstance(first_val, dict) and any(
            isinstance(v, dict) and "donor_position" in v
            for v in first_val.values()
        )

    def convert_single_traj(single_traj_data):
        frame_indices = sorted(int(k) for k in single_traj_data.keys())

        matrices = np.zeros(
            (len(frame_indices), n_regions, n_regions),
            dtype=int,
        )

        per_frame_region_assignments = {}

        for frame_array_idx, frame_idx in enumerate(frame_indices):

            frame_key = frame_idx
            if frame_key not in single_traj_data:
                frame_key = str(frame_idx)

            frame_data = single_traj_data[frame_key]

            O_positions = {}

            for donor_O, donor_data in frame_data.items():
                donor_O_int = int(donor_O)
                O_positions[donor_O_int] = donor_data["donor_position"]

                for acceptor_O, acceptor_data in donor_data.get("acceptors", {}).items():
                    acceptor_O_int = int(acceptor_O)
                    O_positions[acceptor_O_int] = acceptor_data["acceptor_position"]

            O_to_region = {}
            region_to_O = {i: [] for i in range(n_regions)}

            for O_idx, pos in O_positions.items():
                region_idx = get_region_index(pos[2])

                if region_idx is None:
                    continue

                O_to_region[O_idx] = region_idx
                region_to_O[region_idx].append(O_idx)

            per_frame_region_assignments[frame_idx] = region_to_O

            M = matrices[frame_array_idx]

            for donor_O, donor_data in frame_data.items():
                donor_O_int = int(donor_O)

                if donor_O_int not in O_to_region:
                    continue

                donor_region = O_to_region[donor_O_int]

                for acceptor_O, acceptor_data in donor_data.get("acceptors", {}).items():
                    acceptor_O_int = int(acceptor_O)

                    if acceptor_O_int not in O_to_region:
                        continue

                    acceptor_region = O_to_region[acceptor_O_int]
                    M[donor_region, acceptor_region] += 1

        return matrices, frame_indices, per_frame_region_assignments

    if is_single_traj_form(hbond_donor_acceptor_data):
        matrices, frame_indices, per_frame_region_assignments = convert_single_traj(
            hbond_donor_acceptor_data
        )

        if return_metadata:
            metadata = {
                "frame_indices": frame_indices,
                "z_bounds": z_bounds,
                "z_cut": z_cut,
                "region_labels": region_labels,
                "per_frame_region_assignments": per_frame_region_assignments,
                "multi_traj": False,
            }
            return matrices, metadata

        return matrices

    # Multi-trajectory form
    matrices_by_traj = []
    frame_indices_by_traj = {}
    assignments_by_traj = {}

    for traj_idx in sorted(hbond_donor_acceptor_data.keys(), key=int):
        traj_key = traj_idx
        traj_data = hbond_donor_acceptor_data[traj_key]

        matrices_i, frame_indices_i, assignments_i = convert_single_traj(traj_data)

        matrices_by_traj.append(matrices_i)
        frame_indices_by_traj[int(traj_idx)] = frame_indices_i
        assignments_by_traj[int(traj_idx)] = assignments_i

    frame_counts = [m.shape[0] for m in matrices_by_traj]

    if len(set(frame_counts)) == 1:
        matrices = np.stack(matrices_by_traj, axis=0)
    else:
        matrices = np.array(matrices_by_traj, dtype=object)

    if return_metadata:
        metadata = {
            "frame_indices_by_traj": frame_indices_by_traj,
            "z_bounds": z_bounds,
            "z_cut": z_cut,
            "region_labels": region_labels,
            "per_frame_region_assignments_by_traj": assignments_by_traj,
            "multi_traj": True,
        }
        return matrices, metadata

    return matrices



def compute_layer_hbond_all_regions_data(
    layer_populations,
    matrices,
    region_labels=None,
):
    """
    Compute per-layer, per-region H-bond data normalised by layer population.

    Accepts either single-trajectory data:

        layer_populations.shape = (n_frames, n_layers)
        matrices.shape           = (n_frames, n_regions, n_regions)

    or multi-trajectory data:

        layer_populations.shape = (n_trajs, n_frames, n_layers)
        matrices.shape           = (n_trajs, n_frames, n_regions, n_regions)

    Returns mean/std over trajectories, plus per-trajectory values.
    """

    layer_populations = np.asarray(layer_populations, dtype=float)
    matrices = np.asarray(matrices, dtype=float)

    # Promote single-trajectory input to multi-trajectory form
    if layer_populations.ndim == 2:
        layer_populations = layer_populations[None, :, :]

    if matrices.ndim == 3:
        matrices = matrices[None, :, :, :]

    if layer_populations.ndim != 3:
        raise ValueError(
            "layer_populations must have shape "
            "(n_frames, n_layers) or (n_trajs, n_frames, n_layers)"
        )

    if matrices.ndim != 4:
        raise ValueError(
            "matrices must have shape "
            "(n_frames, n_regions, n_regions) or "
            "(n_trajs, n_frames, n_regions, n_regions)"
        )

    n_trajs, n_frames, n_layers = layer_populations.shape
    n_trajs_m, n_frames_m, n_regions, n_regions_2 = matrices.shape

    if n_trajs != n_trajs_m:
        raise ValueError(
            f"Trajectory mismatch: layer_populations has {n_trajs}, "
            f"matrices has {n_trajs_m}"
        )

    if n_frames != n_frames_m:
        raise ValueError(
            f"Frame mismatch: layer_populations has {n_frames}, "
            f"matrices has {n_frames_m}"
        )

    if n_regions != n_regions_2:
        raise ValueError("matrices must be square in the last two dimensions")

    if n_regions != n_layers + 2:
        raise ValueError(
            f"Expected n_regions = n_layers + 2 = {n_layers + 2}, "
            f"but got {n_regions}"
        )

    if region_labels is None:
        region_labels = ["substrate"] + [f"L{i+1}" for i in range(n_layers)] + ["bulk"]

    if len(region_labels) != n_regions:
        raise ValueError("region_labels must have length n_layers + 2")

    outgoing_by_traj = np.full((n_trajs, n_layers, n_regions), np.nan, dtype=float)
    incoming_by_traj = np.full((n_trajs, n_layers, n_regions), np.nan, dtype=float)
    total_by_traj = np.full((n_trajs, n_layers, n_regions), np.nan, dtype=float)

    pop_sum_by_traj = layer_populations.sum(axis=1)  # (n_trajs, n_layers)
    M_sum_by_traj = matrices.sum(axis=1)             # (n_trajs, n_regions, n_regions)

    for traj_idx in range(n_trajs):
        M_sum = M_sum_by_traj[traj_idx]
        pop_sum = pop_sum_by_traj[traj_idx]

        for layer_idx in range(n_layers):
            region_i = layer_idx + 1
            pop_i = pop_sum[layer_idx]

            if pop_i == 0:
                continue

            for j in range(n_regions):
                outgoing_by_traj[traj_idx, layer_idx, j] = M_sum[region_i, j] / pop_i
                incoming_by_traj[traj_idx, layer_idx, j] = M_sum[j, region_i] / pop_i
                total_by_traj[traj_idx, layer_idx, j] = (
                    outgoing_by_traj[traj_idx, layer_idx, j]
                    + incoming_by_traj[traj_idx, layer_idx, j]
                )

    outgoing_mean = np.nanmean(outgoing_by_traj, axis=0)
    incoming_mean = np.nanmean(incoming_by_traj, axis=0)
    total_mean = np.nanmean(total_by_traj, axis=0)

    outgoing_std = np.nanstd(outgoing_by_traj, axis=0)
    incoming_std = np.nanstd(incoming_by_traj, axis=0)
    total_std = np.nanstd(total_by_traj, axis=0)

    return {
        # Means, same names as before for backward compatibility
        "outgoing": outgoing_mean,
        "incoming": incoming_mean,
        "total": total_mean,

        # Errors over trajectories
        "outgoing_std": outgoing_std,
        "incoming_std": incoming_std,
        "total_std": total_std,

        # Raw per-trajectory values
        "outgoing_by_traj": outgoing_by_traj,
        "incoming_by_traj": incoming_by_traj,
        "total_by_traj": total_by_traj,

        # Sums
        "pop_sum_by_traj": pop_sum_by_traj,
        "M_sum_by_traj": M_sum_by_traj,
        "pop_sum": pop_sum_by_traj.sum(axis=0),
        "M_sum": M_sum_by_traj.sum(axis=0),

        # Metadata
        "region_labels": region_labels,
        "layer_labels": [f"Layer {i+1}" for i in range(n_layers)],
        "n_layers": n_layers,
        "n_regions": n_regions,
        "n_trajs": n_trajs,
    }


def save_layer_hbond_all_regions_data(filename, data):
    """
    Save output from compute_layer_hbond_all_regions_data() to JSON.
    """
    with open(filename, "w") as f:
        json.dump(_json_safe(data), f, indent=2)



def plot_layer_hbond_all_regions_data(
    data,
    title=None,
    figsize=None,
    error_key="total_std",
    capsize=4,
    save_path=None,   # <-- new
    dpi=300,          # optional but useful
):
    outgoing = np.asarray(data["outgoing"], dtype=float)
    incoming = np.asarray(data["incoming"], dtype=float)
    total_err = np.asarray(data.get(error_key, np.zeros_like(outgoing)), dtype=float)

    region_labels = data["region_labels"]
    layer_labels = data["layer_labels"]

    n_layers, n_regions = outgoing.shape

    if figsize is None:
        figsize = (max(10, 2.2 * n_layers), 6)

    fig, ax = plt.subplots(figsize=figsize)

    group_centers = np.arange(n_layers)
    total_group_width = 0.85
    subbar_width = total_group_width / n_regions

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [default_colors[j % len(default_colors)] for j in range(n_regions)]

    for layer_idx in range(n_layers):
        group_left = group_centers[layer_idx] - total_group_width / 2

        for j in range(n_regions):
            x = group_left + j * subbar_width + subbar_width / 2

            out_val = outgoing[layer_idx, j]
            in_val = incoming[layer_idx, j]

            if np.isnan(out_val) or np.isnan(in_val):
                continue

            total_val = out_val + in_val
            err_val = total_err[layer_idx, j]

            ax.bar(
                x,
                out_val,
                width=subbar_width * 0.95,
                color=colors[j],
                edgecolor="black",
                hatch="///",
                linewidth=0.8,
            )

            ax.bar(
                x,
                in_val,
                bottom=out_val,
                width=subbar_width * 0.95,
                color=colors[j],
                edgecolor="black",
                linewidth=0.8,
            )

            if capsize is not None and not np.isnan(err_val):
                ax.errorbar(
                    x,
                    total_val,
                    yerr=err_val,
                    fmt="none",
                    color="black",
                    capsize=capsize,
                    linewidth=1.2,
                    zorder=10,
                )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Interfacial layer")
    ax.set_ylabel("H-bonds per water in this layer")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    region_handles = [
        Patch(facecolor=colors[j], edgecolor="black", label=region_labels[j])
        for j in range(n_regions)
    ]

    style_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="///", label="donated by this layer"),
        Patch(facecolor="white", edgecolor="black", label="accepted by this layer"),
    ]

    legend1 = ax.legend(
        handles=region_handles,
        title="Region",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=style_handles,
        title="H-bond direction",
        bbox_to_anchor=(1.02, 0.5),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()

    # <-- save here
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, ax



##################
# Angular analysis
##################



def get_donate_costheta_z_counts(
    hbond_donor_acceptor_data,
    z_min=0.0,
    z_max=30.0,
    n_z_bins=300,
    n_cos_bins=100,
):
    """
    Build count(z_donor, cos(theta_donor_to_acceptor)).

    z is donor z-position.
    cos(theta) is the cosine of the donor -> acceptor vector with +z.
    """

    z_bins = np.linspace(z_min, z_max, n_z_bins + 1)
    cos_bins = np.linspace(-1.0, 1.0, n_cos_bins + 1)

    z_vals = []
    cos_vals = []

    z_hat = np.array([0.0, 0.0, 1.0])

    for frame_idx, frame_data in hbond_donor_acceptor_data.items():
        for donor_O, donor_data in frame_data.items():

            donor_pos = np.asarray(donor_data["donor_position"], dtype=float)
            z_donor = donor_pos[2]

            for acceptor_O, acceptor_data in donor_data.get("acceptors", {}).items():

                acceptor_pos = np.asarray(
                    acceptor_data["acceptor_position"],
                    dtype=float,
                )

                donor_to_acceptor = acceptor_pos - donor_pos
                norm = np.linalg.norm(donor_to_acceptor)

                if norm == 0:
                    continue

                cos_theta = np.dot(donor_to_acceptor, z_hat) / norm
                cos_theta = np.clip(cos_theta, -1.0, 1.0)

                z_vals.append(z_donor)
                cos_vals.append(cos_theta)

    counts, _, _ = np.histogram2d(
        z_vals,
        cos_vals,
        bins=[z_bins, cos_bins],
    )

    return z_bins, cos_bins, counts.astype(int)




def plot_P_costheta_in_z_region(
    z_bins,
    cos_bins,
    counts,
    z_xlim,
    title=r"$P(\cos\theta)$",
    xlabel=r"$\cos(\theta)$",
    ylabel=r"$P(\cos\theta)$",
    filename=None,
):
    """
    Integrate count(z, cos(theta)) over z_xlim and plot P(cos(theta)).

    counts shape:
        (n_z_bins, n_cos_bins)
    """

    counts = np.asarray(counts, dtype=float)

    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_z_bins, n_cos_bins).")

    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    cos_centers = 0.5 * (cos_bins[:-1] + cos_bins[1:])
    dcos = np.diff(cos_bins)

    z_mask = (z_centers >= z_xlim[0]) & (z_centers <= z_xlim[1])

    Pcos = counts[z_mask, :].sum(axis=0)

    # Normalise as a probability density in cos(theta)
    norm = np.sum(Pcos * dcos)
    if norm > 0:
        Pcos = Pcos / norm

    plt.figure(figsize=(7, 5))
    plt.plot(cos_centers, Pcos, "o-", markersize=3, linewidth=1.8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(
        title + rf" for ${z_xlim[0]:.2f} \leq z \leq {z_xlim[1]:.2f}$ Å"
    )
    plt.xlim(-1, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    return cos_centers, Pcos



def plot_hbond_costheta_z_histogram(
    z_bins,
    cos_bins,
    counts,
    filename=None,
    z_xlim=None,
    title="H-bond orientation vs z",
    ylabel=r"$\cos(\theta)$",
    density=True,
    cmap="Blues",
    figsize=(8, 6),
):
    """
    Plot z vs cos(theta) histogram for H-bond orientation data.

    counts can be either:
      (n_z_bins, n_cos_bins)
      or
      (n_trajs, n_z_bins, n_cos_bins)

    This works for either donate_counts or accept_counts.
    """

    counts = np.asarray(counts)

    if counts.ndim == 3:
        counts_plot = counts.sum(axis=0)
    elif counts.ndim == 2:
        counts_plot = counts
    else:
        raise ValueError(
            "counts must have shape (n_z_bins, n_cos_bins) "
            "or (n_trajs, n_z_bins, n_cos_bins)"
        )

    counts_plot = counts_plot.astype(float)

    if density:
        total = counts_plot.sum()
        if total > 0:
            counts_plot = counts_plot / total
        cbar_label = "Probability"
    else:
        cbar_label = "Counts"

    plt.figure(figsize=figsize)

    plt.pcolormesh(
        z_bins,
        cos_bins,
        counts_plot.T,
        shading="auto",
        cmap=cmap,
    )

    plt.colorbar(label=cbar_label)
    plt.xlabel(r"$z$ [Å]")
    plt.ylabel(ylabel)
    plt.title(title)

    if z_xlim is not None:
        plt.xlim(*z_xlim)
    else:
        plt.xlim(z_bins[0], z_bins[-1])

    plt.ylim(-1, 1)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
