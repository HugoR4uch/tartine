from ase.visualize.plot import plot_atoms
import ase.io
import ase
import matplotlib.pyplot as plt
import sys
from tartines.interface_analysis import interface_analysis_tools
from tartines.interface_analysis import water_analyser
import os
import time
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import json
import glob
import re


def get_binned_interfacial_angular_data(
    substrate,
    trajectories,
    mode="OH",  # "OH" or "dipole"
    z_min=0.0,
    z_max=30.0,
    n_z_bins=300,
    n_cos_bins=100,
    sampling_interval=1,
    return_by_traj=False,
):
    """
    Return 2D histogram(s) of z vs cos(theta).

    If return_by_traj=False:
        counts has shape (n_z_bins, n_cos_bins)

    If return_by_traj=True:
        counts has shape (n_trajectories, n_z_bins, n_cos_bins)
    """

    mode = mode.lower()

    if mode in ["oh", "oh_angle", "oh_angles"]:
        angle_getter = interface_analysis_tools.get_interfacial_z_vs_OH_angles
        desc_label = "OH"
    elif mode in ["dipole", "dip", "dipole_angle", "dipole_angles"]:
        angle_getter = interface_analysis_tools.get_interfacial_z_vs_dipole_angles
        desc_label = "dipole"
    else:
        raise ValueError("mode must be one of: 'OH', 'oh', 'dipole', or 'dip'")

    z_bins = np.linspace(z_min, z_max, n_z_bins + 1)
    cos_bins = np.linspace(-1.0, 1.0, n_cos_bins + 1)

    traj_counts = np.zeros(
        (len(trajectories), n_z_bins, n_cos_bins),
        dtype=np.int64,
    )

    for traj_idx, traj in enumerate(trajectories):
        for frame in tqdm(
            traj[::sampling_interval],
            desc=f"Processing trajectory {traj_idx + 1} ({desc_label})",
        ):
            _, z_vals, angles = angle_getter(frame, substrate)

            z_vals = np.asarray(z_vals, dtype=float)
            angles = np.asarray(angles, dtype=float)
            cos_vals = np.cos(np.radians(angles))

            hist, _, _ = np.histogram2d(
                z_vals,
                cos_vals,
                bins=[z_bins, cos_bins],
            )

            traj_counts[traj_idx] += hist.astype(np.int64)

    if return_by_traj:
        counts = traj_counts
    else:
        counts = traj_counts.sum(axis=0)

    return z_bins, cos_bins, counts




def save_costheta_z_histogram(
    output_dir,
    system_name,
    z_bins,
    cos_bins,
    counts,
):
    """
    Save either one counts map or multiple counts maps.

    counts shape:
      (n_z_bins, n_cos_bins)
      or
      (n_trajs, n_z_bins, n_cos_bins)
    """

    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, f"{system_name}_z_bin_edges.csv"),
        z_bins,
        delimiter=",",
    )

    np.savetxt(
        os.path.join(output_dir, f"{system_name}_costheta_bin_edges.csv"),
        cos_bins,
        delimiter=",",
    )

    counts = np.asarray(counts)

    if counts.ndim == 2:
        np.savetxt(
            os.path.join(output_dir, f"{system_name}_costheta_z_counts.csv"),
            counts,
            delimiter=",",
            fmt="%d",
        )

    elif counts.ndim == 3:
        for i, counts_i in enumerate(counts, start=1):
            np.savetxt(
                os.path.join(output_dir, f"{system_name}_costheta_z_counts_{i}.csv"),
                counts_i,
                delimiter=",",
                fmt="%d",
            )

    else:
        raise ValueError(
            "counts must have shape (n_z_bins, n_cos_bins) "
            "or (n_trajs, n_z_bins, n_cos_bins)"
        )


def load_costheta_z_histogram(
    input_dir,
    system_name,
):
    """
    Load z bins, cos(theta) bins, and counts.

    If multiple counts files exist:
        returns counts with shape (n_trajs, n_z_bins, n_cos_bins)

    If only a single counts file exists:
        returns counts with shape (n_z_bins, n_cos_bins)
    """

    z_bins = np.loadtxt(
        os.path.join(input_dir, f"{system_name}_z_bin_edges.csv"),
        delimiter=",",
    )

    cos_bins = np.loadtxt(
        os.path.join(input_dir, f"{system_name}_costheta_bin_edges.csv"),
        delimiter=",",
    )

    multi_pattern = os.path.join(
        input_dir,
        f"{system_name}_costheta_z_counts_*.csv",
    )

    multi_files = glob.glob(multi_pattern)

    if multi_files:
        def get_count_index(path):
            match = re.search(r"_counts_(\d+)\.csv$", path)
            return int(match.group(1)) if match else 10**9

        multi_files = sorted(multi_files, key=get_count_index)

        counts = np.array([
            np.loadtxt(path, delimiter=",")
            for path in multi_files
        ])

    else:
        counts = np.loadtxt(
            os.path.join(input_dir, f"{system_name}_costheta_z_counts.csv"),
            delimiter=",",
        )

    return z_bins, cos_bins, counts


def get_interfacial_euler_angle_data(
    substrate,
    trajectories,
    z_min,
    z_max,
    sampling_interval=1,
    trajectory_indices=None,
):
    """
    Return raw contact-layer water pitch/roll data.

    The returned array has columns:
      trajectory_index, pitch_rad, roll_rad, z
    """

    if trajectory_indices is None:
        trajectory_indices = np.arange(1, len(trajectories) + 1)

    if len(trajectory_indices) != len(trajectories):
        raise ValueError(
            "Length of trajectory_indices must match number of trajectories"
        )

    all_euler_data = []

    for traj_idx, traj in enumerate(trajectories):
        sampled_traj = traj[::sampling_interval]

        interface_indices_traj = interface_analysis_tools.find_water_species_indices_traj(
            sampled_traj,
            substrate,
            interface_analysis_tools.interfacial_water_criterion,
            z_min=z_min,
            z_max=z_max,
        )

        pitch, roll, z_vals = interface_analysis_tools.get_euler_angle_data(
            [sampled_traj],
            substrate,
            sampling_indices_trajectory=interface_indices_traj,
        )

        if len(pitch) == 0:
            continue

        traj_labels = np.full(len(pitch), trajectory_indices[traj_idx])
        traj_data = np.column_stack([traj_labels, pitch, roll, z_vals])
        all_euler_data.append(traj_data)

    if len(all_euler_data) == 0:
        return np.empty((0, 4))

    euler_data = np.vstack(all_euler_data)
    finite_rows = np.all(np.isfinite(euler_data), axis=1)

    return euler_data[finite_rows]


def save_euler_angle_data(
    output_dir,
    system_name,
    euler_data,
):
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, f"{system_name}_euler_raw_data.csv"),
        euler_data,
        delimiter=",",
        header="trajectory,pitch_rad,roll_rad,z",
        comments="",
    )


def load_euler_angle_data(
    input_dir,
    system_name,
):
    data = np.loadtxt(
        os.path.join(input_dir, f"{system_name}_euler_raw_data.csv"),
        delimiter=",",
        skiprows=1,
    )

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        return np.empty((0, 4))

    if data.ndim == 1:
        data = data[None, :]

    return data


def plot_euler_pitch_roll_heatmap(
    euler_data,
    filename=None,
    title="Contact-layer water orientation",
    bins=100,
    density=False,
    cmap="viridis",
    figsize=(6, 5),
):
    """
    Plot the old-style water pitch vs roll heat map.

    euler_data must have columns:
      trajectory_index, pitch_rad, roll_rad, z
    """

    euler_data = np.asarray(euler_data, dtype=float)

    if euler_data.size == 0:
        print(f"No Euler angle data available for plot: {title}")
        return

    if euler_data.ndim == 1:
        euler_data = euler_data[None, :]

    pitch_deg = np.degrees(euler_data[:, 1])
    roll_deg = np.degrees(euler_data[:, 2])

    plt.figure(figsize=figsize)
    plt.hist2d(
        pitch_deg,
        roll_deg,
        bins=bins,
        range=[[0, 180], [0, 90]],
        density=density,
        cmap=cmap,
    )

    cbar_label = "Probability density" if density else "Counts"
    plt.colorbar(label=cbar_label)
    plt.xlabel("Pitch (degrees)")
    plt.ylabel("Roll (degrees)")
    plt.title(title)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()




def plot_costheta_z_histogram(
    z_bins,
    cos_bins,
    counts,
    filename=None,
    z_xlim=None,
    title="Orientation vs z",
    ylabel=r"$\cos(\theta)$",
    density=True,
    cmap="Blues",
    figsize=(8, 6),
):
    """
    Plot z vs cos(theta) histogram.

    counts can be either:
      (n_z_bins, n_cos_bins)
      or
      (n_trajs, n_z_bins, n_cos_bins)
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



def plot_interfacial_angular_distribution(
    z_bins,
    cos_bins,
    theta_counts,
    z_xlim,
    plot_by_traj=False,
    filename=None,
    plot_mean=True,
    title=rf"Angular distribution",
    xlabel=r"$\cos(\theta)$",
    ylabel=r"$P(\cos\theta)$",

):
    """
    Plot P(cos(theta)) using only counts with z inside z_xlim.

    theta_counts can have shape:
      (n_z_bins, n_cos_bins)
      or
      (n_trajs, n_z_bins, n_cos_bins)
    """

    theta_counts = np.asarray(theta_counts)

    if theta_counts.ndim == 2:
        counts_by_traj = theta_counts[None, :, :]
    elif theta_counts.ndim == 3:
        counts_by_traj = theta_counts
    else:
        raise ValueError(
            "theta_counts must have shape (n_z_bins, n_cos_bins) "
            "or (n_trajs, n_z_bins, n_cos_bins)"
        )

    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    cos_centers = 0.5 * (cos_bins[:-1] + cos_bins[1:])
    dcos = np.diff(cos_bins)

    z_mask = (z_centers >= z_xlim[0]) & (z_centers <= z_xlim[1])


    def get_Pcos(counts_matrix):
        Pcos = counts_matrix[z_mask, :].sum(axis=0).astype(float)

        total = np.sum(Pcos * dcos)
        if total > 0:
            Pcos = Pcos / total

        return Pcos

    plt.figure(figsize=(7, 5))

    if plot_by_traj:
        for traj_idx, counts_i in enumerate(counts_by_traj, start=1):
            Pcos = counts_i[z_mask, :].sum(axis=0).astype(float)

            total = np.sum(Pcos * dcos)
            if total > 0:
                Pcos = Pcos / total

            plt.plot(
                cos_centers,
                Pcos,
                "o-",
                markersize=3,
                linewidth=1.5,
                label=f"Traj {traj_idx}",
            )

        if plot_mean and counts_by_traj.shape[0] > 1:
            summed_counts = counts_by_traj.sum(axis=0)
            Pcos_mean = get_Pcos(summed_counts)

            plt.plot(
                cos_centers,
                Pcos_mean,
                "-",
                color="black",
                linewidth=3.0,
                label="Summed",
                zorder=10,
            )


        plt.legend()

    else:
        summed_counts = counts_by_traj.sum(axis=0)
        Pcos = summed_counts[z_mask, :].sum(axis=0).astype(float)

        total = np.sum(Pcos * dcos)
        if total > 0:
            Pcos = Pcos / total

        plt.plot(cos_centers, Pcos, "o-", markersize=3, linewidth=1.8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(
        title+rf" for ${z_xlim[0]:.2f} \leq z \leq {z_xlim[1]:.2f}$ Å"
    )
    plt.xlim(-1, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
