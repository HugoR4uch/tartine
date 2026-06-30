import glob
import os
import re

import numpy as np

from .species_utils import KB_EV_PER_K, _sum_counts


def load_costheta_z_histogram(input_dir, system_name):
    z_bins = np.loadtxt(
        os.path.join(input_dir, f"{system_name}_z_bin_edges.csv"),
        delimiter=",",
    )
    cos_bins = np.loadtxt(
        os.path.join(input_dir, f"{system_name}_costheta_bin_edges.csv"),
        delimiter=",",
    )

    multi_pattern = os.path.join(input_dir, f"{system_name}_costheta_z_counts_*.csv")
    multi_files = glob.glob(multi_pattern)

    if multi_files:
        def get_count_index(path):
            match = re.search(r"_counts_(\d+)\.csv$", path)
            return int(match.group(1)) if match else 10**9

        counts = np.array([
            np.loadtxt(path, delimiter=",")
            for path in sorted(multi_files, key=get_count_index)
        ])
    else:
        counts = np.loadtxt(
            os.path.join(input_dir, f"{system_name}_costheta_z_counts.csv"),
            delimiter=",",
        )

    return z_bins, cos_bins, counts


def load_dipole_costheta_z_histogram_data(
    angular_data_dir,
    whitelist=None,
    blacklist=None,
    mode="dipole",
    normalise_material_names=True,
):
    mode = mode.strip("_")
    pattern = os.path.join(angular_data_dir, f"*_{mode}_z_bin_edges.csv")
    z_edge_files = sorted(glob.glob(pattern))

    data = {}
    for z_file in z_edge_files:
        filename = os.path.basename(z_file)
        system_name = filename[: -len("_z_bin_edges.csv")]
        material = system_name[: -len(f"_{mode}")]
        if normalise_material_names:
            material = material.rstrip("_")

        if whitelist is not None and material not in set(whitelist):
            continue
        if blacklist is not None and material in set(blacklist):
            continue

        z_bins, cos_bins, counts = load_costheta_z_histogram(angular_data_dir, system_name)

        data[material] = {
            "material": material,
            "system_name": system_name,
            "mode": mode,
            "source_dir": angular_data_dir,
            "z_bins": np.asarray(z_bins, dtype=float),
            "cos_bins": np.asarray(cos_bins, dtype=float),
            "counts": np.asarray(counts, dtype=float),
        }

    return data


def histogram_to_probability_density(z_bins, cos_bins, counts):
    counts_2d = _sum_counts(counts)
    total_counts = counts_2d.sum()
    if total_counts <= 0:
        raise ValueError("Total counts are zero")

    dz = np.diff(np.asarray(z_bins, dtype=float))
    dcos = np.diff(np.asarray(cos_bins, dtype=float))
    bin_area = dz[:, None] * dcos[None, :]

    probability = counts_2d / total_counts
    return probability / bin_area


def probability_to_free_energy(
    probability,
    z_bins,
    temperature=330.0,
    ref_z_bounds=(6.0, 8.0),
):
    probability = np.asarray(probability, dtype=float)
    kbt = KB_EV_PER_K * temperature
    p_masked = np.ma.masked_where(probability <= 0, probability)
    free_energy = -kbt * np.ma.log(p_masked)

    z_centers = 0.5 * (np.asarray(z_bins[:-1]) + np.asarray(z_bins[1:]))
    z_ref_mask = (z_centers >= ref_z_bounds[0]) & (z_centers <= ref_z_bounds[1])
    if not np.any(z_ref_mask):
        raise ValueError("No z bins found inside ref_z_bounds")

    return free_energy - free_energy[z_ref_mask, :].mean()


def histogram_to_free_energy(
    z_bins,
    cos_bins,
    counts,
    temperature=330.0,
    ref_z_bounds=(6.0, 8.0),
):
    probability = histogram_to_probability_density(z_bins, cos_bins, counts)
    free_energy = probability_to_free_energy(
        probability,
        z_bins,
        temperature=temperature,
        ref_z_bounds=ref_z_bounds,
    )
    return probability, free_energy


def fit_parameters_to_probability_density(z_bins, cos_bins, fit_entry):
    from .species_fit import correlated_gaussian_basis, smooth_left_exponential_step_basis

    z_bins = np.asarray(z_bins, dtype=float)
    cos_bins = np.asarray(cos_bins, dtype=float)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    cos_centers = 0.5 * (cos_bins[:-1] + cos_bins[1:])
    Z, C = np.meshgrid(z_centers, cos_centers, indexing="ij")
    bin_area = np.diff(z_bins)[:, None] * np.diff(cos_bins)[None, :]

    probability_fit = np.zeros((len(z_centers), len(cos_centers)), dtype=float)
    gaussians = fit_entry.get("gaussians", {})
    component_order = fit_entry.get("component_order")
    if component_order is None:
        component_order = list(gaussians)
        if fit_entry.get("step") is not None:
            component_order.append("step")

    for component_name in component_order:
        if component_name == "step":
            step = fit_entry.get("step")
            if step is None:
                continue
            basis = smooth_left_exponential_step_basis(
                Z,
                step["z0"],
                step["lambda_z"],
                step["switch_width"],
                bin_area,
            )
            probability_fit += step.get("weight", 1.0) * basis
            continue

        g = gaussians[component_name]
        basis = correlated_gaussian_basis(
            Z,
            C,
            g["mu_z"],
            g["mu_c"],
            g["sigma_z"],
            g["sigma_c"],
            g.get("rho", 0.0),
            bin_area,
        )
        probability_fit += g.get("weight", 1.0) * basis

    norm = np.sum(probability_fit * bin_area)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Fitted probability model has invalid norm")

    return probability_fit / norm
