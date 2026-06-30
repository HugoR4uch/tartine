import json
import os
import re

import numpy as np

KB_EV_PER_K = 8.617333262145e-5


def _normalise_name(name):
    return name.rstrip("_")


def _passes_name_filters(name, whitelist=None, blacklist=None):
    if whitelist is not None and name not in set(whitelist):
        return False
    if blacklist is not None and name in set(blacklist):
        return False
    return True


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    return value


def _sum_counts(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 3:
        return counts.sum(axis=0)
    if counts.ndim == 2:
        return counts
    raise ValueError(
        "counts must have shape (n_z_bins, n_cos_bins) "
        "or (n_trajs, n_z_bins, n_cos_bins)"
    )


def _fit_component_order(fit_entry):
    gaussians = fit_entry.get("gaussians", {})
    component_order = fit_entry.get("component_order")
    if component_order is None:
        component_order = list(gaussians)
        if fit_entry.get("step") is not None:
            component_order.append("step")
    return component_order


def _regular_grid_from_spacing(bounds, spacing):
    bounds = tuple(float(x) for x in bounds)
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("Grid spacing must be positive")
    if bounds[1] <= bounds[0]:
        raise ValueError("Grid upper bound must be larger than lower bound")

    n_steps = int(np.floor((bounds[1] - bounds[0]) / spacing))
    values = bounds[0] + spacing * np.arange(n_steps + 1, dtype=float)
    values = values[values < bounds[1]]
    return np.append(values, bounds[1])


def _softmax_from_free_logits(free_logits):
    logits = np.concatenate([np.asarray(free_logits, dtype=float), [0.0]])
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def _pack_initial_params(
    gaussian_guesses,
    step_guess=None,
    mu_z_bounds=None,
    mu_c_bounds=None,
):
    gaussian_names = list(gaussian_guesses)
    use_step = step_guess is not None
    n_components = len(gaussian_names) + int(use_step)

    weights_guess = [gaussian_guesses[name].get("weight", 1.0) for name in gaussian_names]
    if use_step:
        weights_guess.append(step_guess.get("weight", 1.0))

    weights_guess = np.asarray(weights_guess, dtype=float)
    weights_guess = np.maximum(weights_guess, 1e-12)
    weights_guess = weights_guess / weights_guess.sum()
    free_logits = np.log(weights_guess[:-1] / weights_guess[-1])

    p0 = list(free_logits)
    lower = [-10.0] * (n_components - 1)
    upper = [10.0] * (n_components - 1)

    if mu_z_bounds is None:
        mu_z_bounds = (-np.inf, np.inf)
    if mu_c_bounds is None:
        mu_c_bounds = (-1.1, 1.1)

    for name in gaussian_names:
        guess = gaussian_guesses[name]
        p0.extend([
            guess["mu_z"],
            guess["mu_c"],
            guess["sigma_z"],
            guess["sigma_c"],
            guess.get("rho", 0.0),
        ])
        lower.extend([mu_z_bounds[0], mu_c_bounds[0], 1e-4, 1e-4, -0.999])
        upper.extend([mu_z_bounds[1], mu_c_bounds[1], np.inf, np.inf, 0.999])

    if use_step:
        p0.extend([
            step_guess["z0"],
            step_guess["lambda_z"],
            step_guess.get("switch_width", 0.05),
        ])
        lower.extend([mu_z_bounds[0], 1e-4, 1e-4])
        upper.extend([mu_z_bounds[1], np.inf, np.inf])

    p0 = np.asarray(p0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    p0[finite_lower] = np.maximum(p0[finite_lower], lower[finite_lower] + 1e-10)
    p0[finite_upper] = np.minimum(p0[finite_upper], upper[finite_upper] - 1e-10)

    return p0, lower, upper, gaussian_names, use_step


def _unpack_params(params, gaussian_names, use_step):
    params = np.asarray(params, dtype=float)
    n_components = len(gaussian_names) + int(use_step)
    weights = _softmax_from_free_logits(params[: n_components - 1])

    cursor = n_components - 1
    gaussians = {}
    for name in gaussian_names:
        mu_z, mu_c, sigma_z, sigma_c, rho = params[cursor : cursor + 5]
        gaussians[name] = {
            "mu_z": float(mu_z),
            "mu_c": float(mu_c),
            "sigma_z": float(sigma_z),
            "sigma_c": float(sigma_c),
            "rho": float(rho),
        }
        cursor += 5

    step = None
    if use_step:
        z0, lambda_z, switch_width = params[cursor : cursor + 3]
        step = {
            "z0": float(z0),
            "lambda_z": float(lambda_z),
            "switch_width": float(switch_width),
        }

    return weights, gaussians, step


def _compact_fit_parameters(fit_results):
    compact = {}
    drop_keys = {"z_bins", "cos_bins", "P_data", "P_fit", "F_data", "F_fit"}

    for material, fit_entry in fit_results.items():
        compact[material] = {
            key: value
            for key, value in fit_entry.items()
            if key not in drop_keys
        }

    return compact


def _marginal_z(probability, z_bins, cos_bins):
    dcos = np.diff(cos_bins)
    pz = np.sum(probability * dcos[None, :], axis=1)
    norm = np.sum(pz * np.diff(z_bins))
    if norm > 0:
        pz = pz / norm
    return pz


def _density_scale_from_profile(density_profile_entry, density_key="O_density"):
    z_density = np.asarray(density_profile_entry["z"], dtype=float)
    density = np.asarray(density_profile_entry[density_key], dtype=float)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    density_integral = trapezoid(density, z_density)
    if not np.isfinite(density_integral) or density_integral <= 0:
        raise ValueError(f"{density_key} has invalid integrated density")
    return density_integral


def _max_contact_layer_density(density_profile_entry, z_contact, density_key="O_density"):
    z_density = np.asarray(density_profile_entry["z"], dtype=float)
    density = np.asarray(density_profile_entry[density_key], dtype=float)
    contact_mask = (z_density >= z_contact[0]) & (z_density <= z_contact[1])
    if not np.any(contact_mask):
        raise ValueError("No density-profile points found inside contact-layer bounds")
    max_density = np.nanmax(density[contact_mask])
    if not np.isfinite(max_density) or max_density <= 0:
        raise ValueError(f"{density_key} has invalid contact-layer maximum density")
    return max_density


def _marginal_costheta_in_z_window(probability, z_bins, cos_bins, z_xlim):
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    z_mask = (z_centers >= z_xlim[0]) & (z_centers <= z_xlim[1])
    if not np.any(z_mask):
        raise ValueError("No z bins found inside contact-layer bounds")

    dz = np.diff(z_bins)
    pcos = np.sum(probability[z_mask, :] * dz[z_mask, None], axis=0)
    norm = np.sum(pcos * np.diff(cos_bins))
    if norm > 0:
        pcos = pcos / norm
    return pcos


def _contact_layer_bounds(density_profile_entry):
    metadata = density_profile_entry["metadata"]
    for start_key, end_key in [
        ("contact_layer_start", "contact_layer_end"),
        ("O_z_min", "O_z_max"),
        ("z_min", "z_max"),
    ]:
        if start_key in metadata and end_key in metadata:
            return float(metadata[start_key]), float(metadata[end_key])
    raise KeyError(
        "Density metadata does not contain contact_layer_start/contact_layer_end"
    )


def _markers_from_minima(minima, z_values, cos_values):
    markers = np.zeros((len(z_values), len(cos_values)), dtype=int)
    for label, minimum in enumerate(minima, start=1):
        z = minimum["z"] if isinstance(minimum, dict) else minimum[0]
        c = minimum["c"] if isinstance(minimum, dict) else minimum[1]
        i = int(np.argmin(np.abs(z_values - float(z))))
        j = int(np.argmin(np.abs(cos_values - float(c))))
        markers[i, j] = label
    return markers


def _watershed_neighbor_offsets(connectivity):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    raise ValueError("connectivity must be 4 or 8")


def _extract_watershed_boundaries(
    basin_labels,
    z_values,
    cos_values,
    connectivity=8,
    boundary_label=-1,
):
    basin_labels = np.asarray(basin_labels, dtype=int)
    z_values = np.asarray(z_values)
    cos_values = np.asarray(cos_values)
    offsets = _watershed_neighbor_offsets(connectivity)

    boundary_mask = np.zeros(basin_labels.shape, dtype=bool)
    boundary_points = []
    boundary_pairs = {}
    seen_edges = set()

    for i in range(basin_labels.shape[0]):
        for j in range(basin_labels.shape[1]):
            label = basin_labels[i, j]
            if label <= 0:
                continue
            for di, dj in offsets:
                ni = i + di
                nj = j + dj
                if (
                    ni < 0
                    or ni >= basin_labels.shape[0]
                    or nj < 0
                    or nj >= basin_labels.shape[1]
                ):
                    continue
                other_label = basin_labels[ni, nj]
                if other_label <= 0 or other_label == label:
                    continue
                edge_key = tuple(sorted(((i, j), (ni, nj))))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                boundary_mask[i, j] = True
                boundary_mask[ni, nj] = True
                point = [
                    0.5 * (float(z_values[i]) + float(z_values[ni])),
                    0.5 * (float(cos_values[j]) + float(cos_values[nj])),
                ]
                boundary_points.append(point)
                pair = tuple(sorted((int(label), int(other_label))))
                boundary_pairs.setdefault(pair, []).append(point)

    labels = basin_labels.copy()
    labels[boundary_mask] = int(boundary_label)
    boundary_points = np.asarray(boundary_points, dtype=float)
    boundary_pairs = {
        f"{pair[0]}-{pair[1]}": np.asarray(points, dtype=float)
        for pair, points in boundary_pairs.items()
    }

    return labels, boundary_mask, boundary_points, boundary_pairs


def _select_boundary_saddle_candidates(
    points,
    energies,
    radius,
    max_candidates,
    min_separation,
):
    points = np.asarray(points, dtype=float)
    energies = np.asarray(energies, dtype=float)
    finite = np.isfinite(energies) & np.all(np.isfinite(points), axis=1)
    points = points[finite]
    energies = energies[finite]
    if points.size == 0:
        return np.empty((0, 2), dtype=float)

    local_candidates = []
    for idx, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        neighbours = distances <= radius
        if not np.any(neighbours):
            continue
        if energies[idx] <= np.nanmin(energies[neighbours]):
            local_candidates.append(idx)

    candidate_indices = list(local_candidates)
    candidate_indices.extend(np.argsort(energies).tolist())

    selected = []
    seen = set()
    for idx in candidate_indices:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        point = points[idx]
        if any(np.linalg.norm(point - prev) < min_separation for prev in selected):
            continue
        selected.append(point)
        if max_candidates is not None and len(selected) >= int(max_candidates):
            break

    return np.asarray(selected, dtype=float)


def _shift_minima_free_energies(minima, F_ref):
    shifted = []
    for minimum in minima:
        if not isinstance(minimum, dict):
            shifted.append(minimum)
            continue
        item = dict(minimum)
        if "F" in item:
            item["F_unreferenced"] = item["F"]
            item["F"] = float(item["F"]) - F_ref
        shifted.append(item)
    return shifted


def _shift_result_free_energy_entries(entries, F_ref):
    shifted = []
    for entry in entries:
        if not isinstance(entry, dict):
            shifted.append(entry)
            continue
        item = dict(entry)
        if "F" in item:
            item["F_unreferenced"] = item["F"]
            item["F"] = float(item["F"]) - F_ref
        shifted.append(item)
    return shifted
