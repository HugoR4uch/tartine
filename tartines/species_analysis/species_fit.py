import numpy as np

from .species_utils import (
    _contact_layer_bounds,
    _fit_component_order,
    _pack_initial_params,
    _unpack_params,
    _regular_grid_from_spacing,
    _sum_counts,
)


def _normalise_basis_on_grid(basis, bin_area):
    basis = np.asarray(basis, dtype=float)
    norm = np.sum(basis * bin_area)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Basis function has invalid norm")
    return basis / norm


def correlated_gaussian_basis(
    Z,
    C,
    mu_z,
    mu_c,
    sigma_z,
    sigma_c,
    rho,
    bin_area,
):
    sigma_z = max(float(sigma_z), 1e-8)
    sigma_c = max(float(sigma_c), 1e-8)
    rho = float(np.clip(rho, -0.999, 0.999))

    z_scaled = (Z - mu_z) / sigma_z
    c_scaled = (C - mu_c) / sigma_c
    exponent = -0.5 / (1.0 - rho**2) * (
        z_scaled**2 - 2.0 * rho * z_scaled * c_scaled + c_scaled**2
    )
    max_exponent = np.nanmax(exponent)
    if not np.isfinite(max_exponent):
        raise ValueError("Gaussian exponent has no finite values")

    basis = np.exp(np.clip(exponent - max_exponent, -745, 80))
    return _normalise_basis_on_grid(basis, bin_area)


def smooth_left_exponential_step_basis(
    Z,
    z0,
    lambda_z,
    switch_width,
    bin_area,
):
    lambda_z = max(float(lambda_z), 1e-8)
    switch_width = max(float(switch_width), 1e-8)
    left_exp = np.exp(np.clip(-(Z - z0) / lambda_z, -745, 80))
    switch = 1.0 / (1.0 + np.exp(np.clip((Z - z0) / switch_width, -80, 80)))
    basis = switch * left_exp + (1.0 - switch)
    return _normalise_basis_on_grid(basis, bin_area)


def _raw_correlated_gaussian_derivatives(
    z,
    c,
    mu_z,
    mu_c,
    sigma_z,
    sigma_c,
    rho,
):
    sigma_z = max(float(sigma_z), 1e-8)
    sigma_c = max(float(sigma_c), 1e-8)
    rho = float(np.clip(rho, -0.999, 0.999))
    D = 1.0 - rho**2

    z_delta = np.asarray(z, dtype=float) - mu_z
    c_delta = np.asarray(c, dtype=float) - mu_c
    u = z_delta / sigma_z
    v = c_delta / sigma_c
    exponent = -0.5 / D * (u**2 - 2.0 * rho * u * v + v**2)
    basis = np.exp(np.clip(exponent, -745, 80))

    Kzz = 1.0 / (D * sigma_z**2)
    Kzc = -rho / (D * sigma_z * sigma_c)
    Kcc = 1.0 / (D * sigma_c**2)

    qz = Kzz * z_delta + Kzc * c_delta
    qc = Kzc * z_delta + Kcc * c_delta

    grad_z = -basis * qz
    grad_c = -basis * qc
    hzz = basis * (qz**2 - Kzz)
    hzc = basis * (qz * qc - Kzc)
    hcc = basis * (qc**2 - Kcc)

    return basis, grad_z, grad_c, hzz, hzc, hcc


def _raw_smooth_left_step_derivatives(z, z0, lambda_z, switch_width):
    lambda_z = max(float(lambda_z), 1e-8)
    switch_width = max(float(switch_width), 1e-8)
    z = np.asarray(z, dtype=float)

    left_exp = np.exp(np.clip(-(z - z0) / lambda_z, -745, 80))
    switch_arg = np.clip((z - z0) / switch_width, -80, 80)
    switch = 1.0 / (1.0 + np.exp(switch_arg))

    d_left_exp = -left_exp / lambda_z
    d2_left_exp = left_exp / lambda_z**2
    d_switch = -switch * (1.0 - switch) / switch_width
    d2_switch = switch * (1.0 - switch) * (1.0 - 2.0 * switch) / switch_width**2

    basis = switch * left_exp + (1.0 - switch)
    grad_z = d_switch * (left_exp - 1.0) + switch * d_left_exp
    hzz = d2_switch * (left_exp - 1.0) + 2.0 * d_switch * d_left_exp + switch * d2_left_exp

    zeros = np.zeros_like(basis, dtype=float)
    return basis, grad_z, zeros, hzz, zeros, zeros


def fit_component_normalisations(
    fit_entry,
    z_bounds,
    cos_bounds,
    dz,
    dc,
):
    z_bounds = tuple(float(x) for x in z_bounds)
    cos_bounds = tuple(float(x) for x in cos_bounds)
    dz = float(dz)
    dc = float(dc)
    if dz <= 0 or dc <= 0:
        raise ValueError("dz and dc must be positive")

    z_values = _regular_grid_from_spacing(z_bounds, dz)
    c_values = _regular_grid_from_spacing(cos_bounds, dc)
    if z_values.size < 2 or c_values.size < 2:
        raise ValueError("Stationary-point grid must contain at least 2 points per axis")

    Z, C = np.meshgrid(z_values, c_values, indexing="ij")
    normalisations = {}

    for component_name in _fit_component_order(fit_entry):
        if component_name == "step":
            step = fit_entry.get("step")
            if step is None:
                continue
            basis = _raw_smooth_left_step_derivatives(
                Z,
                step["z0"],
                step["lambda_z"],
                step["switch_width"],
            )[0]
        else:
            g = fit_entry["gaussians"][component_name]
            basis = _raw_correlated_gaussian_derivatives(
                Z,
                C,
                g["mu_z"],
                g["mu_c"],
                g["sigma_z"],
                g["sigma_c"],
                g.get("rho", 0.0),
            )[0]

        norm = float(
            np.trapezoid(
                np.trapezoid(basis, c_values, axis=1),
                z_values,
                axis=0,
            )
        )
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Basis function {component_name!r} has invalid norm")
        normalisations[component_name] = norm

    return normalisations


def evaluate_fit_free_energy_derivatives(
    z,
    c,
    fit_entry,
    temperature=None,
    component_normalisations=None,
):
    if temperature is None:
        temperature = fit_entry.get("temperature", 330.0)
    kbt = 8.617333262145e-5 * float(temperature)

    z_arr, c_arr = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(c, dtype=float))
    P = np.zeros_like(z_arr, dtype=float)
    Pz = np.zeros_like(z_arr, dtype=float)
    Pc = np.zeros_like(z_arr, dtype=float)
    Pzz = np.zeros_like(z_arr, dtype=float)
    Pzc = np.zeros_like(z_arr, dtype=float)
    Pcc = np.zeros_like(z_arr, dtype=float)

    for component_name in _fit_component_order(fit_entry):
        if component_name == "step":
            step = fit_entry.get("step")
            if step is None:
                continue
            weight = step.get("weight", 1.0)
            basis, grad_z, grad_c, hzz, hzc, hcc = _raw_smooth_left_step_derivatives(
                z_arr,
                step["z0"],
                step["lambda_z"],
                step["switch_width"],
            )
        else:
            g = fit_entry["gaussians"][component_name]
            weight = g.get("weight", 1.0)
            basis, grad_z, grad_c, hzz, hzc, hcc = _raw_correlated_gaussian_derivatives(
                z_arr,
                c_arr,
                g["mu_z"],
                g["mu_c"],
                g["sigma_z"],
                g["sigma_c"],
                g.get("rho", 0.0),
            )

        norm = 1.0
        if component_normalisations is not None:
            norm = component_normalisations.get(component_name, 1.0)
        scale = float(weight) / float(norm)
        P += scale * basis
        Pz += scale * grad_z
        Pc += scale * grad_c
        Pzz += scale * hzz
        Pzc += scale * hzc
        Pcc += scale * hcc

    tiny = np.finfo(float).tiny
    P_safe = np.maximum(P, tiny)
    F = -kbt * np.log(P_safe)
    Fz = -kbt * Pz / P_safe
    Fc = -kbt * Pc / P_safe
    Fzz = -kbt * (Pzz / P_safe - (Pz * Pz) / (P_safe * P_safe))
    Fzc = -kbt * (Pzc / P_safe - (Pz * Pc) / (P_safe * P_safe))
    Fcc = -kbt * (Pcc / P_safe - (Pc * Pc) / (P_safe * P_safe))

    grad = np.stack([Fz, Fc], axis=-1)
    hessian = np.empty(z_arr.shape + (2, 2), dtype=float)
    hessian[..., 0, 0] = Fzz
    hessian[..., 0, 1] = Fzc
    hessian[..., 1, 0] = Fzc
    hessian[..., 1, 1] = Fcc

    return {
        "F": F,
        "grad": grad,
        "hessian": hessian,
        "P": P,
        "grad_P": np.stack([Pz, Pc], axis=-1),
        "hessian_P": np.stack(
            [
                np.stack([Pzz, Pzc], axis=-1),
                np.stack([Pzc, Pcc], axis=-1),
            ],
            axis=-2,
        ),
    }


def fit_costheta_z_gaussian_mixture(
    z_bins,
    cos_bins,
    counts,
    gaussian_guesses,
    step_guess=None,
    fit_z_bounds=(0.0, 8.0),
    density_profile_entry=None,
    contact_layer_fit_z_max=8.0,
    fit_cos_bounds=(-1.001, 1.001),
    min_counts=1,
    residual_mode="poisson_deviance",
    temperature=330.0,
    ref_z_bounds=(6.0, 8.0),
    max_nfev=20000,
):
    from scipy.optimize import least_squares
    from .species_histogram import histogram_to_probability_density, probability_to_free_energy

    z_bins = np.asarray(z_bins)
    cos_bins = np.asarray(cos_bins)
    counts_data = _sum_counts(counts)
    probability_data = histogram_to_probability_density(z_bins, cos_bins, counts_data)

    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    cos_centers = 0.5 * (cos_bins[:-1] + cos_bins[1:])
    Z, C = np.meshgrid(z_centers, cos_centers, indexing="ij")
    bin_area = np.diff(z_bins)[:, None] * np.diff(cos_bins)[None, :]

    if density_profile_entry is not None:
        contact_layer_start, _ = _contact_layer_bounds(density_profile_entry)
        fit_z_bounds = (contact_layer_start, contact_layer_fit_z_max)

    fit_mask = (
        np.isfinite(probability_data)
        & (probability_data >= 0)
        & (counts_data >= min_counts)
        & (z_centers[:, None] >= fit_z_bounds[0])
        & (z_centers[:, None] <= fit_z_bounds[1])
        & (cos_centers[None, :] >= fit_cos_bounds[0])
        & (cos_centers[None, :] <= fit_cos_bounds[1])
    )

    if not np.any(fit_mask):
        raise ValueError("No bins selected for fitting")

    p0, lower, upper, gaussian_names, use_step = _pack_initial_params(
        gaussian_guesses,
        step_guess,
        mu_z_bounds=fit_z_bounds,
        mu_c_bounds=fit_cos_bounds,
    )

    total_counts = counts_data.sum()

    def make_probability_model(params):
        weights, gaussians, step = _unpack_params(params, gaussian_names, use_step)
        model = np.zeros_like(probability_data, dtype=float)
        component_index = 0

        for name in gaussian_names:
            g = gaussians[name]
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
            model += weights[component_index] * basis
            component_index += 1

        if use_step:
            basis = smooth_left_exponential_step_basis(
                Z,
                step["z0"],
                step["lambda_z"],
                step["switch_width"],
                bin_area,
            )
            model += weights[component_index] * basis

        model_norm = np.sum(model * bin_area)
        return model / model_norm

    def residuals(params):
        model = make_probability_model(params)
        expected_counts = np.maximum(model * bin_area * total_counts, 1e-300)
        observed_counts = counts_data

        if residual_mode == "poisson_deviance":
            ratio = np.ones_like(observed_counts)
            positive = observed_counts > 0
            ratio[positive] = observed_counts[positive] / expected_counts[positive]
            dev = 2.0 * (
                expected_counts
                - observed_counts
                + np.where(positive, observed_counts * np.log(ratio), 0.0)
            )
            residual = np.sign(observed_counts - expected_counts) * np.sqrt(
                np.maximum(dev, 0.0)
            )
            return residual[fit_mask]

        if residual_mode == "pearson_counts":
            return ((observed_counts - expected_counts) / np.sqrt(expected_counts))[fit_mask]

        if residual_mode == "probability_density":
            weights = np.sqrt(np.maximum(observed_counts, 1.0))
            return ((probability_data - model) * weights)[fit_mask]

        raise ValueError(f"Unknown residual_mode: {residual_mode}")

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
    )

    probability_fit = make_probability_model(result.x)
    free_energy_fit = probability_to_free_energy(
        probability_fit,
        z_bins,
        temperature=temperature,
        ref_z_bounds=ref_z_bounds,
    )
    free_energy_data = probability_to_free_energy(
        probability_data,
        z_bins,
        temperature=temperature,
        ref_z_bounds=ref_z_bounds,
    )

    weights, gaussians, step = _unpack_params(result.x, gaussian_names, use_step)
    component_names = list(gaussian_names) + (['step'] if use_step else [])
    for component_index, name in enumerate(component_names):
        if name == 'step':
            step['weight'] = float(weights[component_index])
        else:
            gaussians[name]['weight'] = float(weights[component_index])

    return {
        'success': bool(result.success),
        'message': result.message,
        'cost': float(result.cost),
        'nfev': int(result.nfev),
        'gaussians': gaussians,
        'step': step,
        'component_order': component_names,
        'residual_mode': residual_mode,
        'fit_z_bounds': tuple(fit_z_bounds),
        'fit_cos_bounds': tuple(fit_cos_bounds),
        'min_counts': int(min_counts),
        'temperature': float(temperature),
        'ref_z_bounds': tuple(ref_z_bounds),
        'z_bins': z_bins,
        'cos_bins': cos_bins,
        'P_data': probability_data,
        'P_fit': probability_fit,
        'F_data': free_energy_data,
        'F_fit': free_energy_fit,
    }


def fit_all_material_gaussian_species(
    histogram_data,
    fit_peak_initial_guesses,
    density_profile_data=None,
    step_initial_guesses=None,
    default_step_guess=None,
    **fit_kwargs,
):
    fit_results = {}
    for material, material_data in histogram_data.items():
        if material not in fit_peak_initial_guesses:
            raise KeyError(f"No Gaussian initial guesses supplied for {material}")

        step_guess = default_step_guess
        if step_initial_guesses is not None:
            step_guess = step_initial_guesses.get(material, default_step_guess)

        fit_results[material] = fit_costheta_z_gaussian_mixture(
            material_data['z_bins'],
            material_data['cos_bins'],
            material_data['counts'],
            fit_peak_initial_guesses[material],
            step_guess=step_guess,
            density_profile_entry=(
                density_profile_data.get(material)
                if density_profile_data is not None
                else None
            ),
            **fit_kwargs,
        )

    return fit_results
