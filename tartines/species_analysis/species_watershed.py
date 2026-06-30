import heapq

import numpy as np

from .species_fit import (
    evaluate_fit_free_energy_derivatives,
    fit_component_normalisations,
)
from .species_utils import (
    _markers_from_minima,
    _regular_grid_from_spacing,
    _select_boundary_saddle_candidates,
    _shift_minima_free_energies,
    _shift_result_free_energy_entries,
    _extract_watershed_boundaries,
    _to_jsonable,
)


def _bulk_reference_free_energy_from_fit(
    fit_entry,
    cos_values,
    dz,
    temperature=None,
    ref_z_bounds=None,
    component_normalisations=None,
):
    if ref_z_bounds is None:
        ref_z_bounds = fit_entry.get("ref_z_bounds", (6.0, 8.0))
    ref_z_bounds = tuple(float(x) for x in ref_z_bounds)
    ref_z_values = _regular_grid_from_spacing(ref_z_bounds, dz)
    Z_ref, C_ref = np.meshgrid(ref_z_values, cos_values, indexing="ij")
    ref_eval = evaluate_fit_free_energy_derivatives(
        Z_ref,
        C_ref,
        fit_entry,
        temperature=temperature,
        component_normalisations=component_normalisations,
    )
    F_ref_values = np.asarray(ref_eval["F"], dtype=float)
    finite = np.isfinite(F_ref_values)
    if not np.any(finite):
        raise ValueError("No finite fitted free-energy values found in ref_z_bounds")
    return float(np.nanmean(F_ref_values)), ref_z_bounds


def find_fit_stationary_points(
    fit_entry,
    z_bounds,
    cos_bounds=(-1.0, 1.0),
    dz=0.02,
    dc=0.02,
    temperature=None,
    grad_tol=1e-5,
    sign_tol=0.0,
    candidate_mode="combined",
    max_candidates=None,
    use_component_normalisations=True,
    max_refine_nfev=2000,
    dedup_tol=None,
    ref_z_bounds=None,
    reference_to_bulk=True,
):
    from scipy.optimize import minimize

    z_bounds = tuple(float(x) for x in z_bounds)
    cos_bounds = tuple(float(x) for x in cos_bounds)
    dz = float(dz)
    dc = float(dc)

    component_normalisations = None
    if use_component_normalisations:
        component_normalisations = fit_component_normalisations(
            fit_entry,
            z_bounds,
            cos_bounds,
            dz,
            dc,
        )

    F_ref = 0.0
    used_ref_z_bounds = None
    if reference_to_bulk:
        ref_cos_values = _regular_grid_from_spacing(cos_bounds, dc)
        F_ref, used_ref_z_bounds = _bulk_reference_free_energy_from_fit(
            fit_entry,
            ref_cos_values,
            dz,
            temperature=temperature,
            ref_z_bounds=ref_z_bounds,
            component_normalisations=component_normalisations,
        )
    else:
        used_ref_z_bounds = None if ref_z_bounds is None else tuple(float(x) for x in ref_z_bounds)

    if dedup_tol is None:
        dedup_tol = 0.5 * min(dz, dc)

    gaussians = fit_entry.get("gaussians", {})
    seed_points = []
    for name, g in gaussians.items():
        z0 = float(np.clip(g["mu_z"], z_bounds[0], z_bounds[1]))
        c0 = float(np.clip(g["mu_c"], cos_bounds[0], cos_bounds[1]))
        seed_points.append((name, np.array([z0, c0], dtype=float)))
    if max_candidates is not None:
        seed_points = seed_points[: int(max_candidates)]

    minima = []
    seed_results = []

    bounds = [z_bounds, cos_bounds]

    def objective(x):
        values = evaluate_fit_free_energy_derivatives(
            x[0],
            x[1],
            fit_entry,
            temperature=temperature,
            component_normalisations=component_normalisations,
        )
        return float(np.asarray(values["F"]).item())

    def objective_grad(x):
        values = evaluate_fit_free_energy_derivatives(
            x[0],
            x[1],
            fit_entry,
            temperature=temperature,
            component_normalisations=component_normalisations,
        )
        return np.asarray(values["grad"], dtype=float).reshape(2)

    for seed_name, x0 in seed_points:
        result = minimize(
            objective,
            x0=x0,
            jac=objective_grad,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(max_refine_nfev)},
        )

        grad = objective_grad(result.x)
        grad_norm = float(np.linalg.norm(grad))
        z_star, c_star = result.x
        refined = evaluate_fit_free_energy_derivatives(
            z_star,
            c_star,
            fit_entry,
            temperature=temperature,
            component_normalisations=component_normalisations,
        )
        H = np.asarray(refined["hessian"], dtype=float).reshape(2, 2)
        eigvals, eigvecs = np.linalg.eigh(H)
        is_minimum = bool(eigvals[0] > 0 and eigvals[1] > 0 and grad_norm <= grad_tol)

        seed_results.append(
            {
                "seed": seed_name,
                "x0": x0,
                "success": bool(result.success),
                "message": result.message,
                "z": float(z_star),
                "c": float(c_star),
                "F": float(np.asarray(refined["F"]).item()) - F_ref,
                "F_unreferenced": float(np.asarray(refined["F"]).item()),
                "grad_norm": grad_norm,
                "eigvals": eigvals,
                "is_minimum": is_minimum,
            }
        )

        if not is_minimum:
            continue

        duplicate = any(
            abs(z_star - point["z"]) < dedup_tol
            and abs(c_star - point["c"]) < dedup_tol
            for point in minima
        )
        if duplicate:
            continue

        minima.append(
            {
                "z": float(z_star),
                "c": float(c_star),
                "F": float(np.asarray(refined["F"]).item()) - F_ref,
                "F_unreferenced": float(np.asarray(refined["F"]).item()),
                "kind": "minimum",
                "seed": seed_name,
                "grad": grad,
                "grad_norm": grad_norm,
                "hessian": H,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
            }
        )

    return _to_jsonable(
        {
            "z_bounds": z_bounds,
            "cos_bounds": cos_bounds,
            "dz": dz,
            "dc": dc,
            "temperature": float(temperature if temperature is not None else fit_entry.get("temperature", 330.0)),
            "grad_tol": float(grad_tol),
            "sign_tol": float(sign_tol),
            "candidate_mode": candidate_mode,
            "max_candidates": None if max_candidates is None else int(max_candidates),
            "use_component_normalisations": bool(use_component_normalisations),
            "max_refine_nfev": int(max_refine_nfev),
            "reference_to_bulk": bool(reference_to_bulk),
            "free_energy_reference": float(F_ref),
            "ref_z_bounds": used_ref_z_bounds,
            "n_z_grid": None,
            "n_c_grid": None,
            "n_sign_change_cells": 0,
            "n_grad_norm_cells": 0,
            "n_candidate_cells": int(len(seed_points)),
            "component_normalisations": component_normalisations,
            "critical_points": minima,
            "minima": minima,
            "saddles": [],
            "seed_results": seed_results,
        }
    )


def watershed_free_energy_surface(
    free_energy,
    z_values,
    cos_values,
    markers=None,
    minima=None,
    connectivity=8,
    boundary_label=-1,
):
    F = np.asarray(free_energy, dtype=float)
    z_values = np.asarray(z_values)
    cos_values = np.asarray(cos_values)
    if F.shape != (len(z_values), len(cos_values)):
        raise ValueError("free_energy shape must match z_values/cos_values")

    if markers is None:
        if minima is None:
            raise ValueError("Either markers or minima must be supplied")
        markers = _markers_from_minima(minima, z_values, cos_values)
    markers = np.asarray(markers, dtype=int)
    if markers.shape != F.shape:
        raise ValueError("markers must have the same shape as free_energy")

    valid = np.isfinite(F)
    basin_labels = np.zeros(F.shape, dtype=int)
    queue = []
    counter = 0

    marker_indices = np.argwhere((markers > 0) & valid)
    if marker_indices.size == 0:
        raise ValueError("No positive finite markers supplied")

    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ] if connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i, j in marker_indices:
        label = int(markers[i, j])
        if basin_labels[i, j] != 0 and basin_labels[i, j] != label:
            raise ValueError("Multiple marker labels map to the same grid cell")
        basin_labels[i, j] = label
        heapq.heappush(queue, (float(F[i, j]), counter, int(i), int(j)))
        counter += 1

    while queue:
        _, _, i, j = heapq.heappop(queue)
        label = basin_labels[i, j]
        if label <= 0:
            continue

        for di, dj in offsets:
            ni = i + di
            nj = j + dj
            if ni < 0 or ni >= F.shape[0] or nj < 0 or nj >= F.shape[1]:
                continue
            if not valid[ni, nj]:
                continue
            if basin_labels[ni, nj] == 0:
                basin_labels[ni, nj] = label
                heapq.heappush(queue, (float(F[ni, nj]), counter, ni, nj))
                counter += 1

    labels, boundary_mask, boundary_points, boundary_pairs = _extract_watershed_boundaries(
        basin_labels,
        z_values,
        cos_values,
        connectivity=connectivity,
        boundary_label=boundary_label,
    )

    return {
        "labels": labels,
        "basin_labels": basin_labels,
        "boundary_mask": boundary_mask,
        "boundary_points": boundary_points,
        "boundary_pairs": boundary_pairs,
        "markers": markers,
        "connectivity": int(connectivity),
        "boundary_label": int(boundary_label),
    }


def steepest_descent_fit_watershed(
    fit_entry,
    z_values,
    cos_values,
    minima,
    temperature=None,
    connectivity=8,
    boundary_label=-1,
    descent_step=None,
    minima_tol=None,
    descent_grad_tol=1e-7,
    max_descent_steps=1000,
    component_normalisations=None,
):
    if len(z_values) < 2 or len(cos_values) < 2:
        raise ValueError("z_values and cos_values must each contain at least 2 points")
    if not minima:
        raise ValueError("At least one minimum is required for steepest-descent watershed")

    minima_coords = []
    for minimum in minima:
        if isinstance(minimum, dict):
            minima_coords.append([float(minimum["z"]), float(minimum["c"])])
        else:
            minima_coords.append([float(minimum[0]), float(minimum[1])])
    minima_coords = np.asarray(minima_coords, dtype=float)
    minima_labels = np.arange(1, len(minima_coords) + 1, dtype=int)

    dz = float(np.nanmedian(np.diff(z_values)))
    dc = float(np.nanmedian(np.diff(cos_values)))
    if descent_step is None:
        descent_step = 0.5 * min(abs(dz), abs(dc))
    if minima_tol is None:
        minima_tol = np.sqrt(dz**2 + dc**2)
    descent_step = float(descent_step)
    minima_tol = float(minima_tol)
    if descent_step <= 0:
        raise ValueError("descent_step must be positive")
    if minima_tol <= 0:
        raise ValueError("minima_tol must be positive")

    Z, C = np.meshgrid(z_values, cos_values, indexing="ij")
    points = np.column_stack([Z.ravel(), C.ravel()])
    labels_flat = np.zeros(points.shape[0], dtype=int)
    active = np.ones(points.shape[0], dtype=bool)

    def nearest_minimum_labels(coords):
        distances = np.linalg.norm(coords[:, None, :] - minima_coords[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        return minima_labels[nearest], distances[np.arange(coords.shape[0]), nearest]

    for _ in range(int(max_descent_steps)):
        active_idx = np.flatnonzero(active & (labels_flat == 0))
        if active_idx.size == 0:
            break

        coords = points[active_idx]
        nearest_labels, nearest_distances = nearest_minimum_labels(coords)
        close = nearest_distances <= minima_tol
        if np.any(close):
            labels_flat[active_idx[close]] = nearest_labels[close]

        remaining_idx = active_idx[~close]
        if remaining_idx.size == 0:
            continue

        coords = points[remaining_idx]
        values = evaluate_fit_free_energy_derivatives(
            coords[:, 0],
            coords[:, 1],
            fit_entry,
            temperature=temperature,
            component_normalisations=component_normalisations,
        )
        grad = np.asarray(values["grad"], dtype=float).reshape(-1, 2)
        grad_norm = np.linalg.norm(grad, axis=1)
        finite = np.isfinite(grad_norm) & np.all(np.isfinite(grad), axis=1)
        stalled = (~finite) | (grad_norm <= descent_grad_tol)

        if np.any(stalled):
            stalled_labels, _ = nearest_minimum_labels(coords[stalled])
            labels_flat[remaining_idx[stalled]] = stalled_labels

        moving = ~stalled
        if not np.any(moving):
            continue

        move_idx = remaining_idx[moving]
        direction = -grad[moving] / grad_norm[moving, None]
        new_coords = points[move_idx] + descent_step * direction
        new_coords[:, 0] = np.clip(new_coords[:, 0], z_values[0], z_values[-1])
        new_coords[:, 1] = np.clip(new_coords[:, 1], cos_values[0], cos_values[-1])

        no_motion = np.linalg.norm(new_coords - points[move_idx], axis=1) <= 1e-14
        if np.any(no_motion):
            no_motion_labels, _ = nearest_minimum_labels(new_coords[no_motion])
            labels_flat[move_idx[no_motion]] = no_motion_labels
        points[move_idx[~no_motion]] = new_coords[~no_motion]

    unlabelled = labels_flat == 0
    if np.any(unlabelled):
        final_labels, _ = nearest_minimum_labels(points[unlabelled])
        labels_flat[unlabelled] = final_labels

    basin_labels = labels_flat.reshape((len(z_values), len(cos_values)))
    markers = _markers_from_minima(minima, z_values, cos_values)
    labels, boundary_mask, boundary_points, boundary_pairs = _extract_watershed_boundaries(
        basin_labels,
        z_values,
        cos_values,
        connectivity=connectivity,
        boundary_label=boundary_label,
    )

    return {
        "labels": labels,
        "basin_labels": basin_labels,
        "boundary_mask": boundary_mask,
        "boundary_points": boundary_points,
        "boundary_pairs": boundary_pairs,
        "markers": markers,
        "connectivity": int(connectivity),
        "boundary_label": int(boundary_label),
        "descent_step": float(descent_step),
        "minima_tol": float(minima_tol),
        "descent_grad_tol": float(descent_grad_tol),
        "max_descent_steps": int(max_descent_steps),
        "watershed_method": "steepest_descent",
    }


def find_fit_saddles_from_watershed(
    fit_entry,
    watershed,
    temperature=None,
    saddle_grad_tol=1e-5,
    max_saddle_nfev=2000,
    max_boundary_candidates_per_pair=12,
    boundary_candidate_radius=None,
    candidate_min_separation=None,
    dedup_tol=None,
):
    from scipy.optimize import least_squares

    z_values = np.asarray(watershed["z_values"])
    cos_values = np.asarray(watershed["cos_values"])
    z_bounds = tuple(float(x) for x in watershed.get("z_bounds", (z_values[0], z_values[-1])))
    cos_bounds = tuple(float(x) for x in watershed.get("cos_bounds", (cos_values[0], cos_values[-1])))
    dz = float(watershed.get("dz", np.nanmedian(np.diff(z_values))))
    dc = float(watershed.get("dc", np.nanmedian(np.diff(cos_values))))
    grid_diag = float(np.sqrt(dz**2 + dc**2))
    if boundary_candidate_radius is None:
        boundary_candidate_radius = 2.0 * grid_diag
    if candidate_min_separation is None:
        candidate_min_separation = 2.0 * grid_diag
    if dedup_tol is None:
        dedup_tol = grid_diag

    F_ref = float(watershed.get("free_energy_reference", 0.0))
    component_normalisations = watershed.get("component_normalisations")

    def evaluate_at(z, c):
        return evaluate_fit_free_energy_derivatives(
            z,
            c,
            fit_entry,
            temperature=temperature,
            component_normalisations=component_normalisations,
        )

    def gradient_residual(x):
        values = evaluate_at(x[0], x[1])["grad"]
        return np.asarray(values, dtype=float).reshape(2)

    saddles = []
    saddle_candidates = {}

    for pair_key, boundary_points in watershed.get("boundary_pairs", {}).items():
        points = np.asarray(boundary_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
            saddle_candidates[pair_key] = []
            continue

        evals = evaluate_at(points[:, 0], points[:, 1])
        energies = np.asarray(evals["F"], dtype=float).reshape(-1)
        candidates = _select_boundary_saddle_candidates(
            points,
            energies,
            radius=float(boundary_candidate_radius),
            max_candidates=max_boundary_candidates_per_pair,
            min_separation=float(candidate_min_separation),
        )
        saddle_candidates[pair_key] = candidates

        for x0 in candidates:
            result = least_squares(
                gradient_residual,
                x0=np.asarray(x0, dtype=float),
                bounds=([z_bounds[0], cos_bounds[0]], [z_bounds[1], cos_bounds[1]]),
                max_nfev=int(max_saddle_nfev),
            )
            if not result.success:
                continue

            grad = gradient_residual(result.x)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > saddle_grad_tol:
                continue

            z_star, c_star = result.x
            duplicate = any(
                abs(z_star - point["z"]) < dedup_tol
                and abs(c_star - point["c"]) < dedup_tol
                for point in saddles
            )
            if duplicate:
                continue

            refined = evaluate_at(z_star, c_star)
            H = np.asarray(refined["hessian"], dtype=float).reshape(2, 2)
            eigvals, eigvecs = np.linalg.eigh(H)
            if not (eigvals[0] < 0 and eigvals[1] > 0):
                continue

            F_unreferenced = float(np.asarray(refined["F"]).item())
            saddles.append(
                {
                    "z": float(z_star),
                    "c": float(c_star),
                    "F": F_unreferenced - F_ref,
                    "F_unreferenced": F_unreferenced,
                    "kind": "saddle",
                    "boundary_pair": pair_key,
                    "x0": np.asarray(x0, dtype=float),
                    "grad": grad,
                    "grad_norm": grad_norm,
                    "hessian": H,
                    "eigvals": eigvals,
                    "eigvecs": eigvecs,
                }
            )

    saddles = sorted(saddles, key=lambda point: point["F"])
    return _to_jsonable(
        {
            "saddles": saddles,
            "saddle_candidates": saddle_candidates,
            "saddle_grad_tol": float(saddle_grad_tol),
            "max_saddle_nfev": int(max_saddle_nfev),
            "max_boundary_candidates_per_pair": None
            if max_boundary_candidates_per_pair is None
            else int(max_boundary_candidates_per_pair),
            "boundary_candidate_radius": float(boundary_candidate_radius),
            "candidate_min_separation": float(candidate_min_separation),
            "dedup_tol": float(dedup_tol),
            "free_energy_reference": F_ref,
        }
    )


def compute_fit_watershed(
    fit_entry,
    z_bounds,
    cos_bounds=(-1.0, 1.0),
    dz=0.02,
    dc=0.02,
    minima=None,
    markers=None,
    temperature=None,
    connectivity=8,
    boundary_label=-1,
    use_component_normalisations=True,
    descent_step=None,
    minima_tol=None,
    descent_grad_tol=1e-7,
    max_descent_steps=1000,
    ref_z_bounds=None,
    reference_to_bulk=True,
    find_saddles=False,
    saddle_grad_tol=1e-5,
    max_saddle_nfev=2000,
    max_boundary_candidates_per_pair=12,
    boundary_candidate_radius=None,
    candidate_min_separation=None,
    saddle_dedup_tol=None,
    **minima_kwargs,
):
    z_bounds = tuple(float(x) for x in z_bounds)
    cos_bounds = tuple(float(x) for x in cos_bounds)
    dz = float(dz)
    dc = float(dc)
    z_values = _regular_grid_from_spacing(z_bounds, dz)
    cos_values = _regular_grid_from_spacing(cos_bounds, dc)

    component_normalisations = None
    if use_component_normalisations:
        component_normalisations = fit_component_normalisations(
            fit_entry,
            z_bounds,
            cos_bounds,
            dz,
            dc,
        )

    Z, C = np.meshgrid(z_values, cos_values, indexing="ij")
    fit_eval = evaluate_fit_free_energy_derivatives(
        Z,
        C,
        fit_entry,
        temperature=temperature,
        component_normalisations=component_normalisations,
    )
    F_unreferenced = np.asarray(fit_eval["F"], dtype=float)

    minima_result = None
    if markers is None and minima is None:
        minima_result = find_fit_minima(
            fit_entry,
            z_bounds=z_bounds,
            cos_bounds=cos_bounds,
            dz=dz,
            dc=dc,
            temperature=temperature,
            use_component_normalisations=use_component_normalisations,
            ref_z_bounds=ref_z_bounds,
            reference_to_bulk=reference_to_bulk,
            **minima_kwargs,
        )
        minima = minima_result["minima"]

    F_ref = 0.0
    used_ref_z_bounds = None
    if reference_to_bulk:
        F_ref, used_ref_z_bounds = _bulk_reference_free_energy_from_fit(
            fit_entry,
            cos_values,
            dz,
            temperature=temperature,
            ref_z_bounds=ref_z_bounds,
            component_normalisations=component_normalisations,
        )
    else:
        used_ref_z_bounds = None if ref_z_bounds is None else tuple(float(x) for x in ref_z_bounds)

    F = F_unreferenced - F_ref
    if minima is not None and minima_result is None:
        minima = _shift_minima_free_energies(minima, F_ref)
    if minima_result is not None:
        minima_result = dict(minima_result)
        minima_result["minima"] = minima
        minima_result["critical_points"] = minima
        minima_result["free_energy_reference"] = float(F_ref)
        minima_result["ref_z_bounds"] = used_ref_z_bounds

    if markers is not None and minima is None:
        marker_indices = np.argwhere(np.asarray(markers, dtype=int) > 0)
        minima = [
            {
                "z": float(z_values[i]),
                "c": float(cos_values[j]),
                "marker_label": int(markers[i, j]),
            }
            for i, j in marker_indices
        ]

    watershed = steepest_descent_fit_watershed(
        fit_entry,
        z_values,
        cos_values,
        minima,
        temperature=temperature,
        connectivity=connectivity,
        boundary_label=boundary_label,
        descent_step=descent_step,
        minima_tol=minima_tol,
        descent_grad_tol=descent_grad_tol,
        max_descent_steps=max_descent_steps,
        component_normalisations=component_normalisations,
    )
    watershed.update(
        {
            "F": F,
            "F_unreferenced": F_unreferenced,
            "free_energy_reference": float(F_ref),
            "ref_z_bounds": used_ref_z_bounds,
            "reference_to_bulk": bool(reference_to_bulk),
            "z_values": z_values,
            "cos_values": cos_values,
            "z_bounds": z_bounds,
            "cos_bounds": cos_bounds,
            "dz": dz,
            "dc": dc,
            "minima": minima,
            "minima_result": minima_result,
            "component_normalisations": component_normalisations,
        }
    )
    if find_saddles:
        saddle_result = find_fit_saddles_from_watershed(
            fit_entry,
            watershed,
            temperature=temperature,
            saddle_grad_tol=saddle_grad_tol,
            max_saddle_nfev=max_saddle_nfev,
            max_boundary_candidates_per_pair=max_boundary_candidates_per_pair,
            boundary_candidate_radius=boundary_candidate_radius,
            candidate_min_separation=candidate_min_separation,
            dedup_tol=saddle_dedup_tol,
        )
        watershed["saddles"] = saddle_result["saddles"]
        watershed["saddle_result"] = saddle_result
    else:
        watershed["saddles"] = []
        watershed["saddle_result"] = None
    return watershed


def find_fit_minima(*args, **kwargs):
    return find_fit_stationary_points(*args, **kwargs)
