import os
import re

import numpy as np

from .species_histogram import (
    fit_parameters_to_probability_density,
    histogram_to_probability_density,
    probability_to_free_energy,
)
from .species_utils import (
    _contact_layer_bounds,
    _density_scale_from_profile,
    _max_contact_layer_density,
    _marginal_costheta_in_z_window,
    _marginal_z,
)


def plot_species_fit_summary(
    material,
    histogram_entry,
    fit_entry,
    density_profile_entry,
    output_path,
    z_xlim=None,
    free_energy_vmin_percentile=2,
    free_energy_vmax_percentile=98,
    cmap="viridis",
    dpi=300,
):
    import matplotlib.pyplot as plt

    z_bins = np.asarray(histogram_entry["z_bins"])
    cos_bins = np.asarray(histogram_entry["cos_bins"])
    if "P_data" in fit_entry:
        probability_data = np.asarray(fit_entry["P_data"], dtype=float)
    else:
        probability_data = histogram_to_probability_density(
            z_bins,
            cos_bins,
            histogram_entry["counts"],
        )

    if "P_fit" in fit_entry:
        probability_fit = np.asarray(fit_entry["P_fit"], dtype=float)
    else:
        probability_fit = fit_parameters_to_probability_density(
            z_bins,
            cos_bins,
            fit_entry,
        )

    if "F_data" in fit_entry:
        free_energy_data = np.ma.masked_invalid(np.ma.asarray(fit_entry["F_data"], dtype=float))
    else:
        free_energy_data = probability_to_free_energy(
            probability_data,
            z_bins,
            temperature=fit_entry.get("temperature", 330.0),
            ref_z_bounds=fit_entry.get("ref_z_bounds", (6.0, 8.0)),
        )

    if "F_fit" in fit_entry:
        free_energy_fit = np.ma.masked_invalid(np.ma.asarray(fit_entry["F_fit"], dtype=float))
    else:
        free_energy_fit = probability_to_free_energy(
            probability_fit,
            z_bins,
            temperature=fit_entry.get("temperature", 330.0),
            ref_z_bounds=fit_entry.get("ref_z_bounds", (6.0, 8.0)),
        )

    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    cos_centers = 0.5 * (cos_bins[:-1] + cos_bins[1:])
    z_contact = _contact_layer_bounds(density_profile_entry)

    if z_xlim is None:
        z_xlim = (z_contact[0], min(z_bins[-1], max(6.0, z_contact[1])))

    z_scale_mask = z_centers <= z_xlim[1]
    finite_f = free_energy_data[z_scale_mask, :].compressed()
    vmin = np.nanpercentile(finite_f, free_energy_vmin_percentile)
    vmax = np.nanpercentile(finite_f, free_energy_vmax_percentile)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    mesh0 = axes[0, 0].pcolormesh(
        z_bins,
        cos_bins,
        free_energy_data.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title("Raw free energy")
    axes[0, 0].set_xlabel(r"$z$ [A]")
    axes[0, 0].set_ylabel(r"$\cos(\theta_\mathrm{dip})$")
    axes[0, 0].set_xlim(*z_xlim)
    axes[0, 0].set_ylim(-1, 1)
    fig.colorbar(mesh0, ax=axes[0, 0], label=r"$F$ [eV]")

    mesh1 = axes[0, 1].pcolormesh(
        z_bins,
        cos_bins,
        free_energy_fit.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title("Fitted free energy")
    axes[0, 1].set_xlabel(r"$z$ [A]")
    axes[0, 1].set_ylabel(r"$\cos(\theta_\mathrm{dip})$")
    axes[0, 1].set_xlim(*z_xlim)
    axes[0, 1].set_ylim(-1, 1)
    fig.colorbar(mesh1, ax=axes[0, 1], label=r"$F_\mathrm{fit}$ [eV]")

    density_scale = _density_scale_from_profile(density_profile_entry)
    max_contact_density = _max_contact_layer_density(density_profile_entry, z_contact)
    pz_data = _marginal_z(probability_data, z_bins, cos_bins) * density_scale
    pz_fit = _marginal_z(probability_fit, z_bins, cos_bins) * density_scale
    axes[1, 0].plot(z_centers, pz_data, "o-", markersize=3, label="Raw")
    axes[1, 0].plot(z_centers, pz_fit, "-", linewidth=2, label="Fit")
    axes[1, 0].axvspan(*z_contact, color="0.85", zorder=0)
    axes[1, 0].set_title(r"$\rho(z)$")
    axes[1, 0].set_xlabel(r"$z$ [A]")
    axes[1, 0].set_ylabel(r"$\rho_\mathrm{O}(z)$ [g cm$^{-3}$]")
    axes[1, 0].set_xlim(*z_xlim)
    axes[1, 0].set_ylim(0, 1.2 * max_contact_density)
    axes[1, 0].legend()

    pcos_data = _marginal_costheta_in_z_window(
        probability_data,
        z_bins,
        cos_bins,
        z_contact,
    )
    pcos_fit = _marginal_costheta_in_z_window(
        probability_fit,
        z_bins,
        cos_bins,
        z_contact,
    )
    axes[1, 1].plot(cos_centers, pcos_data, "o-", markersize=3, label="Raw")
    axes[1, 1].plot(cos_centers, pcos_fit, "-", linewidth=2, label="Fit")
    axes[1, 1].set_title(
        rf"$P(\cos\theta_\mathrm{{dip}})$, "
        rf"${z_contact[0]:.2f} \leq z \leq {z_contact[1]:.2f}$ A"
    )
    axes[1, 1].set_xlabel(r"$\cos(\theta_\mathrm{dip})$")
    axes[1, 1].set_ylabel(r"$P(\cos\theta_\mathrm{dip})$")
    axes[1, 1].set_xlim(-1, 1)
    axes[1, 1].legend()

    fig.suptitle(f"{material}: dipole species fit", fontsize=14)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_all_species_fit_summaries(
    histogram_data,
    fit_results,
    density_profile_data,
    output_dir,
    filename_template="{material}_species_fit_summary.png",
    **plot_kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}
    for material, fit_entry in fit_results.items():
        if material not in histogram_data:
            raise KeyError(f"No histogram data found for {material}")
        if material not in density_profile_data:
            raise KeyError(f"No density profile data found for {material}")

        filename = filename_template.format(
            material=re.sub(r"[^A-Za-z0-9_.-]+", "_", material)
        )
        output_path = os.path.join(output_dir, filename)
        output_paths[material] = plot_species_fit_summary(
            material,
            histogram_data[material],
            fit_entry,
            density_profile_data[material],
            output_path,
            **plot_kwargs,
        )

    return output_paths
