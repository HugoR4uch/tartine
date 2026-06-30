"""Species-analysis tools for dipole histogram fitting and plotting."""

from .species_fit import (
    correlated_gaussian_basis,
    evaluate_fit_free_energy_derivatives,
    fit_all_material_gaussian_species,
    fit_component_normalisations,
    smooth_left_exponential_step_basis,
)
from .species_histogram import (
    fit_parameters_to_probability_density,
    histogram_to_free_energy,
    histogram_to_probability_density,
    load_dipole_costheta_z_histogram_data,
    probability_to_free_energy,
)
from .species_io import (
    expand_species_initial_guess_groups,
    load_fit_parameters,
    load_species_initial_guess_groups,
    save_fit_parameters,
)
from .species_plotting import (
    plot_all_species_fit_summaries,
    plot_species_fit_summary,
)
from .species_watershed import (
    compute_fit_watershed,
    find_fit_minima,
    find_fit_saddles_from_watershed,
    find_fit_stationary_points,
    steepest_descent_fit_watershed,
    watershed_free_energy_surface,
)

__all__ = [
    "correlated_gaussian_basis",
    "compute_fit_watershed",
    "evaluate_fit_free_energy_derivatives",
    "expand_species_initial_guess_groups",
    "find_fit_minima",
    "find_fit_saddles_from_watershed",
    "find_fit_stationary_points",
    "fit_all_material_gaussian_species",
    "fit_component_normalisations",
    "fit_parameters_to_probability_density",
    "histogram_to_free_energy",
    "histogram_to_probability_density",
    "load_dipole_costheta_z_histogram_data",
    "load_fit_parameters",
    "load_species_initial_guess_groups",
    "plot_all_species_fit_summaries",
    "plot_species_fit_summary",
    "probability_to_free_energy",
    "save_fit_parameters",
    "smooth_left_exponential_step_basis",
    "steepest_descent_fit_watershed",
    "watershed_free_energy_surface",
]
