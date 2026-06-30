import inspect

import numpy as np

from tartines.species_analysis import species_watershed


def test_fit_watershed_defaults_to_component_normalised_surface():
    assert (
        inspect.signature(
            species_watershed.find_fit_stationary_points
        ).parameters["use_component_normalisations"].default
        is True
    )
    assert (
        inspect.signature(
            species_watershed.compute_fit_watershed
        ).parameters["use_component_normalisations"].default
        is True
    )


def test_compute_fit_watershed_passes_component_normalisations_by_default(monkeypatch):
    normalisations = {"state_1": 12.3}
    seen_normalisations = []

    def fake_normalisations(fit_entry, z_bounds, cos_bounds, dz, dc):
        return normalisations

    def fake_eval(z, c, fit_entry, temperature=None, component_normalisations=None):
        seen_normalisations.append(component_normalisations)
        z_arr, c_arr = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(c, dtype=float))
        return {
            "F": np.zeros_like(z_arr, dtype=float),
            "grad": np.zeros(z_arr.shape + (2,), dtype=float),
        }

    monkeypatch.setattr(
        species_watershed,
        "fit_component_normalisations",
        fake_normalisations,
    )
    monkeypatch.setattr(
        species_watershed,
        "evaluate_fit_free_energy_derivatives",
        fake_eval,
    )

    result = species_watershed.compute_fit_watershed(
        {"temperature": 330.0},
        z_bounds=(0.0, 1.0),
        cos_bounds=(-1.0, 1.0),
        dz=0.5,
        dc=1.0,
        minima=[{"z": 0.5, "c": 0.0}],
        reference_to_bulk=False,
    )

    assert result["component_normalisations"] == normalisations
    assert seen_normalisations
    assert all(item == normalisations for item in seen_normalisations)
