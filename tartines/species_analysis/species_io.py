import json
import os

from .species_utils import _compact_fit_parameters, _normalise_name, _to_jsonable


def load_species_initial_guess_groups(guess_json_path):
    with open(guess_json_path) as handle:
        return json.load(handle)


def _iter_guess_group_materials(group_materials):
    if isinstance(group_materials, str):
        group_materials = [group_materials]

    for item in group_materials:
        for material in str(item).split(","):
            material = _normalise_name(material.strip())
            if material:
                yield material


def expand_species_initial_guess_groups(guess_groups, materials=None):
    material_filter = (
        {_normalise_name(material) for material in materials}
        if materials is not None
        else None
    )
    fit_peak_initial_guesses = {}
    step_initial_guesses = {}

    for group_name, group in guess_groups.items():
        if "materials" not in group:
            raise KeyError(f"Guess group {group_name!r} is missing 'materials'")
        if "guess" not in group:
            raise KeyError(f"Guess group {group_name!r} is missing 'guess'")

        for material in _iter_guess_group_materials(group["materials"]):
            if material_filter is not None and material not in material_filter:
                continue
            if material in fit_peak_initial_guesses:
                raise ValueError(f"Multiple initial-guess groups apply to {material!r}")

            fit_peak_initial_guesses[material] = group["guess"]
            if "step_guess" in group and group["step_guess"] is not None:
                step_initial_guesses[material] = group["step_guess"]

    return fit_peak_initial_guesses, step_initial_guesses


def save_fit_parameters(fit_results, output_json_path):
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as handle:
        json.dump(_to_jsonable(_compact_fit_parameters(fit_results)), handle, indent=2)


def load_fit_parameters(input_json_path):
    with open(input_json_path) as handle:
        return json.load(handle)
