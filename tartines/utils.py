
import numpy as np


def _parse_meta_value(s):
    """Turn a metadata value string into float or np.ndarray if possible."""
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        return np.fromstring(s[1:-1], sep=' ')
    try:
        return float(s)
    except ValueError:
        return s

def load_density_profile_data(path):
    """
        Load a density profile file consisting of a metadata header and
        tabulated z–density data.

        The first line of the file contains metadata in the form:
            key:value;key:value;...

        Expected metadata fields include:
        - contact_layer_start: <float>
        - contact_layer_end: <float>
        - peaks: [z1 z2 ...]
        - troughs: [z1 z2 ...]
        - v_1: [ax ay az]    # lattice vector 1
        - v_2: [bx by bz]    # lattice vector 2
        - v_3: [cx cy cz]    # lattice vector 3

        All bracketed values are parsed as NumPy arrays.
        Scalar numeric values are parsed as floats.

        Returns
        -------
        meta : dict
            Parsed metadata fields with proper Python types.
        z : np.ndarray
            z-coordinates.
        O_density : np.ndarray
            Oxygen density profile.
        H_density : np.ndarray
            Hydrogen density profile.
    """

    with open(path) as f:
        meta_line = f.readline().strip()
        meta = {}
        for field in meta_line.split(';'):
            if not field:
                continue
            key, val = field.split(':', 1)
            meta[key.strip()] = _parse_meta_value(val)

    # z, O_density, H_density as separate arrays
    z, O_density, H_density = np.loadtxt(
        path, delimiter=',', skiprows=2, unpack=True
    )

    return meta, z, O_density, H_density
