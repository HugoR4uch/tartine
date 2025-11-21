
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
