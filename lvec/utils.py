# utils.py
import numpy as np
from .backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                      backend_sin, backend_cos, backend_atan2,
                      backend_sinh, backend_cosh)
from .exceptions import ShapeError

def ensure_array(*args):
    """Convert inputs to consistent array type."""
    use_ak = any(is_ak(arg) for arg in args)
    if use_ak:
        arrays = [to_ak(arg) for arg in args]
        lib = 'ak'
    else:
        arrays = [to_np(arg) if not isinstance(arg, (float, int)) else np.array([arg]) for arg in args]
        lib = 'np'
    return (*arrays, lib)

def check_shapes(*arrays):
    """Verify all arrays have consistent shapes."""
    lib = arrays[-1]  # Last argument is the library type
    arrays = arrays[:-1]  # All but the last argument are arrays
    
    if lib == 'ak':
        lengths = [len(arr) for arr in arrays]
        if not all(l == lengths[0] for l in lengths):
            raise ShapeError("Inconsistent array lengths")
    else:
        shapes = [arr.shape for arr in arrays]
        if not all(s == shapes[0] for s in shapes):
            raise ShapeError("Inconsistent array shapes")

def compute_pt(px, py, lib):
    """Compute transverse momentum."""
    return backend_sqrt(px*px + py*py, lib)

def compute_p(px, py, pz, lib):
    """Compute total momentum."""
    return backend_sqrt(px*px + py*py + pz*pz, lib)

def compute_mass(E, p, lib):
    """Compute mass from energy and momentum."""
    m2 = E*E - p*p
    return backend_sqrt(m2 * (m2 > 0), lib)

def compute_eta(p, pz, lib):
    """Compute pseudorapidity."""
    pt = compute_pt(p, pz, lib)
    return backend_atan2(pt, pz, lib)

def compute_phi(px, py, lib):
    """Compute azimuthal angle."""
    return backend_atan2(py, px, lib)

def compute_p4_from_ptepm(pt, eta, phi, m, lib):
    """Convert pt, eta, phi, mass to px, py, pz, E."""
    px = pt * backend_cos(phi, lib)
    py = pt * backend_sin(phi, lib)
    pz = pt * backend_sinh(eta, lib)
    E = backend_sqrt(pt*pt * backend_cosh(eta, lib)**2 + m*m, lib)
    return px, py, pz, E