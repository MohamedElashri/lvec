# utils.py
import numpy as np
from .backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                      backend_sin, backend_cos, backend_atan2,
                      backend_sinh, backend_cosh, backend_log,
                      backend_where)
from .exceptions import ShapeError, BackendError

def ensure_array(*args):
    """
    Convert inputs to consistent array type.
    
    For Awkward arrays, this function will ensure that not only the outer dimensions 
    match, but also that any jagged (variable-length) inner dimensions are consistent
    when used in vectorized operations.
    
    Returns:
        tuple: Contains the input arrays converted to a consistent type,
               followed by the backend library indicator ('np' or 'ak')
    """
    try:
        # Check for None values first
        if any(x is None for x in args):
            raise ValueError("Cannot process None values")
            
        # Determine if we need to use Awkward
        use_ak = any(is_ak(arg) for arg in args)
        
        if use_ak:
            # Convert all inputs to Awkward arrays
            arrays = [to_ak(arg) for arg in args]
            lib = 'ak'
        else:
            # For NumPy backend:
            # If all inputs are scalars, keep them as scalars
            all_scalars = all(isinstance(arg, (float, int)) for arg in args)
            
            if all_scalars:
                arrays = list(args)  # Keep scalars as they are
            else:
                # Mixed scalar/array inputs or all arrays
                arrays = []
                for arg in args:
                    if isinstance(arg, (float, int)):
                        # Convert scalar to array if mixed with arrays
                        arrays.append(to_np(arg))
                    else:
                        arrays.append(to_np(arg))
            lib = 'np'
            
        return (*arrays, lib)
    except Exception as e:
        raise BackendError("initialization", "unknown", str(e))
    
def check_shapes(*arrays):
    """Require equal shapes and missing-value positions at every nesting depth.

    The last argument names the backend. Awkward validation uses array kernels
    over per-axis counts and masks, never a Python loop over events or leaves.
    Scalar broadcasting is the responsibility of ``ensure_array``.
    """
    lib = arrays[-1]
    arrays = arrays[:-1]
    if len(arrays) < 2:
        return

    if lib != 'ak':
        if any(getattr(getattr(arr, 'dtype', None), 'kind', '') in ('O', 'U', 'S', 'V')
               for arr in arrays):
            raise TypeError('Vector components must be numeric arrays')
        shapes = [getattr(arr, 'shape', ()) for arr in arrays]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ShapeError("Inconsistent array shapes in NumPy arrays",
                             shapes=[f"arrays[{i}]: {shape}" for i, shape in enumerate(shapes)])
        return

    import awkward as ak
    shape_info = [f"arrays[{i}]: length {len(arr)}, depth {arr.ndim}"
                  for i, arr in enumerate(arrays)]
    reference = arrays[0]
    if any(len(arr) != len(reference) for arr in arrays[1:]):
        raise ShapeError("Inconsistent array lengths in Awkward arrays", shapes=shape_info)
    if any(arr.ndim != reference.ndim for arr in arrays[1:]):
        raise ShapeError("Inconsistent nesting depths in Awkward arrays", shapes=shape_info)

    def equal_structure(left, right):
        return ak.almost_equal(left, right, rtol=0, atol=0,
                               check_parameters=False, check_regular=False)

    # Each count array includes the full parent structure, including parent
    # missing values. The final mask covers missing numeric leaves as well.
    try:
        for axis in range(1, reference.ndim):
            counts = ak.num(reference, axis=axis)
            if any(not equal_structure(counts, ak.num(arr, axis=axis)) for arr in arrays[1:]):
                raise ShapeError(f"Inconsistent inner dimensions in Awkward arrays at axis {axis}",
                                 shapes=shape_info)
        mask = ak.is_none(reference, axis=-1)
        if any(not equal_structure(mask, ak.is_none(arr, axis=-1)) for arr in arrays[1:]):
            raise ShapeError("Inconsistent missing-value positions in Awkward arrays", shapes=shape_info)
    except (ValueError, TypeError, IndexError) as exc:
        raise ShapeError("Incompatible nesting structure in Awkward arrays", shapes=shape_info) from exc


def compute_pt(px, py, lib):
    """
    Compute transverse momentum.
    
    Parameters
    ----------
    px, py : scalar or array-like
        X and Y components of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Transverse momentum with the same type as input
    """
    # Use the appropriate backend for the square root
    pt = backend_sqrt(px*px + py*py, lib)
    
    # Convert to scalar only if both inputs were scalars
    if isinstance(px, (float, int)) and isinstance(py, (float, int)):
        return float(pt)
    return pt

def compute_p(px, py, pz, lib):
    """
    Compute total momentum.
    
    Parameters
    ----------
    px, py, pz : scalar or array-like
        X, Y, and Z components of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Total momentum with the same type as input
    """
    # Use the appropriate backend for the square root
    p = backend_sqrt(px*px + py*py + pz*pz, lib)
    
    # Convert to scalar only if all inputs were scalars
    if isinstance(px, (float, int)) and isinstance(py, (float, int)) and isinstance(pz, (float, int)):
        return float(p)
    return p


def compute_mass(E, p, lib):
    """
    Compute mass from energy and momentum.
    
    For physical particles, m² = E² - p² should be positive. When negative values
    are encountered due to numerical inaccuracies, a warning is issued and the
    absolute value is used.
    
    Parameters
    ----------
    E : scalar or array-like
        Energy
    p : scalar or array-like
        Momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Mass with the same type as input
    """
    import warnings
    m2 = E*E - p*p
    
    # Check for negative m² values
    if isinstance(m2, (float, int)):
        if m2 < 0:
            warnings.warn(f"Negative m² value encountered: {m2}. Taking absolute value, but this may indicate unphysical particles or numerical issues.")
            m2 = abs(m2)
    else:
        # For array inputs, check if any values are negative
        has_negative = False
        
        if lib == 'ak' and HAS_AWKWARD:
            # For Awkward arrays
            try:
                import awkward as ak
                # Use ak.any for Awkward arrays
                has_negative = ak.any(m2 < 0)
            except Exception:
                # Fallback if ak.any doesn't work
                has_negative = any(m2 < 0)
        else:
            # For NumPy arrays
            if hasattr(m2, 'any'):
                has_negative = (m2 < 0).any()
            else:
                # For other iterable types
                try:
                    has_negative = any(m2 < 0)
                except TypeError:
                    # Not iterable
                    has_negative = False
            
        if has_negative:
            warnings.warn("Negative m² values encountered. Taking absolute values, but this may indicate unphysical particles or numerical issues.")
            m2 = backend_where(m2 < 0, abs(m2), m2, lib)
    
    m = backend_sqrt(m2, lib)
    
    # Convert to scalar only if all inputs were scalars
    if isinstance(E, (float, int)) and isinstance(p, (float, int)):
        return float(m)
    return m


def compute_eta(p, pz, lib):
    """
    Compute pseudorapidity using the numerically stable formula:
    
        η = 0.5 * ln((p + pz) / (p - pz + ε))
    
    where ε is a small constant to avoid division by zero.
    
    Parameters
    ----------
    p : scalar or array-like
        Total momentum
    pz : scalar or array-like
        Z component of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Pseudorapidity with the same type as input
    """
    epsilon = 1e-10
    eta = 0.5 * backend_log((p + pz) / (p - pz + epsilon), lib)
    
    # Convert to scalar only if all inputs were scalars
    if isinstance(p, (float, int)) and isinstance(pz, (float, int)):
        return float(eta)
    return eta


def compute_phi(px, py, lib):
    """
    Compute azimuthal angle.
    
    Parameters
    ----------
    px, py : scalar or array-like
        X and Y components of momentum
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    scalar or array-like
        Azimuthal angle with the same type as input
    """
    phi = backend_atan2(py, px, lib)
    
    # Convert to scalar only if all inputs were scalars
    if isinstance(px, (float, int)) and isinstance(py, (float, int)):
        return float(phi)
    return phi

def compute_p4_from_ptepm(pt, eta, phi, m, lib):
    """
    Convert pt, eta, phi, mass to px, py, pz, E.
    
    Parameters
    ----------
    pt : scalar or array-like
        Transverse momentum
    eta : scalar or array-like
        Pseudorapidity
    phi : scalar or array-like
        Azimuthal angle
    m : scalar or array-like
        Mass
    lib : str
        Backend library ('np' or 'ak')
        
    Returns
    -------
    tuple
        (px, py, pz, E) with the same type as input
    """
    px = pt * backend_cos(phi, lib)
    py = pt * backend_sin(phi, lib)
    pz = pt * backend_sinh(eta, lib)
    E = backend_sqrt(pt*pt * backend_cosh(eta, lib)**2 + m*m, lib)
    
    # Convert to scalar only if all inputs were scalars
    if all(isinstance(x, (float, int)) for x in [pt, eta, phi, m]):
        return float(px), float(py), float(pz), float(E)
    return px, py, pz, E