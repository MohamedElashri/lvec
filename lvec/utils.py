# utils.py
import numpy as np
from .backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                      backend_sin, backend_cos, backend_atan2,
                      backend_sinh, backend_cosh, backend_log,
                      backend_where)
from .exceptions import ShapeError, BackendError

def ensure_array(*args):
    """Normalize components, broadcasting scalars to a common array structure.

    Non-scalar component arrays must have identical shapes and nested structure.
    Length-one arrays are arrays, not scalars, and are never expanded implicitly.
    Python/NumPy scalars and zero-dimensional NumPy arrays are broadcast to the
    shape of the array components. Awkward missing values propagate to scalars.
    """
    try:
        if any(x is None for x in args):
            raise ValueError("Cannot process None values")

        def is_scalar(value):
            return np.isscalar(value) or (
                isinstance(value, np.ndarray) and value.ndim == 0
            )

        # Reject strings, dictionaries, complex and object arrays before shape
        # validation or arithmetic, with one consistent initialization error.
        for arg in args:
            if not is_ak(arg) and np.asarray(arg).dtype.kind not in 'biuf':
                raise TypeError("Vector components must contain real numeric values")
        use_ak = any(is_ak(arg) for arg in args)
        scalars = [is_scalar(arg) for arg in args]
        if all(scalars):
            return (*[np.asarray(arg).item() for arg in args], 'np')

        if use_ak:
            import awkward as ak
            values = [np.asarray(arg).item() if scalar else to_ak(arg)
                      for arg, scalar in zip(args, scalars)]
            check_shapes(*[arg for arg, scalar in zip(values, scalars)
                           if not scalar], 'ak')
            # Only scalar values are allowed to expand. Existing array shapes
            # were checked above, so broadcasting cannot change particle counts.
            arrays = ak.broadcast_arrays(*values) if any(scalars) else values
            return (*arrays, 'ak')

        arrays = [np.asarray(arg) for arg in args]
        non_scalars = [arg for arg, scalar in zip(arrays, scalars) if not scalar]
        check_shapes(*non_scalars, 'np')
        shape = non_scalars[0].shape
        arrays = [np.broadcast_to(arg, shape) if scalar else arg
                  for arg, scalar in zip(arrays, scalars)]
        return (*arrays, 'np')
    except ShapeError:
        raise
    except Exception as e:
        raise BackendError("initialization", "unknown", str(e)) from e
    

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
    """Return signed sqrt(E²-p²), including spacelike four-vectors."""
    m2 = E*E - p*p
    return np.sign(m2) * np.sqrt(abs(m2))


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