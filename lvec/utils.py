# utils.py
import numpy as np
from .backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                      backend_sin, backend_cos, backend_atan2,
                      backend_sinh, backend_cosh, backend_log)
from .exceptions import ShapeError, BackendError

def ensure_array(*args):
    """
    Convert inputs to consistent array type.
    
    For Awkward arrays, this function will ensure that not only the outer dimensions 
    match, but also that any jagged (variable-length) inner dimensions are consistent
    when used in vectorized operations.
    """
    try:
        # Check for None values first
        if any(x is None for x in args):
            raise ValueError("Cannot process None values")
            
        use_ak = any(is_ak(arg) for arg in args)
        if use_ak:
            arrays = [to_ak(arg) for arg in args]
            lib = 'ak'
        else:
            # Only convert to numpy array if not scalar
            arrays = []
            for arg in args:
                if isinstance(arg, (float, int)):
                    arrays.append(arg)
                else:
                    arrays.append(to_np(arg))
            lib = 'np'
        return (*arrays, lib)
    except Exception as e:
        raise BackendError("initialization", "unknown", str(e))
    
def check_shapes(*arrays):
    """
    Verify all arrays have consistent shapes.
    
    Parameters
    ----------
    *arrays : array-like or scalar
        Arrays to check, with the last element being the library type
    """
    lib = arrays[-1]
    arrays = arrays[:-1]
    
    # Get shape information for error reporting
    shape_info = []
    for i, arr in enumerate(arrays):
        if isinstance(arr, (float, int)):
            shape_info.append(f"arrays[{i}]: scalar")
        elif arr is None:
            shape_info.append(f"arrays[{i}]: None")
        else:
            shape = getattr(arr, 'shape', None) or len(arr)
            shape_info.append(f"arrays[{i}]: {shape}")
    
    # If all inputs are scalars, they're compatible
    if all(isinstance(arr, (float, int)) or arr is None for arr in arrays):
        return
        
    if lib == 'ak':
        # First check outer lengths
        array_lengths = [len(arr) if not isinstance(arr, (float, int)) and arr is not None else 1 
                        for arr in arrays]
        if not all(l == array_lengths[0] for l in array_lengths):
            raise ShapeError(
                "Inconsistent array lengths in Awkward arrays",
                shapes=shape_info
            )
        
        # Filter out scalars and None values
        valid_arrays = [arr for arr in arrays if not isinstance(arr, (float, int)) and arr is not None]
        
        if valid_arrays and all(is_ak(arr) for arr in valid_arrays):
            import awkward as ak
            
            # Check if arrays are jagged by examining their structure
            is_jagged = False
            for arr in valid_arrays:
                if any(isinstance(arr[i], ak.Array) and len(arr[i]) > 0 for i in range(len(arr))):
                    is_jagged = True
                    break
            
            # For jagged arrays, we need to verify that the inner dimensions are consistent
            if is_jagged:
                # For each outer index
                for i in range(len(valid_arrays[0])):
                    lengths = []
                    
                    # Get the length of each array at this index
                    for arr in valid_arrays:
                        try:
                            if isinstance(arr[i], ak.Array) or hasattr(arr[i], '__len__'):
                                lengths.append(len(arr[i]))
                        except (TypeError, IndexError):
                            # If we can't get a length, this array might be scalar at this position
                            lengths.append(1)
                    
                    # If there are differing lengths at this index, raise an error
                    if len(set(lengths)) > 1:  # Use set to find unique values
                        raise ShapeError(
                            f"Inconsistent inner dimensions in jagged Awkward arrays at index {i}",
                            shapes=[f"Found lengths: {lengths}"]
                        )
    else:
        array_shapes = [arr.shape if hasattr(arr, 'shape') and arr is not None else ()
                       for arr in arrays]
        if not all(s == array_shapes[0] for s in array_shapes):
            raise ShapeError(
                "Inconsistent array shapes in NumPy arrays",
                shapes=shape_info
            )

def compute_pt(px, py, lib):
    """Compute transverse momentum."""
    pt = backend_sqrt(px*px + py*py, lib)
    # Convert to scalar if input was scalar
    if isinstance(px, (float, int)) and isinstance(py, (float, int)):
        return float(pt)
    return pt

def compute_p(px, py, pz, lib):
    """Compute total momentum."""
    p = backend_sqrt(px*px + py*py + pz*pz, lib)
    if isinstance(px, (float, int)) and isinstance(py, (float, int)) and isinstance(pz, (float, int)):
        return float(p)
    return p


def compute_mass(E, p, lib):
    """Compute mass from energy and momentum."""
    m2 = E*E - p*p
    m = backend_sqrt(m2 * (m2 > 0), lib)
    if isinstance(E, (float, int)) and isinstance(p, (float, int)):
        return float(m)
    return m


def compute_eta(p, pz, lib):
    """Compute pseudorapidity."""
    pt = compute_pt(p, pz, lib)
    # Pseudorapidity is defined as η = -ln(tan(θ/2)) where θ is the polar angle
    # The polar angle θ is the angle between the momentum vector and the z-axis
    # θ = arctan(pt/pz), so we need to compute arctan(pt/pz) first
    # For numerical stability, use the formula η = 0.5 * ln((p + pz)/(p - pz))
    # This avoids issues when pz approaches ±p
    
    # Handle the case where pz = ±p (θ = 0 or π)
    # When pz = p, η = ∞
    # When pz = -p, η = -∞
    # Use a numerically stable formula
    eta = 0.5 * backend_log((p + pz)/(p - pz + 1e-10), lib)
    if isinstance(p, (float, int)) and isinstance(pz, (float, int)):
        return float(eta)
    return eta

def compute_phi(px, py, lib):
    """Compute azimuthal angle."""
    phi = backend_atan2(py, px, lib)
    if isinstance(px, (float, int)) and isinstance(py, (float, int)):
        return float(phi)
    return phi

def compute_p4_from_ptepm(pt, eta, phi, m, lib):
    """Convert pt, eta, phi, mass to px, py, pz, E."""
    px = pt * backend_cos(phi, lib)
    py = pt * backend_sin(phi, lib)
    pz = pt * backend_sinh(eta, lib)
    E = backend_sqrt(pt*pt * backend_cosh(eta, lib)**2 + m*m, lib)
    
    # Handle scalar inputs
    if all(isinstance(x, (float, int)) for x in [pt, eta, phi, m]):
        return float(px), float(py), float(pz), float(E)
    return px, py, pz, E