# backends.py
import numpy as np
from .exceptions import DependencyError, InputError
try:
    import awkward as ak
    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False

def is_ak(x):
    """Check if input is an Awkward array."""
    if not HAS_AWKWARD:
        return False
    return isinstance(x, ak.Array)

def is_np(x):
    """Check if input is a NumPy array."""
    return isinstance(x, np.ndarray)

def to_ak(x):
    """Convert input to Awkward array."""
    if not HAS_AWKWARD:
        raise DependencyError("awkward", "2.7.4")
    if is_ak(x):
        return x
    # Handle scalar inputs by making them single-element arrays first
    if isinstance(x, (float, int)):
        return ak.Array([x])
    return ak.Array(x)

def to_np(x):
    """Convert input to NumPy array."""
    if is_ak(x):
        return ak.to_numpy(x)
    if isinstance(x, (float, int)):
        return np.array([x])
    return np.asarray(x)

def backend_sqrt(x, lib):
    """
    Compute square root using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.sqrt(x)
    if lib == 'ak' and HAS_AWKWARD:
        # Use NumPy's sqrt directly - it works on Awkward arrays
        return np.sqrt(x)
    return np.sqrt(x)


def backend_where(condition, x, y, lib):
    """
    Compute where using appropriate backend.
    
    For Awkward arrays, uses ak.where which is specifically designed for
    Awkward arrays.
    """
    if lib == 'ak' and HAS_AWKWARD:
        import awkward as ak
        result = ak.where(condition, x, y)
        # ak.where may turn a missing list into a list containing None.
        # Restore option masks at every depth of an Awkward condition.
        if is_ak(condition):
            for axis in range(condition.ndim):
                result = ak.mask(result, ~ak.is_none(condition, axis=axis))
        return result
    else:
        return np.where(condition, x, y)
        
def backend_sin(x, lib):
    """
    Compute sine using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.sin(x)
    return np.sin(x)

def backend_cos(x, lib):
    """
    Compute cosine using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.cos(x)
    return np.cos(x) 


def backend_sinh(x, lib):
    """
    Compute hyperbolic sine using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.sinh(x)
    return np.sinh(x)

def backend_cosh(x, lib):
    """
    Compute hyperbolic cosine using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.cosh(x)
    return np.cosh(x)
    
def backend_atan2(y, x, lib):
    """
    Compute arctangent2 using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)) and isinstance(y, (float, int)):
        return np.arctan2(y, x)
    return np.arctan2(y, x)

def backend_log(x, lib):
    """
    Compute natural logarithm using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.log(x)
    return np.log(x)

def backend_exp(x, lib):
    """
    Compute exponential using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.exp(x)
    return np.exp(x)

def backend_tan(x, lib):
    """
    Compute tangent using appropriate backend.
    
    For Awkward arrays, applies NumPy functions that work properly with
    Awkward arrays without converting to/from NumPy.
    """
    if isinstance(x, (float, int)):
        return np.tan(x)
    return np.tan(x)

def backend_any(value):
    """Reduce a boolean mask over every dimension, ignoring missing values."""
    return bool(ak.any(value, axis=None)) if is_ak(value) else bool(np.any(value))


def validate_beta(bx, by, bz):
    """Return beta² after checking finite components and |beta| < 1."""
    if backend_any(~np.isfinite(bx) | ~np.isfinite(by) | ~np.isfinite(bz)):
        raise InputError("boost velocity", "non-finite", "Components must be finite")
    # Reject obviously invalid components before squaring to avoid overflow.
    invalid_component = (abs(bx) >= 1) | (abs(by) >= 1) | (abs(bz) >= 1)
    scalar = all(np.isscalar(x) for x in (bx, by, bz))
    expected = "Must be < 1 (speed of light)" if scalar else "All values must be < 1 (speed of light)"
    if backend_any(invalid_component):
        raise InputError("boost velocity", "magnitude", expected)
    b2 = bx*bx + by*by + bz*bz
    if backend_any(b2 >= 1):
        raise InputError("boost velocity", "magnitude", expected)
    return b2


def freeze_array(value):
    """Mark NumPy-backed array buffers read-only, without copying them."""
    if is_np(value):
        value.setflags(write=False)
    elif is_ak(value):
        _, _, buffers = ak.to_buffers(value)
        for buffer in buffers.values():
            if isinstance(buffer, np.ndarray):
                buffer.setflags(write=False)
    return value


def own_components(*components):
    """Snapshot components so input aliases cannot invalidate derived caches.

    This protects ordinary public array operations. Deliberate mutation of
    private buffers or forcibly changing writeability is outside the API.
    """
    owned = []
    for value in components:
        if is_ak(value):
            result = ak.copy(ak.values_astype(value, np.float64))
        elif is_np(value):
            if np.iscomplexobj(value):
                raise TypeError("Vector components must be real-valued")
            result = np.array(value, dtype=np.float64, copy=True)
        elif isinstance(value, (int, float, np.number)):
            result = float(value)
        else:
            raise TypeError("Vector components must be real numeric scalars or arrays")
        owned.append(freeze_array(result))
    return tuple(owned)
