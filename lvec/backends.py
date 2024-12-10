# backends.py
import numpy as np
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
        raise DependencyError("Awkward array support requires awkward package")
    if is_ak(x):
        return x
    return ak.Array(x)

def to_np(x):
    """Convert input to NumPy array."""
    if is_ak(x):
        return ak.to_numpy(x)
    return np.asarray(x)

def backend_sqrt(x, lib):
    """Compute square root using appropriate backend."""
    return ak.sqrt(x) if lib == 'ak' else np.sqrt(x)

def backend_sin(x, lib):
    """Compute sine using appropriate backend."""
    return ak.sin(x) if lib == 'ak' else np.sin(x)

def backend_cos(x, lib):
    """Compute cosine using appropriate backend."""
    return ak.cos(x) if lib == 'ak' else np.cos(x)

def backend_sinh(x, lib):
    """Compute hyperbolic sine using appropriate backend."""
    return ak.sinh(x) if lib == 'ak' else np.sinh(x)

def backend_cosh(x, lib):
    """Compute hyperbolic cosine using appropriate backend."""
    return ak.cosh(x) if lib == 'ak' else np.cosh(x)

def backend_atan2(y, x, lib):
    """Compute arctangent2 using appropriate backend."""
    return ak.arctan2(y, x) if lib == 'ak' else np.arctan2(y, x)

def backend_log(x, lib):
    """Compute natural logarithm using appropriate backend."""
    return ak.log(x) if lib == 'ak' else np.log(x)