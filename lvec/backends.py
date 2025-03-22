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
        return ak.where(condition, x, y)
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