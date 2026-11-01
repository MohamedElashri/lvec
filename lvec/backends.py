# backends.py
import numpy as np
import math
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
        # Restore only existing optional LIST axes. Masking every axis also
        # wraps ordinary arrays in unnecessary option buffers and makes large
        # eta/rapidity calculations retain much more memory than their inputs.
        if is_ak(condition):
            axes = set()
            pending = [(ak.type(condition).content, 0)]
            while pending:
                array_type, axis = pending.pop()
                if isinstance(array_type, ak.types.OptionType):
                    if axis < condition.ndim - 1:
                        axes.add(axis)
                    pending.append((array_type.content, axis))
                elif isinstance(array_type, (ak.types.ListType, ak.types.RegularType)):
                    pending.append((array_type.content, axis + 1))
                elif isinstance(array_type, ak.types.UnionType):
                    pending.extend((content, axis) for content in array_type.contents)
            for axis in sorted(axes):
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


def beta_delta(bx, by, bz):
    """Recover 1-beta² only near the light-speed boundary."""
    b2 = bx*bx + by*by + bz*bz
    delta = 1.0-b2
    risk = delta <= np.sqrt(np.finfo(float).eps)
    if backend_any(risk):
        from ._exact import accurate_result
        lib = 'ak' if any(is_ak(x) for x in (bx, by, bz)) else 'np'
        if lib == 'ak':
            exact = accurate_result((bx, by, bz, 1.0), lib, where=risk)
            delta = backend_where(risk, exact, delta, lib)
        else:
            arrays = np.broadcast_arrays(bx, by, bz, 1.0)
            delta = np.array(delta, copy=True)
            delta[risk] = accurate_result(tuple(x[risk] for x in arrays), lib)
    return b2, delta


def validate_beta(bx, by, bz, *, return_delta=False):
    """Return beta² after checking finite components and |beta| < 1."""
    if all(isinstance(b,(float,int)) for b in (bx,by,bz)):
        if not all(math.isfinite(b) for b in (bx,by,bz)):
            raise InputError('boost velocity','non-finite','Components must be finite')
        if any(abs(b)>=1 for b in (bx,by,bz)):
            raise InputError('boost velocity','magnitude','Must be < 1 (speed of light)')
        b2=bx*bx+by*by+bz*bz
        delta=1.-b2
        if delta>math.sqrt(np.finfo(float).eps):
            return (b2,delta) if return_delta else b2
        # Near the light cone, retain exact recovery from the original inputs.
    if backend_any(~np.isfinite(bx) | ~np.isfinite(by) | ~np.isfinite(bz)):
        raise InputError("boost velocity", "non-finite", "Components must be finite")
    # Reject obviously invalid components before squaring to avoid overflow.
    invalid_component = (abs(bx) >= 1) | (abs(by) >= 1) | (abs(bz) >= 1)
    scalar = all(np.isscalar(x) for x in (bx, by, bz))
    expected = "Must be < 1 (speed of light)" if scalar else "All values must be < 1 (speed of light)"
    if backend_any(invalid_component):
        raise InputError("boost velocity", "magnitude", expected)
    b2, delta = beta_delta(bx, by, bz)
    if backend_any(delta <= 0):
        raise InputError("boost velocity", "magnitude", expected)
    return (b2, delta) if return_delta else b2


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


def own_plain_components(*components):
    """Snapshot validated plain Awkward columns with one shared list skeleton.

    Only parameter-free CPU NumPy/ListOffset/Regular layouts are eligible.
    Components must already have matching visible shapes. Options, indexing,
    unions, custom behavior and semantic metadata keep the general copy path.
    """
    if not all(is_ak(c) for c in components):return None
    if any(ak.backend(c)!='cpu' or c.behavior is not None or c.attrs for c in components):return None
    layouts=[ak.to_layout(c) for c in components]
    pending=layouts
    while True:
        if any(c.parameters for c in pending):return None
        if all(c.is_numpy for c in pending):break
        if all(isinstance(c,ak.contents.ListOffsetArray) for c in pending):
            if any(c.offsets.dtype!=pending[0].offsets.dtype for c in pending):return None
            pending=[c.content for c in pending]
        elif all(isinstance(c,ak.contents.RegularArray) for c in pending):
            pending=[c.content for c in pending]
        else:return None
    def copy_columns(nodes):
        if nodes[0].is_numpy:
            return [ak.contents.NumpyArray(np.array(c.data,dtype=np.float64,copy=True)) for c in nodes]
        if isinstance(nodes[0],ak.contents.ListOffsetArray):
            index=np.array(nodes[0].offsets.data,copy=True)
            index-=index[0]
            shared=type(nodes[0].offsets)(index)
            children=[]
            for c in nodes:
                start,stop=int(c.offsets[0]),int(c.offsets[-1])
                children.append(c.content if start==0 and stop==len(c.content) else c.content[start:stop])
            return [ak.contents.ListOffsetArray(shared,child) for child in copy_columns(children)]
        size,length=nodes[0].size,len(nodes[0])
        children=copy_columns([c.content[:size*length] for c in nodes])
        return [ak.contents.RegularArray(child,size,zeros_length=length) for child in children]
    return tuple(freeze_array(ak.Array(layout)) for layout in copy_columns(layouts))


def own_computed_structure(value):
    """Own computed Awkward metadata, retaining proven-owned numeric leaves.

    Internal producers may reuse immutable input leaves or fresh arithmetic
    leaves. Offsets/indices/masks can still originate in external parameters;
    snapshot those buffers before freezing the result. Never use for public
    constructor inputs or arbitrary external views.
    """
    form,length,buffers=ak.to_buffers(value,buffer_key='{form_key}-{attribute}')
    owned={key:buffer if key.endswith('-data') else buffer.copy()
           for key,buffer in buffers.items()}
    return freeze_array(ak.from_buffers(form,length,owned))


def normalize_scalar(value):
    """Normalize real NumPy scalar types without discarding imaginary parts."""
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError("Vector components must be real-valued")
    return float(value) if isinstance(value, np.number) else value
