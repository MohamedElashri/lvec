# utils.py
import math
import numpy as np
from .backends import (is_ak, is_np, to_ak, to_np, backend_sqrt,
                      backend_sin, backend_cos, backend_atan2,
                      backend_sinh, backend_cosh, backend_log,
                      backend_where, backend_any)
from .exceptions import ShapeError, BackendError
from ._exact import accurate_result

_CANCELLATION_RTOL = math.sqrt(np.finfo(np.float64).eps)
# Signed-root error grows as eps * E**2 / abs(mass). Recover before the
# ordinary squared-sum error is amplified beyond a component-scaled root
# accuracy target. Mass squared keeps its own, less restrictive trigger.
_MASS_CANCELLATION_RTOL = 1. / 4096.


def small_invariant(px,py,pz,energy,*,squared=False):
    """Avoid array-kernel setup for scalar and very short owned float64 data."""
    from ._exact import _product_sum,_rounded_float
    tiny=np.finfo(float).tiny
    rtol = _CANCELLATION_RTOL if squared else _MASS_CANCELLATION_RTOL
    def evaluate(x,y,z,e):
        value=e*e-(x*x+y*y+z*z)
        bound=(rtol*abs(e))*abs(e)
        if (not math.isfinite(value) or abs(value)<=bound or abs(value)<tiny) and all(math.isfinite(c) for c in (x,y,z,e)):
            integer,power=_product_sum((x,y,z,e))
            return _rounded_float(integer,power,root=not squared)
        if squared:return value
        return -math.sqrt(-value) if value<0 else math.sqrt(value)
    if not isinstance(energy,np.ndarray):return evaluate(px,py,pz,energy)
    result=np.empty(energy.shape,dtype=np.float64)
    for index in range(energy.size):
        result.flat[index]=evaluate(*(float(c.flat[index]) for c in (px,py,pz,energy)))
    return result


def gamma_result(px, py, pz, energy, lib):
    """Use the recovered signed mass without retaining another cached property."""
    if lib == 'np' and not isinstance(energy, np.ndarray):
        if energy <= 0 or not math.isfinite(energy):
            return math.nan
        mass = small_invariant(px, py, pz, energy)
        if mass < 0 or math.isnan(mass):
            return math.nan
        return energy / mass if mass else math.inf

    def kernel(x, y, z, e):
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            if e.size <= 8:
                mass = small_invariant(x, y, z, e)
            else:
                squared = e*e - (x*x + y*y + z*z)
                mass = invariant_result(squared, e, x, y, z, 'np')
            valid = (e > 0) & np.isfinite(e) & (mass >= 0)
            np.divide(e, mass, out=mass, where=valid)
            mass[~valid] = np.nan
            return (mass,)

    kernel.outputs = 1
    if lib == 'ak':
        from ._transform import dispatch_owned
        return dispatch_owned(kernel, (px, py, pz, energy))
    return kernel(px, py, pz, energy)[0]

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

            def numeric_type(array_type):
                # Inspect type metadata, not particles. Strings are represented
                # by lists of uint8, so their semantic parameters matter too.
                if array_type.parameters.get('__array__') in {
                        'string', 'bytestring', 'char', 'byte'}:
                    return False
                if isinstance(array_type, ak.types.UnknownType):
                    return True  # Empty/all-missing arrays carry no values.
                if isinstance(array_type, ak.types.NumpyType):
                    return np.dtype(array_type.primitive).kind in 'biuf'
                if isinstance(array_type, (ak.types.ListType, ak.types.RegularType,
                                           ak.types.OptionType)):
                    return numeric_type(array_type.content)
                if isinstance(array_type, ak.types.UnionType):
                    return all(numeric_type(t) for t in array_type.contents)
                return False  # Records and other nonnumeric semantic types.

            if any(not numeric_type(ak.type(value).content)
                   for value, scalar in zip(values, scalars) if not scalar):
                raise TypeError("Vector components must contain real numeric values")
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
    

def _plain_structure_equal(layouts):
    """Compare visible plain layout shapes; None requests general validation.

    Only lists, regular dimensions, and NumPy leaves are handled. Trim unused
    content at each level so sliced offsets may differ without changing shape.
    Never inspect particle values or compare hidden content beyond list bounds.
    """
    import awkward as ak

    while True:
        # Semantic arrays (for example strings) can hide a physical list axis.
        if any(layout.parameter('__array__') is not None for layout in layouts):
            return None
        if all(layout.is_numpy for layout in layouts):
            shape = layouts[0].data.shape
            return all(layout.data.shape == shape for layout in layouts[1:])
        if all(isinstance(layout, ak.contents.ListOffsetArray) for layout in layouts):
            offsets = layouts[0].offsets.data
            lengths = None
            children = []
            for layout in layouts:
                other = layout.offsets.data
                if other is not offsets and not np.array_equal(offsets, other):
                    if lengths is None:
                        lengths = np.diff(offsets)
                    if not np.array_equal(lengths, np.diff(other)):
                        return False
                start, stop = int(other[0]), int(other[-1])
                child = layout.content
                children.append(child if start == 0 and stop == len(child) else child[start:stop])
            layouts = children
        elif all(isinstance(layout, ak.contents.RegularArray) for layout in layouts):
            size, length = layouts[0].size, len(layouts[0])
            if any(layout.size != size or len(layout) != length for layout in layouts[1:]):
                return False
            layouts = [layout.content[:length * size] for layout in layouts]
        else:
            return None


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

    plain = _plain_structure_equal([ak.to_layout(arr) for arr in arrays])
    if plain is False:
        raise ShapeError("Inconsistent inner dimensions in Awkward arrays", shapes=shape_info)
    if plain is True:
        return

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


def scaled_invariant(E, px, py, pz, lib):
    """Return a component scale and dimensionless Minkowski norm squared."""
    scale = np.maximum(np.maximum(abs(E), abs(px)), np.maximum(abs(py), abs(pz)))
    denominator = backend_where(scale == 0, 1., scale, lib)
    e, x, y, z = (value / denominator for value in (E, px, py, pz))
    return scale, e*e - x*x - y*y - z*z


def _null_axis(E, px, py, pz):
    """Prove an entire batch is exactly null along one coordinate axis."""
    if backend_any(~np.isfinite(E)):
        return None
    momentum = (px, py, pz)
    for axis in range(3):
        others = [momentum[i] for i in range(3) if i != axis]
        if (not backend_any(others[0] != 0) and not backend_any(others[1] != 0)
                and not backend_any(abs(E) != abs(momentum[axis]))):
            return axis
    return None


def _bounds(value):
    if is_ak(value):
        import awkward as ak
        return (ak.min(value, axis=None, mask_identity=False),
                ak.max(value, axis=None, mask_identity=False))
    if isinstance(value, np.ndarray):
        return value.min(initial=np.inf), value.max(initial=-np.inf)
    return value, value


def dot_result(result, left, right, lib):
    """Correct overflow and cancellation without discarding small products."""
    if np.isscalar(result):
        left_scale, right_scale = max(map(abs, left)), max(map(abs, right))
    else:
        left_scale = np.maximum(np.maximum(abs(left[0]), abs(left[1])),
                                np.maximum(abs(left[2]), abs(left[3])))
        right_scale = np.maximum(np.maximum(abs(right[0]), abs(right[1])),
                                 np.maximum(abs(right[2]), abs(right[3])))
    bound = (_CANCELLATION_RTOL*left_scale)*right_scale
    exceptional = ~np.isfinite(result) | (abs(result) <= bound)
    if not backend_any(exceptional):
        return result
    axis = _null_axis(left[3], *left[:3])
    if axis is not None and _null_axis(right[3], *right[:3]) == axis:
        if not backend_any(np.sign(left[3])*np.sign(right[3])
                           != np.sign(left[axis])*np.sign(right[axis])):
            return np.zeros_like(result)
    if isinstance(result, np.ndarray):
        values = np.broadcast_arrays(*(left + right))
        left, right = values[:4], values[4:]
        left = tuple(value[exceptional] for value in left)
        right = tuple(value[exceptional] for value in right)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        finite = True
        for values in (left, right):
            for value in values:
                finite = finite & np.isfinite(value)
        mask = None if isinstance(result, np.ndarray) else exceptional & finite
        fallback = accurate_result((*left, *right), lib, dot=True, where=mask)
        if isinstance(result, np.ndarray):
            result[exceptional] = np.where(finite, fallback, result[exceptional])
            return result
        return backend_where(exceptional & finite, fallback, result, lib)


def euclidean_result(squared, *components):
    """Keep NumPy's fast square/sum path, using hypot only outside its range."""
    tiny = np.finfo(np.float64).tiny
    lower, upper = squared.min(initial=np.inf), squared.max(initial=-np.inf)
    regular = lower >= tiny and np.isfinite(upper)
    exceptional = None if regular else (~np.isfinite(squared) | (squared < tiny))
    # An opted-in cached intermediate is read-only and must remain squared.
    result = np.sqrt(squared, out=squared if squared.flags.writeable else None)
    if exceptional is not None and np.any(exceptional):
        values = [component[exceptional] for component in components]
        norm = values[0]
        for component in values[1:]:
            norm = np.hypot(norm, component)
        result[exceptional] = abs(norm)
    return result


def invariant_result(m2, E, px, py, pz, lib, *, squared=False):
    """Correct ill-conditioned/range-extreme norms after the vectorized path.

    ``m2`` is a fresh temporary, so NumPy results may reuse its storage. Squared
    quantities outside float64's range remain signed infinity/zero; their square
    roots can still be finite and are calculated without squaring the scale.
    """
    tiny = np.finfo(np.float64).tiny
    rtol = _CANCELLATION_RTOL if squared else _MASS_CANCELLATION_RTOL
    lower, upper = _bounds(m2)
    e_min, e_max = _bounds(E)
    largest_energy = max(abs(float(e_min)), abs(float(e_max)))
    bound = (rtol*largest_energy)*largest_energy
    if ((lower == 0 and upper == 0) or not np.isfinite(lower) or not np.isfinite(upper)):
        if _null_axis(E, px, py, pz) is not None:
            return np.zeros_like(m2)
    # Common timelike/spacelike batches need no masks or scaled temporaries.
    if ((lower >= tiny and lower > bound and np.isfinite(upper))
            or (upper <= -tiny and -upper > bound and np.isfinite(lower))):
        if squared:
            return m2
        if isinstance(m2, np.ndarray):
            if lower < 0:
                np.negative(m2, out=m2)
            np.sqrt(m2, out=m2)
            if lower < 0:
                np.negative(m2, out=m2)
            return m2
        return np.sqrt(m2) if lower > 0 else -np.sqrt(-m2)

    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        local_bound = (rtol*abs(E))*abs(E)
        exceptional = (~np.isfinite(m2) | (abs(m2) <= local_bound)
                       | ((m2 > -tiny) & (m2 < tiny)))
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        if squared:
            result = m2
        elif isinstance(m2, np.ndarray):
            negative = np.signbit(m2)
            np.abs(m2, out=m2)
            np.sqrt(m2, out=m2)
            np.negative(m2, out=m2, where=negative)
            result = m2
        else:
            result = np.sign(m2) * np.sqrt(abs(m2))
        if not backend_any(exceptional):
            return result

        values = (px, py, pz, E)
        if isinstance(m2, np.ndarray):
            values = tuple(value[exceptional] for value in values)
        finite = np.isfinite(values[0])
        for value in values[1:]:
            finite = finite & np.isfinite(value)
        mask = None if isinstance(m2, np.ndarray) else exceptional & finite
        fallback = accurate_result(values, lib, root=not squared, where=mask)
        if isinstance(m2, np.ndarray):
            result[exceptional] = np.where(finite, fallback, result[exceptional])
            return result
        return backend_where(exceptional & finite, fallback, result, lib)


def awkward_invariant(px, py, pz, energy, *, squared=False):
    """Traverse once, applying the existing invariant/recovery kernel at leaves.

    Components have already been normalized and checked by LVec. Awkward aligns
    indexed/optional contents and preserves their visible list/missing structure.
    The NumPy expression retains the original order of the squared sums.
    """
    import awkward as ak

    if ak.backend(energy) != 'cpu':
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            m2 = np.square(energy) - (np.square(px) + np.square(py) + np.square(pz))
            return invariant_result(m2, energy, px, py, pz, 'ak', squared=squared)

    def transform(layouts, **kwargs):
        if all(layout.is_numpy for layout in layouts):
            x, y, z, e = (layout.data for layout in layouts)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                m2 = np.square(e) - (np.square(x) + np.square(y) + np.square(z))
                result = invariant_result(m2, e, x, y, z, 'np', squared=squared)
            return ak.contents.NumpyArray(result)
        if all(layout.is_unknown for layout in layouts):
            return layouts[0]

    return ak.transform(transform, px, py, pz, energy, allow_records=False)


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
    pt = np.hypot(px, py)
    
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
    p = np.hypot(np.hypot(px, py), pz)
    
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
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        pz = pt * backend_sinh(eta, lib)
        exceptional = ~np.isfinite(pz) & np.isfinite(pt) & np.isfinite(eta)
        if backend_any(exceptional):
            # sinh(eta) itself may overflow even when pt*sinh(eta) is finite.
            magnitude = np.exp(np.log(abs(pt)) + abs(eta) - np.log(2.)
                               + np.log1p(-np.exp(-2*abs(eta))))
            fallback = np.sign(pt) * np.sign(eta) * magnitude
            fallback = backend_where(pt == 0, 0., fallback, lib)
            pz = backend_where(exceptional, fallback, pz, lib)
        E = np.hypot(np.hypot(pt, pz), m)
    
    # Convert to scalar only if all inputs were scalars
    if all(isinstance(x, (float, int)) for x in [pt, eta, phi, m]):
        return float(px), float(py), float(pz), float(E)
    return px, py, pz, E
