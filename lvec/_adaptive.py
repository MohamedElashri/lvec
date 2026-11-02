"""Risk-triggered recovery for transforms and reductions.

Normal buffers use NumPy kernels. Only entries with cancellation or range loss
enter compensated/exact arithmetic; Awkward preserves its array layouts.
"""
import math

import numpy as np

from ._exact import accurate_result, _rounded_float, _two_sum
from .backends import is_ak, normalize_scalar

_RTOL = math.sqrt(np.finfo(np.float64).eps)
_TINY = np.finfo(np.float64).tiny


def _dispatch(kernel, *values):
    if any(is_ak(x) for x in values):
        import awkward as ak

        def transform(layouts, **kwargs):
            if all(x.is_numpy for x in layouts):
                result = kernel(*(x.data for x in layouts))
                return tuple(ak.contents.NumpyArray(x) for x in result)
            if all(x.is_unknown for x in layouts):
                return tuple(layouts[0] for _ in range(kernel.outputs))

        return ak.transform(transform, *ak.broadcast_arrays(*values), allow_records=False)
    return kernel(*np.broadcast_arrays(*values))


def _linear(a, u, b, v):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = np.asarray(a*u)
        # Bound the first product before reusing its buffer for the output.
        # Signed reductions avoid allocating another full absolute-value array.
        scale = max(abs(float(np.fmax.reduce(result, axis=None, initial=0.))),
                    abs(float(np.fmin.reduce(result, axis=None, initial=0.))))
        np.add(result, b*v, out=result)
        absolute = abs(result)
        bound = _RTOL*scale
        if (np.fmin.reduce(absolute, axis=None, initial=np.inf) > bound
                and np.fmax.reduce(absolute, axis=None, initial=0.) < np.inf):
            return result
        coarse = ~np.isfinite(result) | (absolute <= bound)
        if not np.any(coarse):
            return result
        su, sv, sa, sb = (x[coarse] for x in (u, v, a, b))
        output = result[coarse]
        risk = ~np.isfinite(output) | (abs(output) <= _RTOL*abs(sa*su))
        risk &= np.isfinite(su) & np.isfinite(sv) & np.isfinite(sa) & np.isfinite(sb)
        # Zero coefficients/coordinates cannot hide a cancelled product.
        risk &= ((sa != 0) & (su != 0) & (sb != 0) & (sv != 0)) | ~np.isfinite(output)
    if np.any(risk):
        su, sv, sa, sb = (x[risk] for x in (su, sv, sa, sb))
        zero = np.zeros_like(su)
        indices = np.flatnonzero(coarse)[risk]
        result.flat[indices] = accurate_result((su, sv, zero, zero, -sa, -sb, zero, zero),
                                               'np', dot=True)
    return result


def _rotation_numpy(u, v, c, s):
    return _linear(c, u, -s, v), _linear(s, u, c, v)


_rotation_numpy.outputs = 2


def rotation_coefficients(angle):
    """Compute trigonometry at component precision, including float32 angles."""
    if is_ak(angle):
        import awkward as ak
        angle = ak.values_astype(angle, np.float64)
    else:
        if np.iscomplexobj(angle):
            raise TypeError("Rotation angles must be real")
        angle = np.asarray(normalize_scalar(angle), dtype=np.float64)
    return np.cos(angle), np.sin(angle)


def rotation_pair(u, v, c, s):
    return _dispatch(_rotation_numpy, u, v, c, s)


def _boost_decimal(row):
    """Recover one ill-conditioned boost using original binary64 inputs."""
    from decimal import Context, Decimal, localcontext

    exponents = [math.frexp(abs(x))[1] for x in row if x]
    precision = max(80, int(2*(max(exponents)-min(exponents))*.302)+80)
    with localcontext(Context(prec=precision)) as context:
        context.prec = precision
        px, py, pz, energy, bx, by, bz = map(Decimal.from_float, map(float, row))
        beta = (bx, by, bz)
        p = (px, py, pz)
        gamma = 1/(1-sum(b*b for b in beta)).sqrt()
        bp = sum(b*x for b, x in zip(beta, p))
        t = gamma*gamma/(gamma+1)*bp+gamma*energy
        return tuple(float(x+t*b) for x, b in zip(p, beta)) + (float(gamma*(energy+bp)),)


def _boost_range_risk(values, gamma):
    """A coarse buffer bound avoids per-entry tests on well-scaled inputs."""
    max_gamma = float(np.max(gamma, initial=1.))
    if max_gamma>1e4:return True
    largest, smallest = 0., math.inf
    for value in values:
        absolute = abs(value)
        upper=float(np.fmax.reduce(absolute,axis=None,initial=0.))
        largest=max(largest,upper)
        if upper==0:continue
        lower=float(np.fmin.reduce(absolute,axis=None,initial=np.inf))
        if lower==0:
            lower=float(np.fmin.reduce(absolute,axis=None,where=absolute!=0,initial=np.inf))
        smallest=min(smallest,lower)
    return (max_gamma > 1e4 or largest > np.finfo(float).max/(8*max_gamma)
            or (0 < largest < _TINY) or smallest < 1e-12*largest)


def _boost_numpy(px, py, pz, energy, bx, by, bz, gamma):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        bp = bx*px+by*py+bz*pz
        t = gamma*gamma/(gamma+1)*bp+gamma*energy
        outputs = [px+t*bx, py+t*by, pz+t*bz, gamma*(energy+bp)]
        if not _boost_range_risk((px, py, pz, energy), gamma):
            return tuple(outputs)
        broken = ~np.isfinite(outputs[3])
        cancellation = ((abs(energy+bp) <= _RTOL*abs(energy))
                        | ((abs(outputs[3]) < _TINY) & (energy != 0)))
        for original, result in zip((px, py, pz), outputs):
            broken |= ~np.isfinite(result)
            cancellation |= ((abs(result) <= _RTOL*abs(original))
                             | (abs(result) < _TINY)) & (original != 0)
        # Examine component ratios only for cancellation candidates.
        # Well-scaled buffers have already returned without these masks.
        risk = np.asarray(broken).copy()
        if np.any(cancellation):
            candidates = [x[cancellation] for x in (px, py, pz, energy)]
            largest = np.maximum.reduce([abs(x) for x in candidates])
            smallest = np.minimum.reduce([np.where(x != 0, abs(x), np.inf) for x in candidates])
            risk[cancellation] |= ((smallest < 1e-12*largest)
                                   | (gamma[cancellation] > 1e4)
                                   | ((largest > 0) & (largest < _TINY)))
    if np.any(risk):
        outputs = [np.asarray(x) for x in outputs]
        values = (px, py, pz, energy, bx, by, bz)
        risk &= np.logical_and.reduce([np.isfinite(x) for x in values])
        for index in np.flatnonzero(risk):
            corrected = _boost_decimal(tuple(x.flat[index] for x in values))
            for result, value in zip(outputs, corrected):
                result.flat[index] = value
    return tuple(outputs)


_boost_numpy.outputs = 4


def boost_components(values, beta, gamma):
    # Scalar parameters need no Awkward-sized buffers. Broadcast zero-stride
    # NumPy views inside each numerical leaf, retaining the exact same kernel
    # and per-entry precision recovery (including gamma indexing).
    if all(np.isscalar(x) for x in (*beta,gamma)):
        from ._transform import dispatch_owned
        def kernel(*components):
            return _constant_beta_numpy(components,beta,float(gamma))
        kernel.outputs=4
        return dispatch_owned(kernel,values)
    return _dispatch(_boost_numpy, *values, *beta, gamma)


def _small_beta(point,beta,gamma):
    px,py,pz,energy=point
    bx,by,bz=beta
    bp=bx*px+by*py+bz*pz
    t=gamma*gamma/(gamma+1)*bp+gamma*energy
    result=(px+t*bx,py+t*by,pz+t*bz,gamma*(energy+bp))
    largest=max(map(abs,point))
    smallest=min((abs(c) for c in point if c!=0),default=math.inf)
    hierarchy=smallest<1e-12*largest or (0<largest<_TINY) or gamma>1e4
    broken=not all(math.isfinite(c) for c in result)
    cancellation=(abs(energy+bp)<=_RTOL*abs(energy) or (abs(result[3])<_TINY and energy!=0))
    cancellation=cancellation or any((abs(out)<=_RTOL*abs(p) or abs(out)<_TINY) and p!=0
                                     for p,out in zip(point[:3],result))
    if (broken or (hierarchy and cancellation)) and all(math.isfinite(c) for c in (*point,*beta)):
        return _boost_decimal((*point,*beta))
    return result


def _constant_beta_numpy(values,beta,gamma):
    shape=values[0].shape;size=values[0].size
    outputs=tuple(np.empty(shape,dtype=np.float64) for _ in range(4))
    if size<=8:
        for i in range(size):
            result=_small_beta(tuple(float(c.flat[i]) for c in values),beta,gamma)
            for output,value in zip(outputs,result):output.flat[i]=value
        return outputs
    from ._transform import _BLOCK
    inputs=tuple(c.reshape(-1) for c in values)
    flat_outputs=tuple(c.reshape(-1) for c in outputs)
    bx,by,bz=beta
    factor=gamma*gamma/(gamma+1)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        for start in range(0,size,_BLOCK):
            stop=min(size,start+_BLOCK)
            px,py,pz,e=(c[start:stop] for c in inputs)
            target=tuple(c[start:stop] for c in flat_outputs)
            bp=bx*px
            scratch=by*py
            np.add(bp,scratch,out=bp)
            np.multiply(pz,bz,out=scratch);np.add(bp,scratch,out=bp)
            t=factor*bp
            np.multiply(e,gamma,out=scratch);np.add(t,scratch,out=t)
            for original,b,output in zip((px,py,pz),beta,target):
                np.multiply(t,b,out=output);np.add(output,original,out=output)
            np.add(e,bp,out=target[3]);np.multiply(target[3],gamma,out=target[3])
            if _boost_range_risk((px,py,pz,e),gamma):
                corrected=_boost_numpy(*np.broadcast_arrays(px,py,pz,e,bx,by,bz,gamma))
                for output,value in zip(target,corrected):output[:]=value
    return outputs


def _finite_sum(values):
    """fsum's accurate accumulator, with exact recovery from intermediate overflow."""
    try:
        return math.fsum(values)
    except OverflowError:
        terms = [float(x).as_integer_ratio() for x in values if x]
        if not terms:
            return 0.
        denominator = max(d for _, d in terms)
        numerator = sum(n*(denominator//d) for n, d in terms)
        return _rounded_float(numerator, -(denominator.bit_length()-1))


def _sum_rows(rows):
    """Compensate short groups together; use fsum for long/extreme groups."""
    if rows.shape[1] > 16:
        return np.array([_finite_sum(row) for row in rows])
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        largest = np.max(abs(rows), axis=1)
        smallest = np.min(np.where(rows != 0, abs(rows), np.inf), axis=1)
        extreme = ((largest > np.finfo(float).max/max(rows.shape[1], 1))
                   | (smallest < 1e-12*largest))
        result = np.empty(len(rows))
        ordinary = rows[~extreme]
        high = np.zeros(len(ordinary))
        low = high.copy()
        tail = high.copy()
        for column in ordinary.T:
            high, error = _two_sum(high, column)
            low, carry = _two_sum(low, error)
            tail += carry
        high, error = _two_sum(high, low)
        result[~extreme] = high+(error+tail)
    for index in np.flatnonzero(extreme):
        result[index] = _finite_sum(rows[index])
    return result


def _numpy_sum(values, axis, keepdims):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.sum(values, axis=axis, keepdims=keepdims)
        # NumPy has already validated axis semantics (including scalar axes).
        if values.ndim == 0:
            return result
        axes = tuple(range(values.ndim)) if axis is None else ((axis,) if isinstance(axis, (int, np.integer)) else axis)
        axes = tuple(int(i) % values.ndim for i in axes)
        count = math.prod(values.shape[i] for i in axes)
        if count <= 2:
            return result
        scale = np.fmax.reduce(abs(values), axis=None, initial=0.)
        risk = (abs(result) <= _RTOL*scale*count) | ~np.isfinite(result)
        if scale == 0:
            return result
    if not np.any(risk):
        return result
    remaining = tuple(i for i in range(values.ndim) if i not in axes)
    rows = np.transpose(values, remaining+axes).reshape(-1, count)
    indices = np.flatnonzero(risk)
    selected = rows[indices]
    with np.errstate(over='ignore', invalid='ignore'):
        bound = np.sum(abs(selected), axis=1)
        needs = ((abs(np.asarray(result).ravel()[indices]) <= _RTOL*bound) & (bound != 0)) | ~np.isfinite(np.asarray(result).ravel()[indices])
    finite = np.all(np.isfinite(selected), axis=1) & needs
    if np.any(finite):
        result = np.array(result, copy=True)
        result.flat[indices[finite]] = _sum_rows(selected[finite])
    return result


def _keys(columns):
    """Lexicographic coordinates avoid mixed-radix integer overflow."""
    result = np.empty(len(columns[0]), dtype=[(str(i), np.int64) for i in range(len(columns))])
    for i, column in enumerate(columns):
        result[str(i)] = column
    return result


def _awkward_sum(values, axis, keepdims):
    import awkward as ak
    with np.errstate(over='ignore', invalid='ignore'):
        result = ak.sum(values, axis=axis, keepdims=keepdims, mask_identity=False)
        bound = ak.sum(abs(values), axis=axis, keepdims=keepdims, mask_identity=False)
        if is_ak(result):
            def risk_buffer(layouts, **kwargs):
                if all(x.is_numpy for x in layouts):
                    total, absolute = (x.data for x in layouts)
                    mask = ((abs(total) <= _RTOL*absolute) & (absolute != 0)) | ~np.isfinite(total)
                    return ak.contents.NumpyArray(mask)
                if all(x.is_unknown for x in layouts):
                    return layouts[0]
            risk = ak.transform(risk_buffer, result, bound, allow_records=False)
        else:
            risk = ((abs(result) <= _RTOL*bound) & (bound != 0)) | ~np.isfinite(result)
    if not (bool(ak.any(risk, axis=None)) if is_ak(risk) else bool(risk)):
        return result
    count = ak.count(values, axis=axis, keepdims=keepdims)
    risk = risk & (count > 2)
    if not (bool(ak.any(risk, axis=None)) if is_ak(risk) else bool(risk)):
        return result
    flattened = ak.to_numpy(ak.flatten(values, axis=None))
    if axis is None or not is_ak(result):
        if np.all(np.isfinite(flattened)):
            total = _finite_sum(flattened)
            if not is_ak(result):
                return total
            return ak.full_like(result, total)
        return result
    depth = values.ndim
    axis = int(axis) % depth
    remaining = [i for i in range(depth) if i != axis]

    def coordinates(array, axes):
        # Variable lists broadcast outer indices from the left. Regular lists
        # otherwise align those indices with the innermost dimension.
        array = ak.from_regular(array, axis=None)
        return [ak.broadcast_arrays(ak.local_index(array, axis=i), array)[0] for i in axes]

    input_coordinates = coordinates(values, remaining)
    output_axes = [i if keepdims else i-(i > axis) for i in remaining]
    output_coordinates = coordinates(result, output_axes)
    input_keys = _keys([ak.to_numpy(ak.flatten(x, axis=None)) for x in input_coordinates])
    output_keys = _keys([ak.to_numpy(ak.flatten(x, axis=None)) for x in output_coordinates])
    flat_risk = ak.to_numpy(ak.flatten(risk, axis=None))
    targets = np.sort(output_keys[flat_risk])
    positions = np.searchsorted(targets, input_keys)
    matches = positions < len(targets)
    matches[matches] &= targets[positions[matches]] == input_keys[matches]
    positions, selected = positions[matches], flattened[matches]
    order = np.argsort(positions, kind='stable')
    positions, selected = positions[order], selected[order]
    groups, starts, counts = np.unique(positions, return_index=True, return_counts=True)
    corrections = np.full(len(targets), np.nan)
    # Group by multiplicity, so ordinary short events compensate in buffers.
    for length in np.unique(counts):
        selected_groups = counts == length
        rows = selected[starts[selected_groups, None]+np.arange(length)]
        finite = np.all(np.isfinite(rows), axis=1)
        corrections[groups[selected_groups][finite]] = _sum_rows(rows[finite])

    def restore(layouts, **kwargs):
        if all(x.is_numpy for x in layouts):
            keys = _keys([x.data for x in layouts[:-1]])
            positions = np.searchsorted(targets, keys)
            matched = positions < len(targets)
            matched[matched] &= targets[positions[matched]] == keys[matched]
            matched[matched] &= ~np.isnan(corrections[positions[matched]])
            data = layouts[-1].data.copy()
            data[matched] = corrections[positions[matched]]
            return ak.contents.NumpyArray(data)
    return ak.transform(restore, *output_coordinates, result, allow_records=False)


def adaptive_sum(values, axis=None, *, keepdims=False):
    if is_ak(values):
        return _awkward_sum(values, axis, keepdims)
    return _numpy_sum(np.asarray(values), axis, keepdims)
