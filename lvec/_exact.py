"""Exact binary product sums for exceptional four-vector calculations.

Float64 inputs are dyadic rationals. Integer accumulation cancels large terms
before rounding, without overflowing products or normalizing away small terms.
Only selected exceptional entries enter this slower path; ordinary arrays keep
their vectorized kernels. No extended float type or optional dependency is used.
"""

import math

import numpy as np

from .backends import is_ak

_EPS = np.finfo(np.float64).eps
_SAFE_MIN = 4 * math.sqrt(np.finfo(np.float64).tiny) / _EPS
_SAFE_MAX = math.sqrt(np.finfo(np.float64).max) / 4
_SPLITTER = float((1 << 27) + 1)


def _two_product(a, b):
    """Error-free product in the bounded range selected by _accurate_numpy.

    Dekker splitting / TwoProduct, as in Ogita, Rump and Oishi (2005),
    https://doi.org/10.1137/030601818. Range extremes use integer accumulation.
    """
    split_a, split_b = _SPLITTER*a, _SPLITTER*b
    a_hi, b_hi = split_a-(split_a-a), split_b-(split_b-b)
    a_lo, b_lo = a-a_hi, b-b_hi
    product = a*b
    error = a_lo*b_lo - (((product-a_hi*b_hi)-a_lo*b_hi)-a_hi*b_lo)
    return product, error


def _two_sum(a, b):
    total = a+b
    copied_b = total-a
    return total, (a-(total-copied_b))+(b-copied_b)


def _compensated_sum(values, *, dot=False):
    left, right = values[:4], values[4:] if dot else values[:4]
    high, low = _two_product(left[3], right[3])
    tail = np.zeros_like(high)
    had_error = low != 0
    for a, b in zip(left[:3], right[:3]):
        product, error = _two_product(a, b)
        high, remainder = _two_sum(high, -product)
        low, carry = _two_sum(low, remainder)
        tail = tail+carry
        low, carry = _two_sum(low, -error)
        tail = tail+carry
        had_error = had_error | (error != 0) | (remainder != 0)
    high, remainder = _two_sum(high, low)
    return high+(remainder+tail), had_error


def _product_sum(values, *, dot=False):
    terms = []
    left = values[:4]
    right = values[4:] if dot else left
    for index, (a, b) in enumerate(zip(left, right)):
        a_num, a_den = float(a).as_integer_ratio()
        if dot:
            b_num, b_den = float(b).as_integer_ratio()
        else:
            b_num, b_den = a_num, a_den
        numerator = a_num * b_num * (1 if index == 3 else -1)
        if numerator:
            power = -(a_den.bit_length() + b_den.bit_length() - 2)
            terms.append((numerator, power))
    if not terms:
        return 0, 0
    power = min(exponent for _, exponent in terms)
    return sum(n << (exponent-power) for n, exponent in terms), power


def _rounded_float(numerator, power, *, root=False):
    if not numerator:
        return 0.
    sign = -1. if numerator < 0 else 1.
    value = abs(numerator)
    if root:
        # Round sqrt(value * 2**power) directly on the output's binary grid.
        # Squaring the halfway point makes the rounding decision exact, even
        # when the squared invariant itself overflows or underflows float64.
        grid = max((value.bit_length()-1+power)//2 - 52, -1074)
        shift = power - 2*grid
        integer = math.isqrt(value << shift if shift >= 0 else value >> -shift)
        left = value << (shift+2) if shift >= 0 else value << 2
        right = (2*integer+1)**2
        if shift < 0:
            right <<= -shift
        if left > right or (left == right and integer & 1):
            integer += 1
    else:
        # Include the subnormal output grid to avoid double rounding.
        shift = max(value.bit_length()-53, -1074-power, 0)
        integer = value >> shift
        if shift:
            remainder = value - (integer << shift)
            halfway = 1 << (shift-1)
            if remainder > halfway or (remainder == halfway and integer & 1):
                integer += 1
        grid = power + shift
    try:
        return math.copysign(math.ldexp(float(integer), grid), sign)
    except OverflowError:
        return math.copysign(math.inf, sign)


def _numpy_result(values, *, root=False, dot=False):
    arrays = np.broadcast_arrays(*values)
    result = np.empty(arrays[0].shape, dtype=np.float64)
    flat_values = zip(*(array.flat for array in arrays))
    output = result.flat
    for index, row in enumerate(flat_values):
        if all(math.isfinite(x) for x in row):
            numerator, power = _product_sum(row, dot=dot)
            output[index] = _rounded_float(numerator, power, root=root)
        else:
            # The caller preserves its original result for nonfinite inputs.
            output[index] = np.nan
    return result


def _accurate_numpy(values, *, root=False, dot=False, where=None):
    arrays = np.broadcast_arrays(*values)
    scale = np.zeros(arrays[0].shape)
    smallest = np.full(arrays[0].shape, np.inf)
    finite = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        absolute = np.abs(array)
        np.maximum(scale, absolute, out=scale)
        np.minimum(smallest, absolute, out=smallest, where=array != 0)
        finite &= np.isfinite(array)
    if where is not None:
        finite &= np.broadcast_to(where, scale.shape)
    integer_mask = finite & ((scale > _SAFE_MAX) | (smallest < _SAFE_MIN)
                             | (smallest < 1e-12*scale))
    ordinary = finite & ~integer_mask
    result = np.full(scale.shape, np.nan)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        if np.any(ordinary):
            selected = (tuple(array.reshape(-1) for array in arrays) if np.all(ordinary)
                        else tuple(array[ordinary] for array in arrays))
            total, had_error = _compensated_sum(selected, dot=dot)
            unresolved = had_error & (abs(total) <= (64*_EPS*_EPS*scale[ordinary])*scale[ordinary])
            if root:
                result[ordinary] = np.copysign(np.sqrt(abs(total)), total)
            else:
                result[ordinary] = total
            # Resolve cancellation beyond double-double precision exactly.
            if np.any(unresolved):
                indices = np.flatnonzero(ordinary)
                integer_mask.flat[indices[unresolved]] = True
        if np.any(integer_mask):
            selected = tuple(array[integer_mask] for array in arrays)
            result[integer_mask] = _numpy_result(selected, root=root, dot=dot)
    return result


def accurate_result(values, lib, *, root=False, dot=False, where=None):
    """Compute selected entries, preserving NumPy broadcasting/Awkward masks.

    Components are ordered px, py, pz, E (twice for a dot product). Awkward
    traverses layouts and dispatches aligned numerical buffers, not events.
    """
    if lib == 'ak' and any(is_ak(value) for value in values):
        import awkward as ak

        def transform(layouts, **kwargs):
            if all(layout.is_numpy for layout in layouts):
                numerical = layouts if where is None else layouts[:-1]
                mask = None if where is None else layouts[-1].data
                data = _accurate_numpy(tuple(layout.data for layout in numerical),
                                      root=root, dot=dot, where=mask)
                return ak.contents.NumpyArray(data)
            if all(layout.is_unknown for layout in layouts):
                return layouts[0]

        inputs = values if where is None else (*values, where)
        return ak.transform(transform, *ak.broadcast_arrays(*inputs), allow_records=False)
    if not any(isinstance(value, np.ndarray) for value in values):
        if not all(math.isfinite(value) for value in values):
            return np.nan
        numerator, power = _product_sum(values, dot=dot)
        return _rounded_float(numerator, power, root=root)
    return _accurate_numpy(values, root=root, dot=dot, where=where)
