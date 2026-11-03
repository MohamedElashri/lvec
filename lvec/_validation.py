"""Particle-domain comparisons with exact recovery at tolerance boundaries."""
import math

import numpy as np


def _exact_allowed(row, relative, absolute):
    # All squares share a dyadic grid. Compare the full tolerance inequality
    # as integers, including tolerances whose rescaling would underflow.
    terms = []
    for value in row:
        numerator, denominator = float(value).as_integer_ratio()
        terms.append((numerator*numerator, -2*(denominator.bit_length()-1)))
    power = min(exponent for number, exponent in terms if number) if any(n for n, _ in terms) else 0
    squares = [number << (exponent-power) if number else 0 for number, exponent in terms]
    spatial, energy = sum(squares[:3]), squares[3]
    norm = energy-spatial
    if norm >= 0:
        return True
    rn, rd = relative
    an, ad = absolute
    margin = (norm*rd + rn*max(energy, spatial))*ad
    constant = an*rd
    return margin + (constant << -power) >= 0 if power < 0 else (margin << power) + constant >= 0


def physical_mask(px, py, pz, energy, rtol, atol):
    """Evaluate the particle-domain equation without rounding its boundary."""
    rtol, atol = float(rtol), float(atol)
    relative = rtol.as_integer_ratio()
    absolute = atol.as_integer_ratio()
    epsilon = 32*np.finfo(float).eps

    def scalar(row):
        if row[3] < 0 or not all(math.isfinite(c) for c in row):
            return False
        scale = max(map(abs, row))
        if scale == 0:
            return True
        x, y, z, e = (c/scale for c in row)
        spatial, temporal = x*x+y*y+z*z, e*e
        tolerance = (atol/scale)/scale + rtol*max(spatial, temporal)
        margin = temporal-spatial+tolerance
        uncertainty = epsilon*(temporal+spatial+abs(tolerance))
        if math.isfinite(margin) and abs(margin) <= uncertainty:
            return _exact_allowed(row, relative, absolute)
        return margin >= 0

    if all(np.isscalar(c) for c in (px, py, pz, energy)):
        return np.bool_(scalar((px, py, pz, energy)))

    def kernel(x, y, z, e):
        if e.size <= 8:
            result = np.empty(e.shape, dtype=bool)
            for i in range(e.size):
                result.flat[i] = scalar(tuple(float(c.flat[i]) for c in (x, y, z, e)))
            return (result,)
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(e)
            visible = finite & (e >= 0)
            scale = np.maximum(np.maximum(abs(x), abs(y)), np.maximum(abs(z), abs(e)))
            denominator = np.where(scale == 0, 1., scale)
            a, b, c, t = (value/denominator for value in (x, y, z, e))
            spatial, temporal = a*a+b*b+c*c, t*t
            tolerance = (atol/denominator)/denominator + rtol*np.maximum(spatial, temporal)
            margin = temporal-spatial+tolerance
            result = visible & (margin >= 0)
            risky = visible & np.isfinite(margin) & (abs(margin) <= epsilon*(temporal+spatial+abs(tolerance)))
            for index in np.flatnonzero(risky):
                row = tuple(float(value.flat[index]) for value in (x, y, z, e))
                result.flat[index] = _exact_allowed(row, relative, absolute)
            return (result,)

    from .backends import is_ak
    kernel.outputs = 1
    if any(is_ak(c) for c in (px, py, pz, energy)):
        from ._transform import dispatch_owned
        return dispatch_owned(kernel, (px, py, pz, energy))
    return kernel(*(np.asarray(c) for c in (px, py, pz, energy)))[0]
