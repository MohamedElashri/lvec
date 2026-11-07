"""Independent signed-root checks through both sides of cancellation recovery."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from lvec import LVec

# These tests exercise recovery, which the default fast policy skips.
pytestmark = pytest.mark.usefixtures('accurate_policy')


def reference(row):
    squares = [Fraction.from_float(float(value)) ** 2 for value in row]
    invariant = squares[3] - sum(squares[:3])
    scale = sum(squares)
    with localcontext() as context:
        context.prec = 800
        root = (Decimal(abs(invariant.numerator)) / Decimal(invariant.denominator)).sqrt()
        if invariant < 0:
            root = -root
        norm = (Decimal(scale.numerator) / Decimal(scale.denominator)).sqrt()
        budget = 64 * Decimal(2) ** -52 * norm + Decimal(2) ** -1075
    return root, budget


def rows():
    result = [(float.fromhex(x), 0., 0., 1.) for x in
              ('0x1.ffffffbffffffp-1', '0x1.0000002000001p+0')]
    directions = ((1., 0., 0.), (.6, .8, 0.),
                  (.2, .3, math.sqrt(.87)))
    for exponent in (-1000, -500, 0, 500, 1000):
        for sign in (-1, 1):
            for power in (6, 10, 11, 12, 13, 20, 25, 26, 27, 40, 52):
                radius = math.sqrt(1 - sign * math.ldexp(1., -power))
                for neighbor in (math.nextafter(radius, -math.inf), radius,
                                 math.nextafter(radius, math.inf)):
                    for direction in directions:
                        result.append(tuple(math.ldexp(neighbor * value, exponent)
                                            for value in direction)
                                      + (math.ldexp(1., exponent),))
    return result


@pytest.mark.parametrize('layout', ['scalar', 'numpy-small', 'numpy-bulk',
                                   'awkward-owned', 'awkward-native'])
def test_signed_mass_meets_component_scaled_accuracy_across_boundaries(layout):
    data = np.asarray(rows())
    if layout == 'scalar':
        actual = np.array([LVec(*row).mass for row in data])
    elif layout == 'numpy-small':
        actual = np.array([LVec(*(np.full(2, value) for value in row)).mass[0]
                           for row in data])
    elif layout == 'numpy-bulk':
        actual = LVec(*data.T).mass
    else:
        ak = pytest.importorskip('awkward')
        from lvec.awkward import zip as native_zip
        components = [ak.Array([event for value in column
                               for event in ([float(value), None], [], None)])
                      for column in data.T]
        obj = (LVec(*components) if layout == 'awkward-owned' else
               native_zip(dict(zip(('px', 'py', 'pz', 'E'), components))))
        output = obj.mass
        assert ak.to_list(ak.num(output, axis=-1)) == [2, 0, None] * len(data)
        actual = ak.to_numpy(ak.flatten(output, axis=None))
    for row, value in zip(data, actual):
        expected, budget = reference(row)
        assert math.isfinite(float(value)), row
        assert abs(Decimal.from_float(float(value)) - expected) <= budget, (row, value, expected)


@pytest.mark.parametrize('x', ['0x1.ffffffbffffffp-1', '0x1.0000002000001p+0'])
def test_reported_boundary_mass_matches_independent_rounded_root(x):
    row = (float.fromhex(x), 0., 0., 1.)
    expected, _ = reference(row)
    assert LVec(*row).mass == float(expected)
    actual = LVec(*(np.full(17, value) for value in row)).mass
    assert np.all(np.abs(actual - float(expected)) <= math.ulp(float(expected)))


def test_near_null_gamma_uses_the_corrected_mass():
    row = (float.fromhex('0x1.ffffffbffffffp-1'), 0., 0., 1.)
    expected, _ = reference(row)
    assert LVec(*row).gamma == 1. / float(expected)
