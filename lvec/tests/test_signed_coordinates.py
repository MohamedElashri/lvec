"""Signed-coordinate energies checked before any rounded spatial magnitude."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

import lvec


ROWS = [(1., 1e-10, 0., -1.), (.6, .8, 0., -1.),
        (3., 0., 0., -2.), (.5, 0., 0., -1.),
        (1., 0., 0., -1.), (3., 0., 0., 4.), (0., 0., 0., -1.)]


def reference(row):
    x, y, z, m = map(Fraction, row)
    squared = x*x+y*y+z*z + (m*m if m >= 0 else -m*m)
    if squared <= 0:
        return 0.
    with localcontext() as context:
        context.prec = 800
        return float((Decimal(squared.numerator)/Decimal(squared.denominator)).sqrt())


@pytest.mark.parametrize('factory', ['obj', 'named', 'numpy', 'owned', 'native'])
@pytest.mark.parametrize('exponent', [-1000, 0, 1000])
def test_signed_energy_matches_original_components(factory, exponent):
    rows = [tuple(math.ldexp(v, exponent) for v in row) for row in ROWS]
    for row in rows:
        fields = dict(zip(('x', 'y', 'z', 'mass'), row))
        if factory == 'obj':
            value = lvec.obj(**fields)
        elif factory == 'named':
            value = lvec.LVec.from_xyztau(*row)
        elif factory == 'numpy':
            value = lvec.array({k: np.full(17, v) for k, v in fields.items()})
        else:
            ak = pytest.importorskip('awkward')
            import lvec.awkward as lawk
            columns = {k: ak.Array([[v]*17, [], None]) for k, v in fields.items()}
            value = lvec.zip(columns) if factory == 'owned' else lawk.zip(columns)
            assert ak.to_list(ak.is_none(value.E, axis=0)) == [False, False, True]
        actual = value.E
        if factory in ('owned', 'native'):
            actual = ak.to_numpy(ak.flatten(actual, axis=None))
        expected = reference(row)
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0.)
        if expected == 0:
            assert not np.any(np.signbit(actual))
