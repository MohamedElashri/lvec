"""Particle-domain comparisons against exact rational supplied inputs."""
from fractions import Fraction
import math

import numpy as np
import pytest

from lvec import LVec
from lvec.exceptions import InputError


ROWS = [(.58, .8146164741766521, 0., 1.), (.6, .8, 0., 1.),
        (1., 1e-10, 0., 1.), (.28, .96, 0., 1.),
        (0., 0., 0., 2.), (1., 0., 0., 1.), (0., 0., 0., 0.),
        (2., 0., 0., 1.), (0., 0., 0., -1.)]


def reference(row, rtol, atol):
    if not all(math.isfinite(x) for x in row) or row[3] < 0:
        return False
    x, y, z, e = map(Fraction, row)
    spatial = x*x + y*y + z*z
    return e*e-spatial >= -Fraction(atol)-Fraction(rtol)*max(e*e, spatial)


def make(layout, rows):
    if layout == 'scalar':
        return LVec(*rows[0])
    if layout == 'numpy':
        return LVec(*np.array(rows).T)
    ak = pytest.importorskip('awkward')
    import lvec.awkward as lawk
    fields = {key: ak.Array([[row[i] for row in rows], [], None])
              for i, key in enumerate(('px', 'py', 'pz', 'E'))}
    return LVec(**fields) if layout == 'owned' else lawk.zip(fields)


def flat(value):
    try:
        import awkward as ak
        if isinstance(value, ak.Array):
            return ak.to_numpy(ak.flatten(value, axis=None))
    except ImportError:
        pass
    return np.asarray(value).reshape(-1)


@pytest.mark.parametrize('layout', ['scalar', 'numpy', 'owned', 'native'])
@pytest.mark.parametrize('exponent', [-1000, 0, 1000])
def test_validation_matches_equation_and_is_monotone(layout, exponent):
    rows = [tuple(math.ldexp(x, exponent) for x in row) for row in ROWS]
    batches = [[row] for row in rows] if layout == 'scalar' else [rows*3]
    for batch in batches:
        value = make(layout, batch)
        previous = np.zeros(len(batch), dtype=bool)
        for rtol in (0., 1e-18, 1e-16, 1e-12, .5, 1.):
            result = flat(value.is_physical(rtol=rtol))
            expected = [reference(row, rtol, 0.) for row in batch]
            np.testing.assert_array_equal(result, expected)
            assert np.all(~previous | result), 'Increasing tolerance rejected an accepted vector'
            previous = result
            if all(expected):
                value.validate_particle(rtol=rtol)
            else:
                with pytest.raises(InputError):
                    value.validate_particle(rtol=rtol)


@pytest.mark.parametrize('layout', ['scalar', 'numpy', 'owned', 'native'])
def test_absolute_tolerance_and_range_extremes(layout):
    rows = [(1e200, 1., 0., 1e200), (1e200, 1e-200, 0., 1e200),
            (1e-200, 1e-210, 0., 1e-200), (2., 0., 0., 1.)]
    for row in rows:
        batch = [row] if layout == 'scalar' else [row]*17
        value = make(layout, batch)
        for rtol in (0., np.nextafter(0., 1.), 1e-16, 1e-12):
            previous = False
            for atol in (0., np.nextafter(0., 1.), 1e-300, 1e-20, 1., 4., 1e300):
                actual = flat(value.is_physical(rtol=rtol, atol=atol))
                expected = reference(row, rtol, atol)
                assert np.all(actual == expected)
                assert not previous or expected
                previous = expected


@pytest.mark.parametrize('layout', ['scalar', 'numpy', 'owned', 'native'])
def test_nonfinite_components_rejected_without_warning(layout):
    for row in [(math.inf, 0., 0., 1.), (0., math.nan, 0., 1.), (0., 0., 0., math.inf)]:
        value = make(layout, [row] if layout == 'scalar' else [row]*17)
        with np.errstate(all='raise'):
            assert not np.any(flat(value.is_physical()))


@pytest.mark.parametrize('layout', ['owned', 'native'])
def test_validation_keeps_missing_masks(layout):
    ak = pytest.importorskip('awkward')
    value = make(layout, ROWS)
    mask = value.is_physical(rtol=1e-16)
    assert ak.to_list(ak.is_none(mask, axis=0)) == [False, False, True]
    assert ak.to_list(ak.num(mask, axis=1)) == [len(ROWS), 0, None]


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('data', [[], [[], []], [None, None], [[None], [], None]])
def test_empty_and_absent_particles_are_not_rejected(native, data):
    ak = pytest.importorskip('awkward')
    import lvec.awkward as lawk
    fields = {name: ak.Array(data) for name in ('px', 'py', 'pz', 'E')}
    value = lawk.zip(fields) if native else LVec(**fields)
    assert ak.to_list(value.is_physical()) == data
    value.validate_particle()
