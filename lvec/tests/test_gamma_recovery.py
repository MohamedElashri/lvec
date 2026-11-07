"""Independent gamma references, domain checks, and layout/cache regressions."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from lvec import LVec

# These tests exercise recovery, which the default fast policy skips.
pytestmark = pytest.mark.usefixtures('accurate_policy')


def reference(row):
    if not all(math.isfinite(x) for x in row) or row[3] <= 0:
        return math.nan
    values = [Fraction.from_float(float(x)) for x in row]
    norm = values[3]**2 - sum(x*x for x in values[:3])
    if norm < 0:
        return math.nan
    if norm == 0:
        return math.inf
    with localcontext() as context:
        context.prec = 800
        mass = (Decimal(norm.numerator) / Decimal(norm.denominator)).sqrt()
        return float(Decimal.from_float(row[3]) / mass)


ROWS = [(1., 1e-10, 0., 1.), (.6, .8, 0., 1.), (.28, .96, 0., 1.),
        (0., 0., 0., 2.), (0., 0., 1., 1.), (2., 0., 0., 1.),
        (0., 0., 0., 0.), (0., 0., 0., -2.),
        (math.nan, 0., 0., 1.), (0., 0., 0., math.inf)]


@pytest.mark.parametrize('row', ROWS)
def test_gamma_scalar_reference_and_domains(row):
    np.testing.assert_allclose(LVec(*row).gamma, reference(row), rtol=3e-15, equal_nan=True)


@pytest.mark.parametrize('exponent', [-1000, -500, 0, 500, 1000])
def test_near_null_gamma_is_invariant_under_binary_scaling(exponent):
    row = tuple(math.ldexp(x, exponent) for x in (.28, .96, 0., 1.))
    np.testing.assert_allclose(LVec(*row).gamma, reference(row), rtol=3e-15)


@pytest.mark.parametrize('size', [1, 8, 17, 257])
def test_gamma_numpy_strided_mixed_values_and_cache(size):
    rows = np.array([ROWS[i % len(ROWS)] for i in range(size)])
    storage = np.repeat(rows, 2, axis=0)
    view = storage[::2]
    p = LVec(*view.T)
    expected = [reference(tuple(row)) for row in rows]
    result = p.gamma
    np.testing.assert_allclose(result, expected, rtol=3e-15, equal_nan=True)
    assert p.cache_size == 1
    assert p.gamma is result
    storage[:] = 0
    np.testing.assert_allclose(p.gamma, expected, rtol=3e-15, equal_nan=True)
    p.clear_cache()
    np.testing.assert_allclose(p.gamma, expected, rtol=3e-15, equal_nan=True)
    assert p.cache_size == 1


@pytest.mark.parametrize('native', [False, True])
def test_gamma_awkward_missing_nested_and_indexed(native):
    ak = pytest.importorskip('awkward')
    import lvec.awkward as lawk
    indices = [[2, 0, None, 4], [], None, [3, 5, 2]]
    columns = {name: ak.Array([None if event is None else [None if index is None else ROWS[index][i]
                                  for index in event] for event in indices])
               for i, name in enumerate(('px', 'py', 'pz', 'E'))}
    p = lawk.zip(columns) if native else LVec(**columns)
    # Force an indexed outer layout, including a repeated event.
    p = p[ak.Array([3, 1, 2, 0, 3])]
    result = p.gamma
    assert ak.to_list(ak.is_none(result, axis=0)) == [False, False, True, False, False]
    assert ak.to_list(ak.is_none(result, axis=-1)) == [[False, False, False], [], None,
                                                   [False, False, True, False], [False, False, False]]
    expected = [reference(ROWS[i]) for i in (3, 5, 2, 2, 0, 4, 3, 5, 2)]
    np.testing.assert_allclose(ak.to_numpy(ak.flatten(result, axis=None)), expected, rtol=3e-15, equal_nan=True)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('data', [[], [[], []], [None, None], [[None], [], None]])
def test_gamma_empty_and_all_missing(native, data):
    ak = pytest.importorskip('awkward')
    import lvec.awkward as lawk
    columns = {name: ak.Array(data) for name in ('px', 'py', 'pz', 'E')}
    p = lawk.zip(columns) if native else LVec(**columns)
    assert ak.to_list(p.gamma) == data


def test_gamma_ordinary_random_inputs():
    rng = np.random.default_rng(39)
    xyz = rng.normal(size=(100, 3))
    energy = np.hypot(np.linalg.norm(xyz, axis=1), 2.)
    rows = np.column_stack((xyz, energy))
    np.testing.assert_allclose(LVec(*rows.T).gamma,
                               [reference(tuple(row)) for row in rows], rtol=3e-15)
