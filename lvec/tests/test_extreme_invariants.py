"""Independent rational/decimal references for cancellation and range extremes."""

from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from lvec import LVec

# These tests exercise recovery, which the default fast policy skips.
pytestmark = pytest.mark.usefixtures('accurate_policy')


def reference(left, right=None):
    right = left if right is None else right
    terms = [Fraction.from_float(float(a))*Fraction.from_float(float(b))
             for a, b in zip(left, right)]
    result = terms[3]-terms[0]-terms[1]-terms[2]
    try:
        squared = float(result)
    except OverflowError:
        squared = math.copysign(math.inf, -1 if result < 0 else 1)
    with localcontext() as context:
        context.prec = 120
        value = Decimal(abs(result.numerator))/Decimal(result.denominator)
        root = float(value.sqrt())
    return squared, math.copysign(root, -1 if result < 0 else 1)


def assert_float_close(actual, expected, ulps=4):
    actual = np.asarray(actual).reshape(-1)
    expected = np.asarray(expected).reshape(-1)
    for a, e in zip(actual, expected):
        if not math.isfinite(e) or e == 0:
            assert a == e
            assert np.signbit(a) == np.signbit(e)
        else:
            assert abs(float(a)-float(e)) <= ulps*math.ulp(float(e))


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('scale', [1e100, 1e200, 1e300])
@pytest.mark.parametrize('residual', [1., 1e-200, 1e100])
def test_axis_permutations_keep_small_residuals(axis, scale, residual):
    components = [0., 0., 0., -scale]
    components[axis] = scale
    components[(axis+1) % 3] = residual
    v = LVec(*components)
    expected, root = reference(components)
    with np.errstate(all='raise'):
        assert_float_close(v.mass2, [expected], ulps=0)
        assert_float_close(v.mass, [root], ulps=0)
        assert_float_close(v.dot(LVec(*components)), [expected], ulps=0)


@pytest.mark.parametrize('exponent', [100, 500, 900])
def test_multiple_large_terms_cancel_before_small_terms(exponent):
    scale = math.ldexp(1., exponent)
    v = LVec(3*scale, 4*scale, 1., 5*scale)
    assert v.mass2 == -1.
    assert v.mass == -1.
    assert v.dot(LVec(*v.to_p4())) == -1.
    assert not v.is_physical(rtol=0, atol=.5)
    assert v.is_physical(rtol=0, atol=1.)


def test_squared_underflow_retains_signed_mass_and_strict_classification():
    v = LVec(1e200, 1e-200, 0., 1e200)
    assert v.mass2 == 0 and np.signbit(v.mass2)
    assert v.mass == -1e-200
    assert not v.is_physical(rtol=0)
    assert v.is_physical(rtol=0, atol=np.nextafter(0., 1.))


def test_subnormal_rounding_uses_the_final_output_grid():
    a = math.ldexp(1., -538)
    half = LVec(1., a, a, 1.)
    assert half.mass2 == 0 and np.signbit(half.mass2)
    one_and_half = LVec(2*a, a, a, 0.)
    assert one_and_half.mass2 == -2*np.nextafter(0., 1.)
    assert_float_close(one_and_half.mass, [-math.sqrt(6.)*a], ulps=1)


def test_mass_can_be_finite_when_its_square_overflows():
    maximum = np.finfo(float).max
    v = LVec(0., 0., 0., maximum)
    assert v.mass2 == np.inf
    assert v.mass == maximum


def adversarial_rows():
    rng = np.random.default_rng(27012026)
    rows = []
    for index in range(300):
        scale = math.ldexp(float(rng.uniform(.5, 1.)), int(rng.integers(-1000, 1000)))
        residual = math.ldexp(float(rng.uniform(.5, 1.)), int(rng.integers(-1000, 1000)))
        p = [0., 0., 0.]
        p[index % 3] = scale
        p[(index+1) % 3] = residual
        energy = scale if index % 2 else float(np.nextafter(scale, np.inf))
        rows.append((*p, energy))
    return np.asarray(rows)


def test_numpy_matches_independent_fraction_and_decimal_references():
    data = adversarial_rows()
    expected = np.array([reference(row) for row in data])
    with np.errstate(all='raise'):
        v = LVec(*data.T)
        # Extreme/cancelling cases require the accurate path. For ordinary
        # arithmetic, measure the usual rounding bound against input scales.
        for name, actual, target in [('mass2', v.mass2, expected[:, 0]),
                                      ('mass', v.mass, expected[:, 1]),
                                      ('dot', v.dot(LVec(*data.T)), expected[:, 0])]:
            for row, a, e, exact_m2 in zip(data, actual, target, expected[:, 0]):
                with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                    energy2 = float(np.square(row[3]))
                severe = (not math.isfinite(energy2) or not math.isfinite(exact_m2)
                          or energy2 < np.finfo(float).tiny
                          or abs(exact_m2) <= math.sqrt(np.finfo(float).eps)*energy2)
                if severe or energy2 == 0 or not math.isfinite(e) or e == 0:
                    assert_float_close([a], [e])
                else:
                    magnitude = float(max(abs(x) for x in row))
                    tolerance = (8*float(np.finfo(float).eps)*magnitude)*magnitude
                    if name == 'mass':
                        tolerance /= abs(e)
                    assert abs(float(a)-float(e)) <= tolerance


def test_jagged_options_and_empty_events_keep_high_precision_results():
    ak = pytest.importorskip('awkward')
    data = adversarial_rows()[:60]
    arrays = tuple(ak.Array([[value, None] if i % 3 == 0 else [] if i % 3 == 1 else None
                              for i, value in enumerate(component)]) for component in data.T)
    v = LVec(*arrays)
    expected = np.array([reference(row) for i, row in enumerate(data) if i % 3 == 0])
    with np.errstate(all='raise'):
        for actual, wanted in [(v.mass2, expected[:, 0]), (v.mass, expected[:, 1]),
                               (v.dot(LVec(*arrays)), expected[:, 0])]:
            assert_float_close(ak.to_numpy(ak.flatten(actual, axis=None)), wanted)
            for i, event in enumerate(ak.to_list(actual)):
                if i % 3 == 1:
                    assert event == []
                elif i % 3 == 2:
                    assert event is None
                else:
                    assert event[1] is None


def test_compensated_ordinary_near_null_batch_matches_exact_products():
    rng = np.random.default_rng(27012027)
    p = rng.normal(size=(3, 500))
    energy = np.sqrt(np.sum(p*p, axis=0))
    data = np.array([*p, energy]).T
    wanted = np.array([reference(row) for row in data])
    v = LVec(*(component.reshape(20, 25) for component in data.T))
    assert_float_close(v.mass2, wanted[:, 0], ulps=1)
    assert_float_close(v.mass, wanted[:, 1], ulps=1)


def test_spacelike_dot_products_use_spatial_cancellation_bounds():
    scale = np.geomspace(1e8, 1e300, 80)
    left = np.array([scale, np.ones(80), scale, np.zeros(80)]).T
    right = np.array([scale, -np.ones(80), -np.nextafter(scale, np.inf), np.zeros(80)]).T
    expected = np.array([reference(a, b)[0] for a, b in zip(left, right)])
    with np.errstate(all='raise'):
        actual = LVec(*left.T).dot(LVec(*right.T))
    assert_float_close(actual, expected)
