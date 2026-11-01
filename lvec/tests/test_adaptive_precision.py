"""Accuracy oracles and fast-path contracts for adaptive arithmetic."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from unittest.mock import patch

import numpy as np
import pytest

from lvec import LVec, Frame
from lvec.vectors import Vec2D, Vec3D
from lvec import _adaptive as adaptive
from lvec.exceptions import InputError


def boost_reference(p, beta):
    with localcontext() as context:
        context.prec = 1500
        p = list(map(Decimal.from_float, map(float, p)))
        b = list(map(Decimal.from_float, map(float, beta)))
        b2 = sum(x*x for x in b)
        gamma = 1/(1-b2).sqrt()
        bp = sum(x*y for x, y in zip(p, b))
        factor = (gamma-1)*bp/b2 if b2 else Decimal(0)
        return tuple(float(p[i]+b[i]*(factor+gamma*p[3])) for i in range(3))+(float(gamma*(p[3]+bp)),)


def test_ordinary_transforms_never_use_precision_fallbacks():
    rng = np.random.default_rng(391)
    p = rng.normal(size=(4, 10000))
    p[3] = np.sqrt(np.sum(p[:3]**2, axis=0)+4)
    v = LVec(*p)
    with patch.object(adaptive, 'accurate_result', side_effect=AssertionError('exact rotation')):
        with patch.object(adaptive, '_boost_decimal', side_effect=AssertionError('decimal boost')):
            v.rotx(.21).roty(.47).rotz(-.32).boost(.1, -.2, .3)
            v.boost_to_rest()


@pytest.mark.parametrize('p,beta', [
    ((1e308, 0, 0, 1e308), (-.9, 0, 0)),
    ((1e308, 0, 0, 1e308), (.9, 0, 0)),
    ((1e308, -1e308, 1, 1), (.6, .6, 0)),
    ((1e-310, 2e-310, 0, 4e-310), (-.9, 0, 0)),
    ((1., 0, 0, 1.), (-np.nextafter(1., 0.), 0, 0)),
    ((2., 3, 1, 4.), (.28, .96, 0)),
])
def test_risky_boost_matches_high_precision(p, beta):
    result = LVec(*p).boost(*beta).to_p4()
    np.testing.assert_allclose(result, boost_reference(p, beta), rtol=3e-15, atol=5e-324)


@pytest.mark.parametrize('backend', ['np', 'ak'])
def test_only_one_boost_in_mixed_batch_is_recovered(backend):
    p = [np.full(1000, x) for x in (1., 2., 3., 5.)]
    p[0][417] = p[3][417] = 1e308
    if backend == 'ak':
        ak = pytest.importorskip('awkward')
        p = [ak.unflatten(x, [300, 0, 400, 300]) for x in p]
    with patch.object(adaptive, '_boost_decimal', wraps=adaptive._boost_decimal) as recover:
        result = LVec(*p).boost(-.9, 0, 0)
    assert recover.call_count == 1
    px = ak.to_numpy(ak.flatten(result.px)) if backend == 'ak' else result.px
    assert px[417] == boost_reference((1e308, 2., 3., 1e308), (-.9, 0., 0.))[0]


def test_near_boundary_beta_validation_uses_original_values():
    with localcontext() as context:
        context.prec = 100
        beta = list(map(Decimal.from_float, (.28, .96)))
        gamma = float(1/(1-sum(x*x for x in beta)).sqrt())
    assert Frame(.28, .96, 0).gamma == gamma
    with pytest.raises(InputError):
        Frame(.6, .8, 0)  # exact squared magnitude of these floats exceeds one


@pytest.mark.parametrize('scale', [1., 1e308, 1e-310])
@pytest.mark.parametrize('cls', [Vec2D, Vec3D, LVec])
def test_rotation_cancellation_matches_exact_products(cls, scale):
    angle = np.pi/4
    c, s = np.cos(angle), np.sin(angle)
    expected = float(Fraction(float(c))*Fraction(scale)-Fraction(float(s))*Fraction(scale))
    values = (scale, scale) if cls is Vec2D else ((scale, scale, 0.) if cls is Vec3D else (scale, scale, 0., 1.))
    result = cls(*values).rotate(angle)
    assert (result.px if cls is LVec else result.x) == expected


def test_rotation_only_recovers_cancelled_rows():
    x, y = np.full(1000, 2.), np.full(1000, 3.)
    x[51] = y[51] = 1e308
    with patch.object(adaptive, 'accurate_result', wraps=adaptive.accurate_result) as recover:
        LVec(x, y, 0, 5).rotz(np.pi/4)
    assert recover.call_count == 1
    assert recover.call_args.args[0][0].size == 1


@pytest.mark.parametrize('axis', [None, 0, 1, -1, (0, 1), (), np.int64(1)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_reduction_axes_and_keepdims(axis, keepdims):
    if isinstance(axis, np.integer):
        axis = int(axis)
    values = np.array([[1e16, 1., -1e16], [1e308, 1e308, -1e308]])
    result = LVec(values, values, values, values).sum(axis, keepdims=keepdims).px
    if axis is None or axis == (0, 1):
        expected = math.fsum([1., 1e308])
    elif axis == ():
        expected = values
    elif axis == 0:
        expected = np.array([1e308, 1e308, -1e308])
    else:
        expected = np.array([1., 1e308])
    if keepdims and axis != ():
        expected = np.reshape(expected, np.sum(np.ones_like(values), axis=axis, keepdims=True).shape)
    np.testing.assert_array_equal(result, expected)


def test_numpy_sum_recovers_only_risky_groups():
    values = np.tile([1., 2., 3.], (1000, 1))
    values[12] = [1e16, 1., -1e16]
    with patch.object(adaptive, '_sum_rows', wraps=adaptive._sum_rows) as recover:
        result = LVec(values, 0, 0, values).sum(-1)
    assert recover.call_count == 2  # px and E; zero py/pz need no recovery
    assert all(call.args[0].shape == (1, 3) for call in recover.call_args_list)
    assert result.px[12] == 1.


def test_ordinary_sums_never_use_precision_fallback():
    values = np.arange(1., 4001.).reshape(1000, 4)
    with patch.object(adaptive, '_sum_rows', side_effect=AssertionError('sum fallback')):
        LVec(values, 0, 0, values).sum(-1)


@pytest.mark.parametrize('axis', [None, 0, 1, -1])
@pytest.mark.parametrize('keepdims', [False, True])
def test_awkward_missing_reductions(axis, keepdims):
    ak = pytest.importorskip('awkward')
    values = ak.Array([[1e16, None, 1., -1e16], None, [], [1., 2., 3., 4.]])
    result = LVec(values, 0, 0, values).sum(axis, keepdims=keepdims).px
    expected = {None: 11., 0: [1e16, 2., 4., -9999999999999996.],
                1: [1., None, 0., 10.], -1: [1., None, 0., 10.]}[axis]
    if keepdims:
        if axis is None:
            expected = [[expected]]
        elif axis == 0:
            expected = [expected]
        else:
            expected = [None if x is None else [x] for x in expected]
    assert (ak.to_list(result) if isinstance(result, ak.Array) else result) == expected


@pytest.mark.parametrize('axis', [0, 1, 2, -1])
@pytest.mark.parametrize('keepdims', [False, True])
def test_deep_awkward_reduction_coordinates(axis, keepdims):
    ak = pytest.importorskip('awkward')
    # Each axis sees cancellation in a different output group.
    values = ak.Array([[[1e16, 1., -1e16], [1., 2., 3.]],
                       [[1., 2., 3.], None],
                       [[-1e16, 4., 1e16], [4., 5., 6.]], None])
    # Independent exact reference, replacing each binary float by an integer
    # before summation (all test inputs are exactly represented integers).
    integers = ak.values_astype(values, np.int64)
    expected = ak.values_astype(ak.sum(integers, axis=axis, keepdims=keepdims, mask_identity=False), np.float64)
    result = LVec(values, 0, 0, values).sum(axis, keepdims=keepdims).px
    assert ak.to_list(result) == ak.to_list(expected)


def test_reductions_preserve_nonfinite_inputs_and_correct_true_overflow():
    values = np.array([[np.inf, 1., -np.inf], [np.nan, 1., 2.],
                       [1e308, 1e308, 1e308], [1e308, 1e308, -1e308]])
    result = LVec(values, 0, 0, values).sum(-1).px
    assert np.isnan(result[:2]).all()
    assert result[2] == np.inf
    assert result[3] == 1e308


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('keepdims', [False, True])
def test_irregular_reduction_stress_against_integer_oracle(axis, keepdims):
    ak = pytest.importorskip('awkward')
    rng = np.random.default_rng(731)
    data = rng.integers(-20, 20, size=(5, 5, 5)).astype(np.float64)
    # Force severe cancellation along each axis, with small residuals.
    data[0, :, :] = 1e16
    data[4, :, :] = -1e16
    data[1:4, 0, :] = 1e16
    data[1:4, 4, :] = -1e16
    data[1:4, 1:4, 0] = 1e16
    data[1:4, 1:4, 4] = -1e16
    values = ak.Array(data.tolist())
    values = ak.mask(values, ak.local_index(values, axis=0) != 2)
    values = values[:, :, ::2]  # unequal slicing, unused content buffers
    integers = ak.values_astype(values, np.int64)
    expected = ak.values_astype(ak.sum(integers, axis=axis, keepdims=keepdims), np.float64)
    result = LVec(values, 0, 0, values).sum(axis, keepdims=keepdims).px
    assert ak.to_list(result) == ak.to_list(expected)


def test_nan_in_one_row_does_not_disable_other_rows_recovery():
    values = np.array([[np.nan, 1., 2.], [1e16, 1., -1e16]])
    result = LVec(values, 0, 0, values).sum(-1).px
    assert np.isnan(result[0])
    assert result[1] == 1.
    v = LVec([np.nan, 1e308], [0., 0.], [0., 0.], [1., 1e308])
    assert v.boost(-.9, 0, 0).px[1] == boost_reference((1e308, 0., 0., 1e308), (-.9, 0., 0.))[0]


def test_awkward_transform_masks_survive_precision_recovery():
    ak = pytest.importorskip('awkward')
    x = ak.Array([[1e308, None], None, [], [2.]])
    y = ak.Array([[1e308, None], None, [], [3.]])
    rotated = LVec(x, y, 0, 1.).rotz(np.pi/4)
    expected = float(Fraction(float(np.cos(np.pi/4)))*Fraction(1e308)
                     - Fraction(float(np.sin(np.pi/4)))*Fraction(1e308))
    assert ak.to_list(rotated.px) == [[expected, None], None, [], [2*np.cos(np.pi/4)-3*np.sin(np.pi/4)]]
    boosted = LVec(x, 0, 0, x).boost(-.9, 0, 0)
    expected_boost = boost_reference((1e308, 0., 0., 1e308), (-.9, 0., 0.))[0]
    assert ak.to_list(boosted.px)[0] == [expected_boost, None]
    assert ak.to_list(boosted.px)[1:3] == [None, []]


def test_mixed_awkward_beta_boundary():
    ak = pytest.importorskip('awkward')
    bx = ak.Array([[.1, .28], [], None])
    by = ak.Array([[.2, .96], [], None])
    frame = Frame(bx, by, 0.)
    assert ak.to_list(frame.gamma)[0][1] == Frame(.28, .96, 0).gamma
    result = LVec(1., 2., 3., 4.).boost(bx, by, 0.)
    assert ak.to_list(result.E)[1:] == [[], None]
    assert ak.to_list(result.E)[0][1] == pytest.approx(boost_reference((1., 2., 3., 4.), (.28, .96, 0.))[3])


def test_float32_angles_are_evaluated_at_component_precision():
    angle = np.array([.47, 1.2], dtype=np.float32)
    result = LVec([1., 1.], [0., 0.], 0., 2.).rotz(angle)
    np.testing.assert_array_equal(result.px, np.cos(angle.astype(np.float64)))
    np.testing.assert_array_equal(result.py, np.sin(angle.astype(np.float64)))


@pytest.mark.parametrize('backend', ['np', 'ak'])
def test_long_reduction_with_intermediate_overflow(backend):
    values = np.r_[1e308, 1e308, np.ones(100), -1e308, -1e308]
    if backend == 'ak':
        ak = pytest.importorskip('awkward')
        values = ak.Array([values.tolist(), [], [1., 2.]])
    result = LVec(values, 0., 0., values).sum(-1).px
    assert (ak.to_list(result) if backend == 'ak' else result) == ([100., 0., 3.] if backend == 'ak' else 100.)


def test_compensated_short_groups_do_not_iterate_over_events():
    values = np.tile([1., 2., -3.], (10000, 1))
    with patch.object(adaptive, '_finite_sum', side_effect=AssertionError('scalar sum')):
        result = LVec(values, 0., 0., values).sum(-1)
    assert not np.any(result.px)


def test_boost_recovery_is_independent_of_callers_decimal_context():
    from decimal import Inexact
    with localcontext() as context:
        context.prec = 6
        context.traps[Inexact] = True
        result = LVec(1e308, 0., 0., 1e308).boost(-.9, 0., 0.)
    assert result.px == boost_reference((1e308, 0., 0., 1e308), (-.9, 0., 0.))[0]
