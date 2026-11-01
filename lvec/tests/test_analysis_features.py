"""Analytic and cross-backend checks for physics analysis helpers."""
import numpy as np
import pytest

from lvec import LVec, ShapeError
from lvec.exceptions import InputError


def test_minkowski_product_and_squared_momenta():
    v = LVec(3., 4., 0., 13.)
    w = LVec(0., 0., 2., 5.)
    assert v.pt2 == 25.
    assert v.p2 == 25.
    assert v.dot(v) == 144.
    assert v.dot(w) == 65.
    np.testing.assert_allclose(v.boost(.2, -.1, .3).dot(w.boost(.2, -.1, .3)),
                               v.dot(w), rtol=1e-13)
    v.px  # Component access does not affect derived values.
    v.touch()
    assert v.pt2 == 25.
    with pytest.raises(TypeError):
        v.dot(2.)


def test_rapidity_beta_gamma_analytic_and_degenerate():
    y = .8
    v = LVec(0., 0., 2 * np.sinh(y), 2 * np.cosh(y))
    assert v.rapidity == pytest.approx(y)
    assert v.beta == pytest.approx(np.tanh(y))
    assert v.gamma == pytest.approx(np.cosh(y))
    photon = LVec(0., 0., 1., 1.)
    assert photon.rapidity == np.inf
    assert photon.beta == 1.
    assert photon.gamma == np.inf
    assert LVec(0., 0., -1., 1.).rapidity == -np.inf
    zero = LVec(0., 0., 0., 0.)
    assert np.isnan(zero.rapidity)
    assert np.isnan(zero.beta)
    assert np.isnan(zero.gamma)
    spacelike = LVec(2., 0., 0., 1.)
    assert spacelike.beta == 2.
    assert np.isnan(spacelike.gamma)
    assert np.isnan(LVec(0., 0., 2., 1.).rapidity)
    # Division pz/E would round away information near the light cone.
    energy, pz = 1e16, 1e16 - 2.
    assert LVec(0., 0., pz, energy).rapidity == pytest.approx(
        .5 * (np.log(energy + pz) - np.log(energy - pz)), rel=1e-14)
    # Scaling the logarithm arguments also avoids overflowing E+pz.
    huge = LVec(0., 0., 1e308, 1.5e308)
    assert huge.rapidity == pytest.approx(.5 * np.log(5.))
    assert huge.beta == pytest.approx(2. / 3.)
    assert huge.gamma == pytest.approx(3. / np.sqrt(5.))


def test_delta_phi_branch_cut_and_delta_r():
    a = LVec.from_ptepm(2., .3, np.pi - .01, 1.)
    b = LVec.from_ptepm(3., -.2, -np.pi + .01, 2.)
    assert a.delta_phi(b) == pytest.approx(-.02)
    assert b.delta_phi(a) == pytest.approx(.02)
    assert a.delta_r2(b) == pytest.approx(.5**2 + .02**2, abs=1e-9)
    assert a.delta_r(b) == pytest.approx(np.sqrt(.5**2 + .02**2), abs=1e-9)
    assert LVec(1., 0., 0., 2.).delta_phi(LVec(-1., 0., 0., 2.)) == -np.pi
    boosted_a, boosted_b = a.boostz(.4), b.boostz(.4)
    assert a.delta_r2(b, rapidity=True) == pytest.approx(
        boosted_a.delta_r2(boosted_b, rapidity=True), rel=1e-12)


def test_numpy_scalar_broadcast_and_strict_array_shapes():
    v = LVec(np.array([1., 2., 3.]), 0., np.array(0.), np.float64(5.))
    np.testing.assert_array_equal(v.py, np.zeros(3))
    np.testing.assert_array_equal(v.E, np.full(3, 5.))
    np.testing.assert_array_equal(v.pt2, [1., 4., 9.])
    assert LVec(np.array(1.), np.int64(0), 0., 2.).px == 1.
    with pytest.raises(ShapeError):
        LVec(np.ones(3), np.zeros(1), 0., 5.)
    with pytest.raises(ShapeError):
        LVec(np.ones((2, 1)), np.ones((1, 2)), 0., 5.)


def test_numpy_sum_total_eventwise_and_keepdims():
    v = LVec(np.array([[1., 2.], [3., 4.]]), 0., 0., 10.)
    total = v.sum()
    assert total.to_p4() == (10., 0., 0., 40.)
    np.testing.assert_array_equal(v.sum(axis=-1).px, [3., 7.])
    assert v.sum(axis=1, keepdims=True).px.shape == (2, 1)
    empty = LVec(np.empty((2, 0)), 0., 0., 1.)
    np.testing.assert_array_equal(empty.sum(axis=-1).E, [0., 0.])


def test_boost_to_rest_and_opening_angles():
    parent = LVec(1., 2., 3., 8.)
    rest = parent.boost_to_rest()
    np.testing.assert_allclose(rest.to_p4()[:3], 0., atol=1e-14)
    assert rest.E == pytest.approx(np.sqrt(parent.dot(parent)))
    x, y = LVec(1., 0., 0., 2.), LVec(0., 1., 0., 2.)
    assert x.opening_angle(y) == pytest.approx(np.pi / 2)
    assert x.opening_angle(x) == pytest.approx(0.)
    assert x.opening_angle(LVec(-1., 0., 0., 2.)) == pytest.approx(np.pi)
    assert np.isnan(x.opening_angle(LVec(0., 0., 0., 1.)))
    for invalid in [LVec(1., 0., 0., 1.), LVec(2., 0., 0., 1.),
                    LVec(0., 0., 0., 0.)]:
        with pytest.raises(InputError):
            x.boost_to_rest(invalid)


def test_helicity_angle_convention_and_common_boost_invariance():
    # In R's rest frame, the mother P travels along -z. Daughter d at theta
    # relative to +z therefore has helicity angle theta in P -> R -> d.
    theta = .7
    daughter = LVec(np.sin(theta), 0., np.cos(theta), 2.)
    parent = LVec(0., 0., 0., 5.)
    mother = LVec(0., 0., -3., 10.)
    assert daughter.helicity_angle(parent, mother) == pytest.approx(theta)
    beta = (.2, -.15, .3)
    assert daughter.boost(*beta).helicity_angle(
        parent.boost(*beta), mother.boost(*beta)) == pytest.approx(theta, abs=1e-13)
    assert np.isnan(daughter.helicity_angle(parent, LVec(0., 0., 0., 10.)))


def test_numpy_properties_match_vector():
    vector = pytest.importorskip('vector')
    rng = np.random.default_rng(1623)
    px, py, pz = rng.normal(size=(3, 40))
    energy = np.sqrt(px**2 + py**2 + pz**2 + .7**2)
    v = LVec(px, py, pz, energy)
    # Generic coordinate names avoid vector's deprecated in-place dtype rename
    # under NumPy 2.5; the public momentum view supplies momentum aliases.
    oracle = vector.array(dict(x=px, y=py, z=pz, t=energy)).view(vector.MomentumNumpy4D)
    np.testing.assert_allclose(v.rapidity, oracle.rapidity, rtol=1e-12)
    np.testing.assert_allclose(v.beta, oracle.beta, rtol=1e-12)
    np.testing.assert_allclose(v.gamma, oracle.gamma, rtol=1e-12)
    np.testing.assert_allclose(v.dot(v), oracle.mass2, rtol=1e-12)


def test_awkward_scalar_broadcast_records_and_reductions():
    ak = pytest.importorskip('awkward')
    px = ak.Array([[1., None, 2.], [], [3.]])
    v = LVec(px, 0., np.array(0.), 5.)
    assert ak.to_list(v.py) == [[0., None, 0.], [], [0.]]
    assert ak.to_list(v.pt2) == [[1., None, 4.], [], [9.]]
    assert ak.to_list(v.sum(axis=-1).px) == [3., 0., 3.]
    assert v.sum().to_p4() == (6., 0., 0., 15.)
    assert ak.to_list(v.sum(axis=-1, keepdims=True).px) == [[3.], [0.], [3.]]
    records = v.to_ak_record()
    assert ak.to_list(records)[0][1] is None
    rebuilt = LVec.from_ak_record(records)
    assert ak.to_list(rebuilt.px) == ak.to_list(v.px)
    assert isinstance(v.to_ak(), dict)
    scalar = LVec(1., 0., 0., 2.)
    assert isinstance(scalar.to_ak_record(), ak.Record)
    assert LVec.from_ak_record(scalar.to_ak_record()).to_p4() == scalar.to_p4()
    with pytest.raises(InputError):
        LVec.from_ak_record(ak.Array([{'px': 1.}]))
    with pytest.raises(ShapeError):
        LVec(ak.Array([[1., 2.], [3.]]), ak.Array([[0.], [0.]]), 0., 5.)


def test_awkward_properties_angles_and_missing_events():
    ak = pytest.importorskip('awkward')
    px = ak.Array([[1., None], [], None, [0.]])
    v = LVec(px, 0., 0., 2.)
    assert ak.to_list(v.rapidity) == [[0., None], [], None, [0.]]
    beta, gamma = ak.to_list(v.beta), ak.to_list(v.gamma)
    assert beta == [[.5, None], [], None, [0.]]
    assert gamma[0][0] == pytest.approx(2. / np.sqrt(3.))
    assert gamma[0][1] is None and gamma[1:3] == [[], None]
    assert ak.to_list(v.sum(axis=-1).px) == [1., 0., None, 0.]
    assert ak.to_list(v.delta_r(v, rapidity=True)) == [[0., None], [], None, [0.]]
    angles = ak.to_list(v.opening_angle(v))
    assert angles[:3] == [[0., None], [], None]
    assert np.isnan(angles[3][0])
    rest = v.boost_to_rest()
    assert ak.to_list(ak.num(rest.px, axis=1)) == [2, 0, None, 1]
    assert ak.to_list(ak.is_none(rest.px, axis=1)) == [[False, True], [], None, [False]]
    np.testing.assert_allclose(ak.to_numpy(ak.flatten(rest.px, axis=None)),
                               0., atol=1e-14)


@pytest.mark.parametrize('missing_level', ['particle', 'event', 'nested_list'])
def test_awkward_reduction_rejects_incomplete_four_vectors(missing_level):
    ak = pytest.importorskip('awkward')
    if missing_level == 'particle':
        px, py = ak.Array([[1., None], []]), ak.Array([[None, 1.], []])
    elif missing_level == 'event':
        px, py = ak.Array([None, [1.]]), ak.Array([[1.], None])
    else:
        px = ak.Array([[[1.], None], []])
        py = ak.Array([[None, [1.]], []])
    # Newer constructor validation may reject these immediately. The reducer
    # also guards the contract when used independently of that implementation.
    with pytest.raises(ShapeError):
        LVec(px, py, px, px).sum(axis=-1)
