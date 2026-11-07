"""Seeded, independent invariants and adversarial floating-point inputs."""

from decimal import Decimal, localcontext
import gc
import weakref

import numpy as np
import pytest

from lvec import Frame, LVec, Vector2D, Vector3D

# These tests exercise recovery, which the default fast policy skips.
pytestmark = pytest.mark.usefixtures('accurate_policy')


def present_values(value):
    if type(value).__module__.startswith('awkward'):
        import awkward as ak
        return ak.to_numpy(ak.flatten(value, axis=None))
    return np.asarray(value).reshape(-1)


@pytest.fixture(params=['scalar', 'numpy', 'awkward'])
def components(request):
    if request.param == 'scalar':
        return lambda *values: values
    if request.param == 'numpy':
        return lambda *values: tuple(np.array([v]) for v in values)
    ak = pytest.importorskip('awkward')
    return lambda *values: tuple(ak.Array([[v, None], [], None]) for v in values)


@pytest.mark.parametrize('scale', [1e-200, 1e-160, 1e-100, 1., 1e100, 1e160, 1e200])
def test_representable_magnitudes_and_particle_classification(components, scale):
    with np.errstate(all='raise'):
        v = LVec(*components(3*scale, 4*scale, 0., 13*scale))
        np.testing.assert_allclose(present_values(v.pt) / scale, 5., rtol=3e-14)
        np.testing.assert_allclose(present_values(v.p) / scale, 5., rtol=3e-14)
        np.testing.assert_allclose(present_values(v.mass) / scale, 12., rtol=3e-14)
        assert np.all(present_values(v.is_physical()))
        spacelike = LVec(*components(2*scale, 0., 0., scale))
        assert not np.any(present_values(spacelike.is_physical()))
        np.testing.assert_allclose(present_values(spacelike.mass) / scale,
                                   -np.sqrt(3.), rtol=3e-14)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_euclidean_magnitudes_do_not_square_out_of_range(scale):
    with np.errstate(all='raise'):
        assert Vector2D(3*scale, 4*scale).r / scale == pytest.approx(5.)
        v = Vector3D(3*scale, 4*scale, 12*scale)
        assert v.r / scale == pytest.approx(13.)
        assert v.rho / scale == pytest.approx(5.)


def test_eta_when_ratio_overflows_but_angle_is_representable(components):
    with np.errstate(all='raise'):
        v = LVec(*components(1e-200, 0., 1e200, 2e200))
        expected = 400*np.log(10.) + np.log(2.)
        np.testing.assert_allclose(present_values(v.eta), expected, rtol=1e-14)


@pytest.mark.parametrize('pt,eta', [(1e-200, 700.), (1e-300, 1000.), (1e150, 20.)])
def test_coordinate_constructor_avoids_unrepresentable_intermediates(pt, eta):
    with localcontext() as context:
        context.prec = 80
        x = Decimal.from_float(eta)
        pz = float(Decimal.from_float(pt) * (x.exp() - (-x).exp()) / 2)
    with np.errstate(all='raise'):
        v = LVec.from_ptepm(pt, eta, 0., 0.)
    assert np.isfinite(v.E)
    assert v.pz == pytest.approx(pz, rel=2e-13)
    assert v.E == pytest.approx(np.hypot(pt, pz), rel=2e-13)


@pytest.mark.parametrize('theta', [1e-3, 1e-6, 1e-9, 1e-12])
def test_collimated_opening_angles_keep_small_nonzero_angles(components, theta):
    a = LVec(*components(1., 0., 0., 2.))
    b = LVec(*components(np.cos(theta), np.sin(theta), 0., 2.))
    np.testing.assert_allclose(present_values(a.opening_angle(b)), theta,
                               rtol=2e-12, atol=1e-16)


@pytest.mark.parametrize('seed', [1701, 1702, 1703])
def test_random_boost_rotation_and_frame_invariants(seed):
    rng = np.random.default_rng(seed)
    count = 20000
    scale = 10.**rng.uniform(-100, 100, count)
    p = rng.normal(size=(3, count)) * scale
    energy = np.hypot(np.hypot(np.hypot(*p[:2]), p[2]), 2*scale)
    v = LVec(*p, energy)
    beta = rng.normal(size=(3, count))
    beta *= rng.uniform(0., .95, count) / np.linalg.norm(beta, axis=0)
    boosted = v.boost(*beta)
    for actual, original in zip(boosted.boost(*(-beta)).to_p4(), v.to_p4()):
        np.testing.assert_allclose(actual / scale, original / scale,
                                   rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(boosted.mass2 / scale**2, v.mass2 / scale**2,
                               rtol=2e-12, atol=2e-12)
    rotated = v.rotx(.23).roty(-.56).rotz(.89)
    np.testing.assert_allclose(rotated.mass2 / scale**2, v.mass2 / scale**2,
                               rtol=2e-13, atol=2e-13)
    rest = v.to_frame(Frame.from_lvec(v))
    np.testing.assert_allclose(rest.p / scale, 0., atol=5e-14)


def test_boost_against_decimal_reference_near_light_speed():
    v = LVec(.3, -.4, .7, 2.)
    for beta in [0., 1e-12, .5, .99, 1-1e-8, np.nextafter(1., 0.)]:
        with localcontext() as context:
            context.prec = 80
            b = Decimal.from_float(beta)
            gamma = 1 / (1-b*b).sqrt()
            pz, energy = Decimal.from_float(v.pz), Decimal.from_float(v.E)
            expected = float(gamma*(pz+b*energy)), float(gamma*(energy+b*pz))
        result = v.boostz(beta)
        np.testing.assert_allclose([result.pz, result.E], expected, rtol=2e-9)


@pytest.mark.parametrize('size', [0, 1, 3, None])
@pytest.mark.parametrize('intermediates', [False, True])
def test_cache_churn_does_not_retain_dead_vectors(size, intermediates):
    rng = np.random.default_rng(1729)
    references = []
    for _ in range(100):
        p = rng.normal(size=(3, 128))
        v = LVec(*p, np.sqrt(np.sum(p*p, axis=0)+4),
                 max_cache_size=size, cache_intermediates=intermediates)
        for property_name in ('mass', 'mass2', 'p2', 'pt2', 'eta', 'pt', 'p', 'rapidity', 'gamma'):
            result = getattr(v, property_name)
            assert np.all(np.isfinite(result))
        # Magnitude calculations may reuse fresh temporaries but must never
        # turn a retained squared property into its square root.
        np.testing.assert_allclose(v.p2, np.sum(p*p, axis=0), rtol=1e-14)
        np.testing.assert_allclose(v.pt2, np.sum(p[:2]*p[:2], axis=0), rtol=1e-14)
        np.testing.assert_allclose(v.mass2, 4., rtol=1e-13)
        if size is not None:
            assert v.cache_size <= size
        references.extend((weakref.ref(v), weakref.ref(result)))
        del v, result
    gc.collect()
    assert all(reference() is None for reference in references)


def test_seeded_jagged_missing_event_roundtrip_and_conservation():
    ak = pytest.importorskip('awkward')
    rng = np.random.default_rng(1800)
    counts = rng.integers(0, 12, size=1000)
    p = rng.normal(size=(3, counts.sum()))
    data = (*p, np.sqrt(np.sum(p*p, axis=0)+4.))
    present_particle = rng.random(counts.sum()) > .15
    present_event = rng.random(len(counts)) > .1
    arrays = tuple(ak.mask(ak.unflatten(ak.mask(x, present_particle), counts),
                          present_event) for x in data)
    v = LVec(*arrays)
    boosted = v.boost(.2, -.1, .3)
    for source, restored in zip(v.to_p4(), LVec.from_ak_record(v.to_ak_record()).to_p4()):
        assert ak.to_list(source) == ak.to_list(restored)
    np.testing.assert_allclose(present_values(boosted.mass2), present_values(v.mass2),
                               rtol=2e-13, atol=2e-13)
    total = v.sum(axis=-1)
    for component, source in zip(total.to_p4(), arrays):
        assert ak.to_list(component) == ak.to_list(ak.sum(source, axis=-1))


@pytest.mark.parametrize('seed', [2001, 2002])
def test_two_body_decay_conservation_and_helicity_across_frames(seed):
    rng = np.random.default_rng(seed)
    count = 10000
    mass = 10.**rng.uniform(-100, 100, count)
    # Two-body phase space with daughter masses 0.2 M and 0.3 M.
    momentum = mass * np.sqrt((1-.5**2)*(1-.1**2)) / 2
    cosine = rng.uniform(-1., 1., count)
    phi = rng.uniform(-np.pi, np.pi, count)
    sine = np.sqrt((1-cosine)*(1+cosine))
    p = np.array([momentum*sine*np.cos(phi), momentum*sine*np.sin(phi), momentum*cosine])
    a = LVec(*p, np.hypot(momentum, .2*mass))
    b = LVec(*(-p), np.hypot(momentum, .3*mass))
    parent = a+b
    np.testing.assert_allclose(parent.E / mass, 1., rtol=2e-15)
    mother = LVec(np.zeros(count), np.zeros(count), -mass, 2*mass)
    beta = rng.normal(size=(3, count))
    beta *= rng.uniform(0., .9995, count) / np.linalg.norm(beta, axis=0)
    boosted_a, boosted_b = a.boost(*beta), b.boost(*beta)
    total = boosted_a+boosted_b
    np.testing.assert_allclose(total.mass / mass, 1., rtol=2e-10)
    np.testing.assert_allclose(boosted_a.mass / mass, .2, rtol=2e-10)
    rest = total.to_frame(Frame.from_lvec(total))
    np.testing.assert_allclose(rest.p / mass, 0., atol=3e-11)
    angles = boosted_a.helicity_angle(parent.boost(*beta), mother.boost(*beta))
    np.testing.assert_allclose(angles, np.arccos(cosine), atol=1e-10)


def test_massless_invariants_use_condition_appropriate_tolerance():
    rng = np.random.default_rng(2100)
    direction = rng.normal(size=(3, 10000))
    direction /= np.linalg.norm(direction, axis=0)
    energy = 10.**rng.uniform(-100, 100, 10000)
    photons = LVec(*(direction*energy), energy).boost(.5, -.3, .2)
    # Near-null norms lose relative mass accuracy in float64. Bound error by
    # the squared components, rather than demanding exactly zero photon mass.
    assert np.max(abs(photons.mass2) / photons.E**2) < 5e-14


@pytest.mark.parametrize('scale', [1e-200, 1., 1e200])
def test_null_vectors_keep_zero_signed_norm_without_invalid_intermediates(components, scale):
    with np.errstate(all='raise'):
        v = LVec(*components(scale, 0., 0., scale))
        np.testing.assert_array_equal(present_values(v.mass2), [0.])
        np.testing.assert_array_equal(present_values(v.mass), [0.])


@pytest.mark.parametrize('regular', [False, True])
def test_angular_properties_do_not_retain_unnecessary_missing_value_buffers(regular):
    ak = pytest.importorskip('awkward')
    values = (ak.Array(np.ones((10000, 2))) if regular
              else ak.unflatten(np.ones(20000), np.full(10000, 2)))
    v = LVec(values, values, values, 5.)
    for property_name in ('eta', 'rapidity', 'beta', 'gamma'):
        result = getattr(v, property_name)
        assert result.nbytes <= 1.5 * v.px.nbytes
        assert len(result) == len(values)
        assert not ak.any(ak.is_none(result, axis=-1))


def test_dot_product_cancellation_with_large_finite_components(components):
    with np.errstate(all='raise'):
        a = LVec(*components(1e200, 0., 0., 1e200))
        b = LVec(*components(2e200, 0., 0., 2e200))
        np.testing.assert_array_equal(present_values(a.dot(b)), [0.])
        wide = LVec(*components(3e200, 4e200, 0., 13e200))
        small = LVec(*components(0., 0., 2e-200, 5e-200))
        np.testing.assert_allclose(present_values(wide.dot(small)), 65., rtol=1e-14)


def test_overflowing_dot_fallback_preserves_scalar_array_broadcasting():
    a = LVec(1e200, 0., 0., 1e200)
    b = LVec(np.full((2, 3), 2e200), 0., 0., 2e200)
    with np.errstate(all='raise'):
        np.testing.assert_array_equal(a.dot(b), np.zeros((2, 3)))


def test_extreme_hierarchy_preserves_transverse_invariant():
    v = LVec(1e200, 1., 0., 1e200)
    assert v.mass2 == -1.
    assert v.mass == -1.
    assert v.dot(v) == -1.
    assert not v.is_physical(rtol=0)
