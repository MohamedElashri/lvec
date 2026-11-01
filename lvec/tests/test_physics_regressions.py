"""Checks of independent physics identities and numerical edge cases."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from lvec import Frame, LVec, Vector2D, Vector3D
from lvec.exceptions import BackendError, InputError


@pytest.fixture(params=['scalar', 'numpy', 'awkward'])
def make_components(request):
    if request.param == 'scalar':
        return lambda *xs: xs
    if request.param == 'numpy':
        return lambda *xs: tuple(np.array([x]) for x in xs)
    ak = pytest.importorskip('awkward')
    return lambda *xs: tuple(ak.Array([[x], []]) for x in xs)


def flatten(value):
    if type(value).__module__.startswith('awkward'):
        import awkward as ak
        return ak.to_numpy(ak.flatten(value, axis=None))
    return np.asarray(value).reshape(-1)


def test_signed_mass_and_validation(make_components):
    v = LVec(*make_components(2., 0., 0., 1.))
    np.testing.assert_allclose(flatten(v.mass2), [-3.])
    np.testing.assert_allclose(flatten(v.mass), [-np.sqrt(3.)])
    assert not np.any(flatten(v.is_physical()))
    with pytest.raises(InputError):
        v.validate_particle()
    physical = LVec(*make_components(1., 2., 3., 5.), validate=True)
    assert physical.validate_particle() is physical
    past = physical * -1.
    assert np.all(flatten(past.E) < 0)
    assert not np.any(flatten(past.is_physical()))


@pytest.mark.parametrize('px,pz,expected', [(1e-6, 1., np.arcsinh(1e6)),
                                          (0., 1., np.inf),
                                          (0., -1., -np.inf),
                                          (0., 0., 0.)])
def test_stable_eta(make_components, px, pz, expected):
    with np.errstate(all='raise'):
        v = LVec(*make_components(px, 0., pz, 2.))
        np.testing.assert_allclose(flatten(v.eta), [expected])


def test_frame_paths_and_inverse(make_components):
    v = LVec(*make_components(1., 2., 3., 10.))
    cm = Frame.from_lvec(v)
    direct = v.to_frame(cm)
    explicit = v.transform_frame(Frame.rest(), cm)
    for a, b in zip(direct.to_p4(), explicit.to_p4()):
        np.testing.assert_allclose(flatten(a), flatten(b), atol=1e-13)
    for p in direct.to_p4()[:3]:
        np.testing.assert_allclose(flatten(p), [0.], atol=1e-13)
    restored = direct.transform_frame(cm, Frame.rest())
    for a, b in zip(v.to_p4(), restored.to_p4()):
        np.testing.assert_allclose(flatten(a), flatten(b), rtol=1e-13)
    source, target = Frame(.1, -.2, .05), Frame(-.1, .05, .2)
    back = v.transform_frame(source, target).transform_frame(target, source)
    for a, b in zip(v.to_p4(), back.to_p4()):
        np.testing.assert_allclose(flatten(a), flatten(b), rtol=1e-13)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, 1., -1.])
def test_invalid_boost_components(bad):
    with pytest.raises(InputError):
        Frame(bad, 0., 0.)
    with pytest.raises(InputError):
        LVec(0., 0., 0., 1.).boost(bad, 0., 0.)
    with pytest.raises(InputError):
        LVec(0., 0., 0., 1.).boost(np.array([0., bad]), np.zeros(2), np.zeros(2))


@pytest.mark.parametrize('constructor, args', [
    (LVec, (None, 0., 0., 1.)),
    (Vector2D, (None, 0.)),
    (Vector3D, (None, 0., 0.)),
    (Frame, (None, 0., 0.)),
])
def test_malformed_input_keeps_public_error(constructor, args):
    with pytest.raises(BackendError):
        constructor(*args)


def test_validation_tolerance_scales_with_units():
    for scale in [1e-9, 1., 1e9]:
        v = LVec(scale, 0., 0., scale * (1 - 1e-14))
        assert v.mass2 < 0
        assert v.is_physical(rtol=1e-12)
        assert not v.is_physical(rtol=0)
    for bad in [np.nan, -1., np.inf]:
        with pytest.raises(ValueError):
            v.is_physical(rtol=bad)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32])
def test_float64_physics_avoids_integer_overflow(dtype):
    magnitude = 100000 if dtype is np.int32 else 1e10
    px = np.array([magnitude], dtype=dtype)
    v = LVec(px, np.zeros_like(px), np.zeros_like(px), 2 * px)
    np.testing.assert_allclose(v.mass2, 3 * px.astype(np.float64)**2)
    assert v.px.dtype == np.float64
    scalar = LVec(dtype(1), dtype(2), dtype(3), dtype(5))
    assert scalar.mass2 == 11.


def test_components_are_owned_readonly_snapshots():
    px = np.array([3.])
    v = LVec(px, np.array([4.]), np.array([0.]), np.array([10.]))
    before = v.mass.copy()
    px[0] = 6.
    np.testing.assert_array_equal(v.px, [3.])
    np.testing.assert_array_equal(v.mass, before)
    with pytest.raises(ValueError):
        v.px[0] = 9.
    beta = np.array([.1])
    frame = Frame(beta, np.array([0.]), np.array([0.]))
    beta[0] = 2.
    np.testing.assert_array_equal(frame.beta_x, [.1])


def test_awkward_components_are_owned_snapshots():
    ak = pytest.importorskip('awkward')
    buffer = np.array([3.])
    x = ak.Array(buffer)
    v = LVec(x, ak.Array([4.]), ak.Array([0.]), ak.Array([10.]))
    buffer[0] = 6.
    np.testing.assert_array_equal(ak.to_numpy(v.px), [3.])
    with pytest.raises(ValueError):
        ak.to_numpy(v.px)[0] = 9.


def test_random_boosts_preserve_invariants():
    rng = np.random.default_rng(1729)
    p = rng.normal(size=(3, 300))
    q = rng.normal(size=(3, 300))
    v = LVec(*p, np.sqrt(np.sum(p*p, axis=0) + 4.))
    w = LVec(*q, np.sqrt(np.sum(q*q, axis=0) + 9.))
    def dot(a, b):
        return a.E*b.E - a.px*b.px - a.py*b.py - a.pz*b.pz
    for beta in [(0., 0., 0.), (1e-8, 2e-8, -1e-8), (.3, -.4, .1)]:
        vb, wb = v.boost(*beta), w.boost(*beta)
        np.testing.assert_allclose(vb.mass2, v.mass2, atol=1e-13)
        np.testing.assert_allclose(dot(vb, wb), dot(v, w), atol=1e-13)
        inverse = vb.boost(*(-x for x in beta))
        for a, b in zip(v.to_p4(), inverse.to_p4()):
            np.testing.assert_allclose(a, b, atol=1e-13)


def test_numpy_backend_works_when_awkward_import_is_blocked():
    script = textwrap.dedent('''
        import importlib.abc, sys
        class BlockAwkward(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'awkward' or fullname.startswith('awkward.'):
                    raise ImportError('Awkward deliberately absent')
        sys.meta_path.insert(0, BlockAwkward())
        from lvec import LVec, Frame, Vector2D, Vector3D
        from lvec.exceptions import DependencyError
        v = LVec(3., 4., 0., 7.)
        assert v.pt == 5.
        assert v.mass2 == 24.
        assert Vector2D(3., 4.).r == 5.
        assert Vector3D(0., 0., 2.).r == 2.
        assert abs(v.to_frame(Frame.from_lvec(v)).p) < 1e-12
        try:
            v.to_ak()
        except DependencyError:
            pass
        else:
            raise AssertionError('Missing optional backend should be reported')
    ''')
    result = subprocess.run([sys.executable, '-c', script], cwd=os.getcwd(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_constructor_ptepm_promotes_before_squaring():
    value = np.array([100000], dtype=np.int32)
    v = LVec.from_ptepm(value, np.zeros_like(value), np.zeros_like(value), value)
    np.testing.assert_allclose(v.mass, [100000.])
    scalar = LVec.from_ptepm(np.float32(1), np.float32(0), np.float32(0), np.float32(2))
    np.testing.assert_allclose(scalar.mass, 2.)
    assert Frame(np.float32(.1), np.float32(0), np.float32(0)).gamma > 1
    assert LVec(0., 0., 0., 1.).boost(np.float32(.1), np.float32(0), np.float32(0)).px > 0


def test_rest_frame_rejects_lightlike_and_spacelike_systems():
    for v in [LVec(1., 0., 0., 1.), LVec(2., 0., 0., 1.), LVec(np.nan, 0., 0., 1.)]:
        with pytest.raises(InputError):
            Frame.from_lvec(v)


def test_missing_awkward_entries_are_preserved():
    ak = pytest.importorskip('awkward')
    v = LVec(ak.Array([[1., None], [], None]), ak.Array([[0., None], [], None]),
             ak.Array([[0., None], [], None]), ak.Array([[2., None], [], None]), validate=True)
    assert ak.to_list(v.eta) == [[0., None], [], None]
    assert ak.to_list(v.is_physical()) == [[True, None], [], None]


def test_center_of_mass_accepts_generators():
    particles = [LVec(1., 0., 0., 2.), LVec(0., 1., 0., 2.)]
    frame = Frame.center_of_mass(v for v in particles)
    total = particles[0] + particles[1]
    np.testing.assert_allclose(total.to_frame(frame).p, 0., atol=1e-13)
    with pytest.raises(ValueError):
        Frame.center_of_mass(iter(()))


def test_nonfinite_particle_validation_is_warning_free():
    with np.errstate(all='raise'):
        v = LVec(np.array([np.inf, np.nan]), np.zeros(2), np.zeros(2), np.array([np.inf, 1.]))
        assert not np.any(v.is_physical())
        with pytest.raises(InputError):
            v.validate_particle()
        with pytest.raises(InputError):
            Frame(np.array([1e300]), np.array([0.]), np.array([0.]))
