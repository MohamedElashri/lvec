"""Physics, structure, and ownership contracts for Awkward optimizations."""

import numpy as np
import pytest

from lvec import LVec, ShapeError
from lvec.exceptions import BackendError
from lvec.utils import check_shapes


@pytest.fixture
def ak():
    return pytest.importorskip('awkward')


@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_plain_jagged_structure_uses_exact_offsets_and_accepts_slices(ak, monkeypatch, index_dtype):
    def array(offsets, values):
        return ak.Array(ak.contents.ListOffsetArray(
            ak.index.Index(np.asarray(offsets, dtype=index_dtype)),
            ak.contents.NumpyArray(np.asarray(values, dtype=float))))

    left = array([0, 2, 2, 3], [1., 2., 3., 100.])
    shifted = array([2, 4, 4, 5], [100., 100., 4., 5., 6., 100.])
    wrong = array([0, 1, 2, 3], [1., 2., 3.])
    monkeypatch.setattr(ak, 'almost_equal', lambda *a, **k: pytest.fail('generic structural comparison'))
    check_shapes(left, shifted, 'ak')
    check_shapes(left[1:], shifted[1:], 'ak')
    with pytest.raises(ShapeError):
        check_shapes(left, wrong, 'ak')
    with pytest.raises(ShapeError):
        LVec(left, wrong, left, left)


def test_shape_validation_matches_independent_nested_structure_oracle(ak):
    def structure(value):
        if isinstance(value, list):
            return tuple(structure(x) for x in value)
        return None if value is None else 'number'

    rng = np.random.default_rng(20261102)
    samples = [[], [[], []], [None, []], [[1., None], []],
               [[[1.], []], None], [[[]], []]]
    for _ in range(40):
        sample = []
        for _ in range(int(rng.integers(1, 8))):
            sample.append(None if rng.random() < .2 else
                          [None if rng.random() < .2 else float(rng.normal())
                           for _ in range(int(rng.integers(0, 6)))])
        samples.append(sample)
    for index, left in enumerate(samples):
        for right in (left, samples[(index + 1) % len(samples)]):
            a, b = ak.Array(left), ak.Array(right)
            if structure(left) == structure(right):
                check_shapes(a, b, 'ak')
            else:
                with pytest.raises(ShapeError):
                    check_shapes(a, b, 'ak')


def test_general_layouts_retain_structural_validation(ak):
    ordinary = ak.Array([[1., 2.], [], [3.]])
    layout = ordinary.layout
    lists = ak.contents.ListArray(layout.offsets[:-1], layout.offsets[1:], layout.content)
    check_shapes(ak.Array(lists), ordinary, 'ak')
    regular = ak.Array(np.ones((3, 2)))
    check_shapes(regular, ak.Array([[4., 5.], [6., 7.], [8., 9.]]), 'ak')
    for left, right in [([[1., None], []], [[None, 1.], []]),
                        ([None, [1.]], [[1.], None]),
                        ([[[1.], None]], [[None, [1.]]])]:
        with pytest.raises(ShapeError):
            check_shapes(ak.Array(left), ak.Array(right), 'ak')


def test_exact_nested_structure_handles_shifted_unused_content(ak, monkeypatch):
    left = ak.Array([[[1., 2.], []], [], [[3.], [4., 5.]]])
    right = ak.Array([[[100.]], [[6., 7.], []], [], [[8.], [9., 10.]], [[]]])[1:-1]
    wrong = ak.Array([[[6., 7.], []], [], [[8., 9.], [10.]]])
    monkeypatch.setattr(ak, 'almost_equal', lambda *a, **k: pytest.fail('generic structural comparison'))
    check_shapes(left, right, 'ak')
    check_shapes(left[1:], right[1:], 'ak')
    with pytest.raises(ShapeError):
        check_shapes(left, wrong, 'ak')


@pytest.mark.parametrize('regular_layout', [False, True])
def test_exact_regular_structure_preserves_dimensions(ak, monkeypatch, regular_layout):
    left = ak.Array(np.ones((3, 2, 4)))
    if regular_layout:
        left = ak.to_regular(ak.to_regular(ak.Array(ak.to_list(left)), axis=2), axis=1)
    right = left * 2.
    monkeypatch.setattr(ak, 'almost_equal', lambda *a, **k: pytest.fail('generic structural comparison'))
    check_shapes(left, right, 'ak')
    with pytest.raises(ShapeError):
        check_shapes(left, right[:, :, :3], 'ak')


def test_random_nested_plain_shapes_match_python_oracle(ak):
    rng = np.random.default_rng(20261104)

    def make(depth):
        if depth == 0:
            return float(rng.normal())
        return [make(depth - 1) for _ in range(int(rng.integers(0, 5)))]

    def structure(value):
        return tuple(structure(item) for item in value) if isinstance(value, list) else 'number'

    samples = [make(3) for _ in range(40)]
    for left, right in zip(samples, samples[1:] + samples[:1]):
        check_shapes(ak.Array(left), ak.Array(left), 'ak')
        if structure(left) != structure(right):
            with pytest.raises(ShapeError):
                check_shapes(ak.Array(left), ak.Array(right), 'ak')
        else:
            check_shapes(ak.Array(left), ak.Array(right), 'ak')


def test_semantic_layouts_keep_logical_shape_comparison(ak):
    # Character storage lengths are not dimensions of the logical string array.
    check_shapes(ak.Array(['a', 'long']), ak.Array(['longer', 'b']), 'ak')


def shaped(ak, values, kind):
    values = np.asarray(values)
    if kind == 'regular':
        return ak.Array(values.reshape(2, 4))
    if kind == 'lists':
        return ak.Array([values[:3].tolist(), [], values[3:].tolist()])
    if kind == 'sliced':
        return ak.Array([[-99.], values[:3].tolist(), [], values[3:].tolist(), [-99.]])[1:-1]
    if kind == 'list_array':
        return ak.Array(ak.contents.ListArray(ak.index.Index64(np.array([4, 0, 8, 2])),
                                            ak.index.Index64(np.array([8, 3, 8, 4])),
                                            ak.contents.NumpyArray(values)))
    if kind == 'indexed':
        return ak.Array(ak.contents.IndexedArray(ak.index.Index64(np.array([7, 0, 3, 0, 2])),
                                               ak.contents.NumpyArray(values)))
    if kind == 'byte_mask':
        return ak.Array(ak.contents.ByteMaskedArray(ak.index.Index8(np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=np.int8)),
                                                  ak.contents.NumpyArray(values), valid_when=True))
    if kind == 'bit_mask':
        return ak.Array(ak.contents.BitMaskedArray(ak.index.IndexU8(np.array([0b11110101], dtype=np.uint8)),
                                                 ak.contents.NumpyArray(values), valid_when=True,
                                                 length=8, lsb_order=True))
    if kind == 'missing':
        return ak.Array([values[:3].tolist() + [None], [], None, values[3:].tolist()])
    return ak.Array([[values[:3].tolist(), []], None, [[None] + values[3:].tolist()]])


@pytest.mark.usefixtures('accurate_policy')
@pytest.mark.parametrize('kind', ['regular', 'lists', 'sliced', 'list_array', 'indexed',
                                 'byte_mask', 'bit_mask', 'missing', 'deep'])
def test_mass_leaf_kernel_matches_exact_oracles_and_preserves_structure(ak, kind):
    from .test_extreme_invariants import reference, assert_float_close
    data = np.array([(3., 4., 0., 13.), (2., 0., 0., 1.), (1., 0., 0., 1.),
                     (1e200, 1., 0., 1e200), (1e200, 1e-200, 0., 1e200),
                     (0., 0., 0., np.finfo(float).max), (1e-200, 0., 0., 0.),
                     (3e100, 4e100, 1., 5e100)])
    expected = np.array([reference(row) for row in data])
    v = LVec(*(shaped(ak, component, kind) for component in data.T))
    before = tuple(ak.to_list(c) for c in v.to_p4())
    with np.errstate(all='raise'):
        for property_name, column in [('mass2', 0), ('mass', 1)]:
            actual = getattr(v, property_name)
            wanted = shaped(ak, expected[:, column], kind)
            assert ak.to_list(ak.ones_like(actual)) == ak.to_list(ak.ones_like(wanted))
            assert_float_close(ak.to_numpy(ak.flatten(actual, axis=None)),
                               ak.to_numpy(ak.flatten(wanted, axis=None)), ulps=4)
    assert before == tuple(ak.to_list(c) for c in v.to_p4())


@pytest.mark.parametrize('intermediates', [False, True])
@pytest.mark.parametrize('max_size', [None, 0, 1, 3])
def test_mass_paths_preserve_cache_controls(ak, intermediates, max_size):
    data = tuple(ak.Array([[x, None], [], None, [x]]) for x in (3., 4., 0., 13.))
    v = LVec(*data, cache_intermediates=intermediates, max_cache_size=max_size,
             cache_stats=False)
    for _ in range(2):
        assert ak.to_list(v.mass) == [[12., None], [], None, [12.]]
        assert ak.to_list(v.mass2) == [[144., None], [], None, [144.]]
        if max_size is not None:
            assert v.cache_size <= max_size
        assert v.cache_stats['overall']['total'] == 0
        v.clear_cache()
    if intermediates and max_size is None:
        v.mass
        assert v.cache_size > 1


@pytest.mark.parametrize('values', [[], [[], []], [None, None], [[None], [], None]])
def test_mass_leaf_kernel_handles_unknown_empty_and_all_missing_arrays(ak, values):
    component = ak.Array(values)
    v = LVec(component, component, component, component)
    assert ak.to_list(v.mass) == values
    assert ak.to_list(v.mass2) == values


@pytest.mark.usefixtures('accurate_policy')
def test_mass_leaf_kernel_preserves_nonfinite_values(ak):
    v = LVec(ak.Array([[np.nan, np.inf, 0., 1e200]]), 0., 0.,
             ak.Array([[1., np.inf, np.inf, 1e200]]))
    for name in ('mass', 'mass2'):
        actual = ak.to_numpy(ak.flatten(getattr(v, name)))
        assert np.isnan(actual[:2]).all()
        assert actual[2] == np.inf
        assert actual[3] == 0.


@pytest.mark.parametrize('kind', ['regular', 'lists', 'sliced', 'list_array', 'indexed',
                                 'byte_mask', 'bit_mask', 'missing', 'deep'])
@pytest.mark.parametrize('beta', [(0., 0., 0.), (.1, -.2, .3), (-.9, 0., 0.),
                                 (.28, .96, 0.), (np.nextafter(1., 0.), 0., 0.)])
def test_scalar_boost_leaves_match_decimal_across_layouts(ak, kind, beta):
    from .test_adaptive_precision import boost_reference
    data = np.array([(1., 2., 3., 5.), (0., 0., 0., 0.), (1., 0., 0., 1.),
                     (1e308, 1., 0., 1e308), (1e-310, 2e-310, 0., 4e-310),
                     (-1e308, 0., 1., 1e308), (2., 3., 1., 4.),
                     (3e100, 4e100, 1., 5e100)])
    expected = np.asarray([boost_reference(row, beta) for row in data])
    v = LVec(*(shaped(ak, component, kind) for component in data.T))
    with np.errstate(all='raise'):
        boosted = v.boost(*beta)
    for component, wanted in zip(boosted.to_p4(), expected.T):
        reference = shaped(ak, wanted, kind)
        assert ak.to_list(ak.ones_like(component)) == ak.to_list(ak.ones_like(reference))
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(component, axis=None)),
                                   ak.to_numpy(ak.flatten(reference, axis=None)),
                                   rtol=3e-14, atol=5e-324)


def test_scalar_boost_parameters_stay_scalar_and_recovery_remains_selective(ak, monkeypatch):
    from lvec import _adaptive
    original = _adaptive._constant_beta_numpy
    original_recover = _adaptive._boost_decimal
    recoveries = []
    calls = []

    def checked(values,beta,gamma):
        assert all(np.isscalar(parameter) for parameter in (*beta,gamma))
        calls.append(None)
        return original(values,beta,gamma)

    def recover(row):
        recoveries.append(None)
        return original_recover(row)

    monkeypatch.setattr(_adaptive, '_constant_beta_numpy', checked)
    monkeypatch.setattr(_adaptive, '_boost_decimal', recover)
    x = ak.Array([[1., None], [], None, [1e308]])
    v = LVec(x, 0., 0., ak.Array([[5., None], [], None, [1e308]]))
    result = v.boost(-.9, 0., 0.)
    assert calls and len(recoveries) == 1
    assert np.isfinite(ak.to_numpy(ak.flatten(result.E, axis=None))).all()


@pytest.mark.parametrize('values', [[], [[], []], [None, None], [[None], [], None]])
def test_scalar_boost_preserves_empty_and_all_missing_arrays(ak, values):
    v = LVec(*(ak.Array(values) for _ in range(4)))
    assert all(ak.to_list(component) == values for component in v.boost(.1, .2, .3).to_p4())


@pytest.mark.parametrize('kind', ['regular', 'lists', 'sliced', 'list_array', 'indexed',
                                 'byte_mask', 'bit_mask', 'missing', 'deep'])
def test_owned_results_preserve_physics_structure_and_readonly_components(ak, kind):
    p = np.arange(24, dtype=float).reshape(3, 8) / 8
    energy = np.sqrt(np.sum(p * p, axis=0) + 4.)
    data = tuple(shaped(ak, c, kind) for c in (*p, energy))
    v = LVec(*data)
    before = tuple(ak.to_list(c) for c in v.to_p4())
    for result, expected_mass in [(v + v, 4.), (v * 2., 4.), (v - v, 0.),
                                  (v.boost(.2, -.1, .3), 2.),
                                  (v.rotx(.3).roty(-.2).rotz(.1), 2.)]:
        reference = ak.ones_like(data[0]) * expected_mass
        assert ak.to_list(ak.ones_like(result.mass)) == ak.to_list(ak.ones_like(reference))
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(result.mass, axis=None)),
                                   ak.to_numpy(ak.flatten(reference, axis=None)), atol=2e-13)
        for component in result.to_p4():
            _, _, buffers = ak.to_buffers(component)
            assert all(not buffer.flags.writeable for buffer in buffers.values())
            assert all(buffer.dtype == np.float64 for key, buffer in buffers.items()
                       if key.endswith('-data'))
    total = v.sum(axis=-1)
    for actual, source in zip(total.to_p4(), data):
        expected = ak.sum(source, axis=-1)
        if isinstance(expected, ak.Array):
            assert ak.to_list(ak.ones_like(actual)) == ak.to_list(ak.ones_like(expected))
            np.testing.assert_allclose(ak.to_numpy(ak.flatten(actual, axis=None)),
                                       ak.to_numpy(ak.flatten(expected, axis=None)))
        else:
            np.testing.assert_allclose(actual, expected)
    assert before == tuple(ak.to_list(c) for c in v.to_p4())


def test_owned_results_reuse_safe_buffers_and_public_construction_still_snapshots(ak, monkeypatch):
    import lvec.lvec as implementation
    v = LVec(*(ak.Array([[x, None], [], [x]]) for x in (1., 2., 3., 5.)))
    original = implementation.own_components

    def snapshots_only_scalars(*components):
        assert all(np.isscalar(value) for value in components)
        return original(*components)

    monkeypatch.setattr(implementation, 'own_components', snapshots_only_scalars)
    results = [v + v, v - v, v * 2., v.boost(.1, .2, .3), v.rotx(.3), v.sum(-1)]
    for source, reused in [(v.px, results[4].px), (v.E, results[4].E)]:
        assert np.shares_memory(ak.to_layout(source).content.content.data,
                               ak.to_layout(reused).content.content.data)
    monkeypatch.setattr(implementation, 'own_components', original)
    offsets = np.array([0, 2, 2, 3])
    values = np.array([1., 2., 3.])
    external = ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets),
                                                 ak.contents.NumpyArray(values)))
    for factory in (LVec, LVec._from_owned):
        copied = factory(external, external, external, external)
        assert not np.shares_memory(copied.px.layout.content.data, values)
        assert not np.shares_memory(copied.px.layout.offsets.data, offsets)
    assert values.flags.writeable and offsets.flags.writeable


@pytest.mark.parametrize('operation', ['boost', 'rotation', 'scale', 'custom_scale', 'index'])
def test_external_parameter_buffers_cannot_mutate_results(ak, operation):
    offsets = np.array([0, 2, 2, 3])
    values = np.array([.1, .2, .3])
    parameter = ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(offsets),
                                                  ak.contents.NumpyArray(values)))
    v = LVec(*(ak.Array([[x, x], [], [x]]) for x in (1., 2., 3., 5.)))
    if operation == 'boost':
        result = v.boost(parameter, 0., 0.)
    elif operation == 'rotation':
        result = v.rotz(parameter)
    elif operation == 'scale':
        result = v * parameter
    elif operation == 'custom_scale':
        class ExternalScalar(float):
            def __mul__(self, other):
                return parameter
        result = v * ExternalScalar(2.)
    else:
        # An indexed selection can retain a caller-owned integer buffer.
        values = np.array([2, 0], dtype=np.int64)
        result = v[ak.Array(values)]
    before = tuple(ak.to_list(c) for c in result.to_p4())
    mass = ak.to_list(result.mass)
    assert values.flags.writeable and offsets.flags.writeable
    values[:] = 0
    offsets[1] = 1
    result.clear_cache()
    assert before == tuple(ak.to_list(c) for c in result.to_p4())
    assert mass == ak.to_list(result.mass)


def test_subclasses_retain_constructor_and_reduction_behavior(ak):
    class Child(LVec):
        def __init__(self, *components, **kwargs):
            super().__init__(*components, **kwargs)
            self.initialized = True

    v = Child(*(ak.Array([[x], []]) for x in (1., 2., 3., 5.)))
    assert isinstance(v.sum(-1), Child) and v.sum(-1).initialized
    result = Child._from_owned(*v.to_p4(), _trusted_awkward=True)
    assert isinstance(result, Child) and result.initialized


def test_owned_arithmetic_handles_different_input_masks_and_mixed_backends(ak):
    a = LVec(*(ak.Array([[value, None], [value], None, []]) for value in (1., 2., 3., 5.)))
    b = LVec(*(ak.Array([[None, value], [value], None, []]) for value in (2., 1., 3., 7.)))
    scalar = LVec(1., 2., 3., 5.)
    for left, right in [(a, b), (a, scalar), (scalar, a)]:
        for result, sign in [(left + right, 1), (left - right, -1)]:
            for actual, x, y in zip(result.to_p4(), left.to_p4(), right.to_p4()):
                assert ak.to_list(actual) == ak.to_list(x + sign * y)


@pytest.mark.parametrize('kind', ['regular', 'lists', 'sliced', 'list_array', 'indexed',
                                 'byte_mask', 'bit_mask', 'missing', 'deep'])
def test_cache_policy_preserves_random_near_null_physics_across_exponents(ak, kind):
    rng = np.random.default_rng(20261103)
    for _ in range(3):
        scale = np.power(10., rng.uniform(-250, 250, 8))
        p = rng.normal(size=(3, 8)) * scale
        energy = np.hypot(np.hypot(p[0], p[1]), p[2])
        energy[::2] = np.nextafter(energy[::2], np.inf)
        data = tuple(shaped(ak, c, kind) for c in (*p, energy))
        direct = LVec(*data)
        intermediate = LVec(*data, cache_intermediates=True)
        with np.errstate(all='raise'):
            for name in ('mass', 'mass2'):
                a, b = getattr(direct, name), getattr(intermediate, name)
                assert ak.to_list(ak.ones_like(a)) == ak.to_list(ak.ones_like(b))
                a, b = (ak.to_numpy(ak.flatten(x, axis=None)) for x in (a, b))
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(np.signbit(a[a == 0]), np.signbit(b[b == 0]))


@pytest.mark.parametrize('scalar', [1j, 2. + 0j, np.complex64(1j), np.complex128(2.)])
def test_complex_scaling_remains_rejected(ak, scalar):
    v = LVec(*(ak.Array([[value]]) for value in (1., 2., 3., 5.)))
    with pytest.raises((BackendError, TypeError)):
        v * scalar


@pytest.mark.parametrize('scalar', [2, 2., True, np.float32(2), np.float64(2),
                                   pytest.param(np.longdouble(2), marks=pytest.mark.skipif(
                                       np.dtype(np.longdouble).itemsize > 8,
                                       reason='Awkward does not support extended real dtypes')),
                                   np.int64(2), np.bool_(True)])
def test_real_scaling_preserves_float64_readonly_snapshots(ak, scalar):
    v = LVec(*(ak.Array([[value]]) for value in (1., 2., 3., 5.)))
    result = v * scalar
    for actual, original in zip(result.to_p4(), v.to_p4()):
        assert ak.to_list(actual) == ak.to_list(original * float(scalar))
        _, _, buffers = ak.to_buffers(actual)
        assert all(buffer.dtype == np.float64 for key, buffer in buffers.items()
                   if key.endswith('-data'))
        assert all(not buffer.flags.writeable for buffer in buffers.values())
