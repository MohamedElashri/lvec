"""Physics, structure, and ownership contracts for Awkward optimizations."""

import numpy as np
import pytest

from lvec import LVec, ShapeError
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


def test_scalar_boost_parameters_are_views_and_recovery_remains_selective(ak, monkeypatch):
    from lvec import _adaptive
    original = _adaptive._boost_numpy
    original_recover = _adaptive._boost_decimal
    recoveries = []
    calls = []

    def checked(*args):
        for parameter in args[4:]:
            assert parameter.shape == args[0].shape
            assert all(stride == 0 for stride in parameter.strides)
        calls.append(None)
        return original(*args)

    def recover(row):
        recoveries.append(None)
        return original_recover(row)

    monkeypatch.setattr(_adaptive, '_boost_numpy', checked)
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
