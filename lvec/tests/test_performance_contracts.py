"""Behavioral contracts for structural validation and bounded property storage."""

import numpy as np
import pytest

from lvec import LVec, Vector2D, Vector3D, ShapeError
from lvec.caching import PropertyCache


def test_mass_only_retains_final_array_and_intermediates_are_opt_in():
    data = [np.full(1000, x) for x in (1., 2., 3., 5.)]
    final = LVec(*data)
    shared = LVec(*data, cache_intermediates=True)
    np.testing.assert_array_equal(final.mass, shared.mass)
    assert final.cache_size == 1
    assert final.cache_stats['overall']['retained_bytes'] == final.mass.nbytes
    assert shared.cache_stats['overall']['retained_bytes'] > final.mass.nbytes
    for vector in (Vector2D(data[0], data[1]), Vector3D(*data[:3])):
        _ = vector.r
        assert len(vector._cache) == 1


@pytest.mark.parametrize('size', [0, 1, 2])
def test_small_cache_returns_values_even_during_nested_computations(size):
    v = LVec(3., 4., 0., 13., max_cache_size=size, cache_intermediates=True)
    for _ in range(3):
        assert v.pt == 5.
        assert v.mass == 12.
        assert v.cache_size <= size


@pytest.mark.parametrize('value', [-1, 0.5, True, '1'])
def test_invalid_cache_size(value):
    with pytest.raises(ValueError, match='max_size'):
        PropertyCache(max_size=value)


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), True, '1'])
def test_invalid_ttl(value):
    with pytest.raises(ValueError, match='TTL'):
        PropertyCache(default_ttl=value)
    with pytest.raises(ValueError, match='TTL'):
        PropertyCache().set_ttl('mass', value)


def test_property_ttl_persists_before_access_recomputation_eviction_and_touch(monkeypatch):
    now = [100.]
    monkeypatch.setattr('lvec.caching.time.monotonic', lambda: now[0])
    cache = PropertyCache(max_size=1, default_ttl=100)
    cache.register_dependency('mass', ['E'])
    cache.set_ttl('mass', 2.)
    calls = []
    def compute():
        calls.append(None)
        return len(calls)
    assert cache.get_cached('mass', compute) == 1
    now[0] += 3
    assert cache.get_cached('mass', compute) == 2
    now[0] += 3
    assert cache.clear_expired() == 1
    assert cache.get_cached('mass', compute) == 3
    cache.get_cached('other', lambda: 0)  # Evict mass.
    assert cache.get_cached('mass', compute) == 4
    cache.touch_component('E')
    assert cache.get_cached('mass', compute) == 5
    cache.clear_cache()
    cache.get_cached('mass', compute)
    now[0] += 3
    assert cache.clear_expired() == 1
    cache.set_ttl('mass', None)  # Explicitly override default expiration.
    cache.get_cached('mass', compute)
    now[0] += 1000
    assert cache.clear_expired() == 0


def test_instrumentation_can_be_enabled_after_untracked_accesses():
    v = LVec(3., 4., 0., 13., cache_stats=False, cache_intermediates=True)
    assert v.mass == 12.
    assert v.cache_stats['overall']['total'] == 0
    v._cache.enable_instrumentation()
    assert v.mass == 12.
    v.clear_cache()
    assert v.mass == 12.
    assert v.cache_stats['properties']['mass']['hits'] == 1
    assert v.cache_stats['properties']['mass']['misses'] == 1


def test_component_touch_invalidates_transitive_dependencies():
    cache = PropertyCache()
    cache.register_dependency('p2', ['px'])
    cache.register_dependency('mass', ['p2', 'E'])
    cache.get_cached('p2', lambda: 1.)
    cache.get_cached('mass', lambda: 2.)
    cache.touch_component('px')
    assert len(cache) == 0


def test_nested_shape_mismatches_and_missing_positions_are_rejected():
    ak = pytest.importorskip('awkward')
    from lvec.utils import check_shapes
    cases = [
        ([[[1., 2.]], []], [[[3.]], []]),
        ([[None], [1.]], [[1.], [None]]),
        ([None, [1.]], [[1.], None]),
        ([[[1.], []]], [[[1.]], []]),
        ([[1.]], [1.]),
    ]
    for left, right in cases:
        with pytest.raises(ShapeError):
            check_shapes(ak.Array(left), ak.Array(right), 'ak')


def test_nested_empty_missing_and_regular_layouts_preserve_structure():
    ak = pytest.importorskip('awkward')
    from lvec.utils import check_shapes
    arrays = [
        ak.Array([[[1., None], []], None, []]),
        ak.Array([[[2., None], []], None, []]),
    ]
    check_shapes(*arrays, 'ak')
    regular = ak.Array(np.ones((3, 2, 4)))
    jagged = ak.Array(ak.to_list(regular))
    check_shapes(regular, jagged, 'ak')
    check_shapes(ak.Array([]), ak.Array([]), 'ak')
    check_shapes(ak.Array([[], []]), ak.Array([[], []]), 'ak')
    check_shapes(jagged, 'ak')
    check_shapes('ak')


def test_cached_property_results_cannot_be_mutated():
    data = [np.full(3, x) for x in (1., 2., 3., 5.)]
    v = LVec(*data)
    mass = v.mass
    with pytest.raises(ValueError, match='read-only'):
        mass[0] = 0.
    ak = pytest.importorskip('awkward')
    jagged = LVec(*(ak.Array([x]) for x in data))
    result = jagged.mass
    _, _, buffers = ak.to_buffers(result)
    assert all(not buffer.flags.writeable for buffer in buffers.values())
