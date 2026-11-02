# Arrays, ownership, and caching

## Collection shapes and selection

Owned vectors contain independent, read-only float64 snapshots. Mutating an
input array cannot change a collection or a cached result. Scalars broadcast;
non-scalar component shapes and missing masks must match. A length-one array
is not a scalar. Use `lvec.zip` for explicit event-to-particle broadcasting.

Indexing and Boolean masks select complete vectors. `shape`, `ndim`, and
`reshape(*shape, order='C')` describe regular collections. For jagged data,
inspect the component layout with Awkward operations such as `ak.num`.

## Reductions and NumPy operations

```{py:method} lvec.LVec.sum(axis=None, dtype=None, out=None, keepdims=False, *, initial=None, where=None)
Sum components to form a total four-momentum. `axis=None` reduces all
collection dimensions. For jagged events, `axis=-1` sums particles within
each event; `keepdims=True` retains that axis. NumPy collections also accept
axis tuples.

Empty groups give zero vectors. Missing particles are skipped as whole
vectors; missing events stay missing when reducing their particle axis.
```

The extra NumPy-compatible arguments `dtype`, `out`, `initial`, and `where`
must retain their defaults; non-default values raise `TypeError`.

NumPy supports `sum`, `count_nonzero`, `stack`, `concatenate`,
`broadcast_arrays`, `reshape`, `isclose`, `allclose`, arithmetic, absolute,
powers, roots, and matrix multiplication. Collections are vector objects,
not ndarray subclasses. Use components for unsupported NumPy operations;
writable ufunc `out` is unsupported.

## Interchange

```{py:method} lvec.LVec.to_numpy()
Return an independent structured NumPy array for regular, complete data.
Jagged or missing collections cannot be represented by this export.
```

```{py:method} lvec.LVec.to_ak_record(*, with_behavior=False)
Return Awkward coordinate records. The default is plain records;
`with_behavior=True` attaches local LVec vector behavior.
```

```{py:classmethod} lvec.LVec.from_ak_record(records)
Return an owned snapshot of vector coordinate records. Extra record fields
are ignored; coordinate fields determine the imported vector components.
```

| Method | Output |
| --- | --- |
| `to_p4()` | Tuple `(px, py, pz, E)` |
| `to_ptepm()` | Tuple `(pt, eta, phi, mass)` |
| `to_np()`, `to_ak()` | Component dictionaries in the requested backend |
| `to_root_dict()` | NumPy components keyed `fX`, `fY`, `fZ`, `fE` for external I/O |

## Native Awkward records

`lvec.awkward.zip(arrays, depth_limit=None)` and
`lvec.awkward.Array(data, **kwargs)` return native records with vector
behavior. Cartesian float64 construction can share input buffers;
coordinate or dtype conversion may allocate. Native records retain ordinary
Awkward view semantics and calculate properties without a persistent cache.

Use native records inside larger event structures and with `ak.sum`,
`ak.count`, `ak.count_nonzero`, `ak.concatenate`, or `ak.cartesian`.
Convert with `LVec.from_ak_record` when an immutable snapshot is required.
Call `lvec.register_awkward()` to restore behavior for serialized `LVec2D`,
`LVec3D`, or `LVec4D` records.

## Property caches

Owned vectors calculate observables lazily. Reusing a collection can avoid
repeated calculations; native records do not retain these properties.
`LVec` accepts the cache options documented under [constructors](constructors.md).
Spatial constructors accept `cache_intermediates` and `cache_stats`.

`max_cache_size` counts property entries, not bytes. `None` is unlimited,
zero disables retention, and bounded caches evict least recently used entries.
`cache_intermediates=True` retains additional arrays. New arithmetic results
use default cache settings. Cached arrays remain read-only.

```{py:method} lvec.LVec.clear_cache()
Discard cached properties. Available on spatial vector classes as well.
```

```{py:method} lvec.LVec.set_ttl(property_name, ttl_seconds)
Set an LVec property's lifetime in seconds on a monotonic clock.
`None` disables expiration. `clear_expired()` removes expired entries.
```

`cache_stats`, `cache_hit_ratio`, and `reset_cache_stats()` are available on
all vector classes; `cache_size` is available on `LVec`. `touch()` invalidates
dependent entries but does not permit component mutation. Compare cached
and uncached calls separately when [measuring performance](../benchmarks.md).
