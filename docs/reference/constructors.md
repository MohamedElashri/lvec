# Constructors and coordinates

Choose a constructor from the data you already have. All public LVec
collections own immutable Cartesian float64 components. Scalars broadcast;
non-scalar components must have matching shapes and missing positions.

| Input | Constructor | Result |
| --- | --- | --- |
| Cartesian four-momenta | `LVec` | Four-vector or collection |
| Other coordinates | `obj` | Dimension selected from the keywords |
| NumPy columns or structured data | `array` | Owned vector collection |
| Jagged columns needing broadcasting | `zip` | Owned vector collection |
| Awkward-compatible records | `Array` | Owned vector collection |
| Native Awkward views | `lvec.awkward.zip`, `lvec.awkward.Array` | Records with vector behavior |

## Cartesian vectors

```{py:class} lvec.LVec(px, py, pz, E, max_cache_size=None, default_ttl=None, *, validate=False, cache_intermediates=False, cache_stats=True)
:no-index:
Construct four-momenta in component order $(p_x,p_y,p_z,E)$.

:param px: Transverse momentum along $x$.
:param py: Transverse momentum along $y$.
:param pz: Longitudinal momentum.
:param E: Energy. All components accept real scalars, lists, NumPy arrays, or Awkward arrays in consistent units.
:param validate: Check physical particle momenta after construction; default `False` permits general mathematical four-vectors.
:param max_cache_size: Maximum retained property entries; `None` is unlimited and zero disables retention.
:param default_ttl: Cache lifetime in seconds on a monotonic clock; `None` disables expiration.
:param cache_intermediates: Retain intermediate arrays as well as final observables; default `False`.
:param cache_stats: Count cache hits and misses; default `True`.
```

`Vector4D` and `Momentum4D` are aliases of `LVec`. The metric is $(+,-,-,-)$
with energy written first in four-vector notation, even though the Python
constructor takes energy last. `LVec.from_p4` accepts the same components and
cache options. For general coordinate factories, use `obj` below.

```{py:class} lvec.Vector2D(x, y, *, cache_intermediates=False, cache_stats=True)
:no-index:
Construct a spatial vector in the transverse plane. `Momentum2D` is an alias.
```

```{py:class} lvec.Vector3D(x, y, z, *, cache_intermediates=False, cache_stats=True)
:no-index:
Construct a spatial three-vector. `Momentum3D` is an alias.
```

All three classes accept scalar and collection inputs. Invalid component
shapes raise `ShapeError`; nonnumeric or complex components are rejected.
The physical validation domain and tolerances are described under
[particle validation](vectors.md#particle-validation).

## Coordinate factories

```{py:function} lvec.obj(**coordinates)
Construct a 2D, 3D, or 4D vector from coordinate keywords.

:param coordinates: One complete transverse coordinate pair, optionally one longitudinal coordinate, then optionally one temporal coordinate.
:returns: `Vector2D`, `Vector3D`, or `LVec`, with owned Cartesian storage.
```

| Part | Coordinate choices | Momentum aliases |
| --- | --- | --- |
| Transverse | `x`, `y` or `rho`, `phi` | `px`, `py`, `pt` |
| Longitudinal | `z`, `theta`, or `eta` | `pz`, `pseudorapidity` |
| Temporal | `t` or `tau` | `E`, `e`, `energy`; `mass`, `m`, `M` |

This gives two planar, six spatial, and twelve Lorentz coordinate families.
Angles are radians. Unknown keys, duplicate aliases, incomplete coordinate
pairs, or multiple choices for one coordinate raise `TypeError`.

Negative `mass` or `tau` denotes a spacelike signed invariant. The legacy
`LVec.from_ptepm(pt, eta, phi, m, ...)` instead squares its mass argument;
with `validate=True` it rejects negative transverse momentum or mass.

```{py:function} lvec.array(data, *args, **kwargs)
Construct an owned collection from a coordinate mapping or a structured
NumPy array. `arr` is an alias.

NumPy constructor arguments are accepted for non-mapping input. Coordinate
mappings do not accept additional ndarray-constructor options.
```

```{py:function} lvec.zip(arrays, depth_limit=None)
Broadcast Awkward coordinate columns and return an owned collection.

:param arrays: Mapping of coordinate names to scalars or Awkward-compatible arrays.
:param depth_limit: Broadcasting depth passed to `ak.zip`; `None` uses Awkward's default.
```

```{py:function} lvec.Array(data, **kwargs)
Read coordinate fields from Awkward-compatible records and return an owned
collection. Extra keyword arguments go to `ak.Array`. `awk` is an alias.
```

## Projections and named factories

Each vector class has `from_coordinates(**coordinates)` and factories such
as `Vector3D.from_rhophieta` or `LVec.from_xyztau`. Named factories accept
coordinates in the order written in the name.

`to_2D()`, `to_3D(**coordinates)`, and `to_4D(**coordinates)` explicitly select
the algebra's dimension. Dimension-only embeddings add Cartesian zeros.
Named conversions such as `to_xyztau()` instead add zero in the named
coordinate: $\tau=0$ gives a lightlike embedding, while $t=0$ generally does not.
`like(other)` projects or embeds to the other vector's dimension.

LVec retains Cartesian storage after every conversion; these methods do not
select a different calculation backend. See [migration](../vector-migration.md)
for differences from vector's coordinate dispatch.
