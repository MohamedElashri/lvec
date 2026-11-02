# API reference

[Overview](../README.md) · [Physics guide](../user-guide.md) · [Migration](vector-migration.md)

All vector classes use owned Cartesian float64 components. Operations apply
elementwise to scalars, NumPy arrays, and optional Awkward collections.
The [physics guide](../user-guide.md) defines units, boost signs, signed masses,
missing-value behavior, and numerical limits.

## Construction and coordinates

| Interface | Use |
| --- | --- |
| `LVec(px, py, pz, E, ...)` | Four-momentum; `Vector4D` and `Momentum4D` are aliases |
| `Vector2D(x, y)`, `Vector3D(x, y, z)` | Spatial vectors; also `Momentum2D`, `Momentum3D` |
| `obj(**coordinates)` | Select dimension and coordinate family from keywords |
| `array(columns)` / `arr` | Coordinate mapping or structured NumPy array → owned collection |
| `zip(columns, depth_limit=None)` | Explicit Awkward broadcasting → owned collection |
| `Array(records)` / `awk` | Coordinate records → owned collection |
| `LVec.from_p4(px, py, pz, E)` | Cartesian constructor |
| `LVec.from_ptepm(pt, eta, phi, m)` | Legacy transverse-coordinate constructor |
| `LVec.from_ary(mapping)`, `LVec.from_vec(obj)` | Import four named components |
| `from_ak_record(records)` | Import applicable coordinate components; extra fields are ignored |

Coordinate factories combine `(x,y)` or `(rho,phi)`, optionally one of
`z`/`theta`/`eta`, then `t`/`tau` for 4D. Momentum synonyms are `px`, `py`, `pt`,
`pz`, `pseudorapidity`, `E`/`e`/`energy`, and `mass`/`m`/`M`. These give
2 planar, 6 spatial, and 12 Lorentz coordinate families. Constructors reject
complex and nonnumeric components.

Classes also expose `from_coordinates(**coordinates)` and named factories such
as `Vector3D.from_rhophieta(...)` and `LVec.from_xyztau(...)`.

## Observables

Properties are available in applicable dimensions; aliases share calculations.

| Quantity | Names |
| --- | --- |
| Cartesian | `x`, `y`, `z`, `t`; momentum aliases `px`, `py`, `pz`, `E`, `e`, `energy` |
| Transverse | `rho`, `rho2`, `pt`, `pt2`, `phi` |
| Spatial | `mag`, `mag2`, `p`, `p2`, `theta`, `eta`, `pseudorapidity`, `costheta`, `cottheta` |
| Signed invariant | `tau`, `tau2`, `mass`, `mass2`, `m`, `m2`, `M`, `M2` |
| Kinematic | `beta`, `gamma`, `rapidity` |
| Transverse energy | `Et`, `Et2`, `et`, `et2`, `transverse_energy`, `transverse_energy2` |
| Transverse mass | `Mt`, `Mt2`, `mt`, `mt2`, `transverse_mass`, `transverse_mass2` |
| Temporal square | `t2`, `E2`, `e2`, `energy2` |

`Vector2D.r` is the planar magnitude; `Vector3D.r` is the full spatial magnitude.

## Algebra and geometry

| Operation | Interface |
| --- | --- |
| Arithmetic | `+`, `-`, scalar `*` and `/`; `add`, `subtract`, `scale` |
| Scalar product | `dot(q)`, `p @ q`; Minkowski in 4D, Euclidean in 2D/3D |
| Norm and normalization | `abs(p)` gives signed mass in 4D and Euclidean magnitude in 2D/3D; `unit()` |
| Partial scaling | `scale2D/3D/4D(factor)`, `neg2D/3D/4D()` |
| Equality | `==`, `!=`, `equal(q)`, `not_equal(q)`; one mask entry per vector |
| Approximate equality | `isclose(q, rtol=1e-5, atol=1e-8, equal_nan=False)`, `allclose(...)` |
| Spatial products and angles | `cross(q)` → 3D, `deltaangle(q)`, `opening_angle(q)` on LVec |
| Angular separation | `delta_phi` / `deltaphi`, `deltaeta`, `delta_r` / `deltaR`, `delta_r2` / `deltaR2` |
| Rapidity separation | LVec `delta_r(q, rapidity=True)`, `deltaRapidityPhi(q)`, and squared counterparts |
| Direction tests | `is_parallel`, `is_antiparallel`, `is_perpendicular` with `tolerance=1e-5` on direction cosine |
| Causal tests | `is_timelike(tolerance=0)`, `is_spacelike(tolerance=0)`, `is_lightlike(tolerance=1e-5)`; tolerance has invariant-squared units |
| Particle tests | LVec `is_physical(rtol=1e-12, atol=0)`, `validate_particle(...)` |

Binary algebra and equality require the same dimension. Spatial geometry may
use the spatial part of a 4D vector. Project explicitly with `to_2D()`,
`to_3D()`, `to_4D(**coordinates)`, or `like(other)`.

Named `to_<coordinates>` methods mirror the coordinate families, including
momentum spellings such as `to_ptphietamass()`. Storage remains Cartesian.
Dimension-only embeddings add Cartesian zeros; named embeddings add zero in
the requested coordinate. Thus `to_4D()` adds `t=0`, while `to_xyztau()` adds
`tau=0`; `to_3D(z=0)` specifies a planar embedding unambiguously.

## Transformations

| Operation | Interface |
| --- | --- |
| Axis rotation | `rotateX/Y/Z(angle)`; `rotx/y/z`, `rotate(angle, axis='z')` |
| General rotation | `rotate_axis(axis, angle)`, `rotate_euler(phi, theta, psi, order='zxz')` |
| Other conventions | `rotate_nautical(yaw, pitch, roll)`, `rotate_quaternion(u, i, j, k)` |
| Matrix transform | `transform2D/3D/4D(matrix)`; mappings keyed `xx`, `xy`, … or arrays shaped `(..., n, n)` |
| Active beta boost | `boost(bx, by, bz)`, `boost_beta3(beta3)`, `boostz(beta)` |
| Active reference boost | `boost_p4(p4)`, or `boost(booster)` dispatching 3D beta / 4D momentum |
| Axis boost | `boostX/Y/Z(beta=...)` or `boostX/Y/Z(gamma=...)`; exactly one parameter |
| CM boost | `boostCM_of_beta3`, `boostCM_of_p4`, `boostCM_of`; `boost_to_rest(reference=None)` |
| Velocity vector | `to_beta3()` |
| Named frame | `to_frame(frame)`, `transform_frame(source, target)` |
| Decay geometry | `helicity_angle(parent, reference)` |

`Frame(bx, by, bz, name=None)` accepts scalar or array velocities.
`Frame.rest()`, `Frame.from_lvec(total)`, and `Frame.center_of_mass(vectors)`
construct common frames. Parameters are `beta_x`, `beta_y`, `beta_z`, and `gamma`.
See [frame and angle conventions](../user-guide.md#boosts-and-reference-frames).
For 2D rotations, use `Vector2D.rotate(angle)` without an axis argument.

## Arrays and interchange

Indexing and masks select complete vectors. `sum(axis=None, keepdims=False)`
combines components; `shape`, `ndim`, and `reshape` describe regular arrays.
NumPy supports `sum`, `count_nonzero`, `stack`, `concatenate`, `broadcast_arrays`,
`reshape`, `isclose`/`allclose`, arithmetic, absolute, powers, roots, and matmul.
Writable ufunc `out` and reduction `dtype`, `out`, `initial`, `where` are unsupported.

| Conversion | Result |
| --- | --- |
| `to_p4()`, `to_ptepm()` | LVec tuples `(px,py,pz,E)` and `(pt,eta,phi,mass)` |
| `to_numpy()` | Independent structured NumPy export for regular, complete data |
| `to_np()`, `to_ak()` | LVec component dictionaries in the requested backend |
| `to_ak_record(with_behavior=False)` | Plain coordinate records; `True` attaches local vector behavior |
| `to_root_dict()` | LVec NumPy components keyed `fX`, `fY`, `fZ`, `fE` for external I/O tools |

Awkward functions such as `ak.sum`, `ak.count`, `ak.count_nonzero`,
`ak.concatenate`, and `ak.cartesian` work through named vector records.
`lvec.awkward.Array` and `lvec.awkward.zip` return native views without property
caches; coordinate/dtype conversions can allocate. Owned `lvec.Array` and
`lvec.zip` snapshot inputs. Call `lvec.register_awkward()` to restore behavior
for serialized `LVec2D`, `LVec3D`, or `LVec4D` records.

## Cache options and errors

`LVec`, `from_p4`, and `from_ptepm` accept `max_cache_size=None`,
`default_ttl=None`, `validate=False`, `cache_intermediates=False`, and
`cache_stats=True`. Spatial constructors accept the two `cache_*` options.

`max_cache_size` counts derived-property entries: `None` is unlimited; `0`
disables retention. Bounded caches evict least recently used entries. TTLs are
seconds on a monotonic clock; `set_ttl(name, None)` disables expiration.
`cache_intermediates=True` retains additional arrays. Fresh arithmetic results
use default cache settings. Cached numerical arrays are read-only.

| Controls | Availability |
| --- | --- |
| `clear_cache()`, `cache_stats`, `cache_hit_ratio`, `reset_cache_stats()` | All vector classes |
| `set_ttl(name, seconds)`, `clear_expired()`, `cache_size` | LVec |
| `touch()` | Invalidates dependent entries; does not authorize component mutation |

Shape mismatches raise `lvec.ShapeError`. `InputError`, `BackendError`, and
`DependencyError` are defined in `lvec.exceptions`. Invalid transformations
can raise `ValueError`; unsupported operations raise `TypeError`.
