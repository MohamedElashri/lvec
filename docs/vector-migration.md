# Migrating from vector to LVec

LVec provides familiar numerical vector operations while storing owned,
Cartesian float64 components and caching derived properties. Start by changing
construction and checking the policies below. The [coverage matrix](vector-feature-parity.md)
accounts for the mathematical API; the [benchmark report](../benchmarks/README.md)
identifies the workloads for which performance has been measured.

## Construction and calculations

```python
import lvec

p = lvec.obj(pt=45., eta=.7, phi=1.2, mass=.105)
q = lvec.obj(px=10., py=5., pz=3., E=15.)
print((p + q).mass, p.deltaR(q), p @ q)

spatial = p.to_3D()
rotated = p.rotate_axis(lvec.Vector3D(1., 2., 3.), .2)
rest = p.boostCM_of_p4(p + q)
```

`obj` accepts Cartesian or polar azimuthal coordinates, z/theta/eta, and
energy/time or signed mass/proper-time. Momentum synonyms are supported.
`Vector2D`, `Vector3D`, and `LVec` / `Vector4D` remain direct Cartesian
constructors. Use `obj` or a named `from_<coordinates>` factory for other inputs.
Momentum class aliases share the same numerical classes; they do not introduce
a separate coordinate-keyword constructor.

For NumPy columns or structured arrays:

```python
import numpy as np
import lvec

p = lvec.array({"px": np.array([1., 2., 3.]), "py": 0., "pz": 0., "E": 5.})
selected = p[p.pt > 1.]
total = np.sum(selected)
assert selected.shape == (2,)
assert total == lvec.LVec(5., 0., 0., 10.)

structured = p.to_numpy()  # Independent structured NumPy export.
rebuilt = lvec.array(structured)
assert rebuilt.allclose(p)
```

`array` / `arr` returns an owned vector collection, rather than an ndarray
subclass. Use its components for ndarray APIs outside the supported vector
protocol. Vector-aware arithmetic, comparisons, reductions, stacking,
concatenation, reshaping, and broadcasting are available directly. Components
are immutable snapshots; mutable structured exports are independent copies.

## Jagged collections and native records

The default `zip` factory returns an owned, cached collection and explicitly
permits event-to-particle broadcasting. `Array` / `awk` accepts coordinate
records and returns the corresponding owned collection.

```python
import awkward as ak
import lvec

p = lvec.zip({
    "pt": ak.Array([[10., 20.], [], [30.]]),
    "phi": 0., "eta": .5, "mass": .1,
})
masses = p.mass                  # Cached on this collection.
event_totals = p.sum(axis=-1)     # Owned collection result.
pairs = ak.cartesian({"a": p, "b": p}, axis=1)
pair_masses = (pairs.a + pairs.b).mass
```

When vectors need to remain native records inside a larger Awkward structure,
attach local behavior explicitly:

```python
records = p.to_ak_record(with_behavior=True)
events = ak.zip({"particles": records}, depth_limit=1)
angles = events.particles.deltaR(events.particles)
totals = ak.sum(events.particles, axis=-1)

# Factories returning native records rather than owned collection objects:
from lvec.awkward import zip as record_zip
native = record_zip({"pt": ak.Array([[1., 2.], []]), "phi": 0.})
assert ak.to_list(native.pt) == [[1., 2.], []]
```

Native records have ordinary Awkward view semantics. Their properties calculate
through transient numerical views without retaining a property cache. Repeated
analysis of the same columns can benefit from constructing an owned LVec
collection once. `to_ak_record()` without the keyword continues to export plain
records. `ak.sum` on an owned collection delegates through named native records;
use `.sum` when an owned collection result is preferred.

For Cartesian float64 columns, `lvec.awkward.zip` and `lvec.awkward.Array`
attach behavior directly and normally share the numerical buffers. Changes to
those external buffers are visible to subsequent calculations. Coordinate or
dtype conversion can allocate new buffers. Use `lvec.zip`, `lvec.Array`, or
`LVec.from_ak_record` when an independent, immutable snapshot is required.

The record names are `LVec2D`, `LVec3D`, and `LVec4D`. Call
`lvec.register_awkward()` before reading serialized named records without local
behavior. It registers these names without replacing existing vector behaviors.
Convert existing vector momentum records with `LVec.from_ak_record(records)`;
geometric x/y/z/t records are accepted too.

## Scientific policies

| Topic | Migration decision |
| --- | --- |
| Storage and dtype | Cartesian float64 snapshots. LVec does not retain the supplied polar representation or structured field dtypes. |
| Cached values | Reuse a collection to benefit from property caching. `clear_cache()` forces recalculation; instrumentation and intermediate retention are configurable. |
| Component shapes | Direct constructors require equal non-scalar shapes and matching missing masks; scalars expand. Use `lvec.zip` for explicit event-to-particle broadcasting or broadcast components first. |
| Missing particles | A missing vector is missing in every component. `zip` moves coordinate options outside the record. Sums skip a missing particle as a whole, and preserve missing events when reducing their particle axis. Vector records with only some fields missing can produce different sums; normalize those records before migration. |
| Equality | `==` and `!=` now compare components, producing a scalar or array mask. Use `isclose` / `allclose` for floating-point comparisons. |
| Dimensions | Binary algebra rejects mixed dimensions. Project or embed explicitly with `to_2D`, `to_3D`, `to_4D`, or `like`. Spatial geometry may use the spatial parts of 4D inputs. |
| Signed mass input | `obj(..., mass=-m)` represents a spacelike signed invariant, following vector. Legacy `LVec.from_ptepm(..., m=-m)` retains its unsigned-mass calculation; `validate=True` rejects negative particle mass. |
| Negative energy and spacelike gamma | Mathematical four-vectors remain accepted, but LVec's `beta` and `gamma` are NaN for nonpositive energy; gamma is also NaN for spacelike inputs. Vector's algebraic values differ in those cases. |
| Zero and beam limits | Zero-vector eta is 0; zero four-vector beta/gamma/rapidity are NaN. Beam-axis photon rapidity is signed infinity. Undefined opening angles are NaN. |
| Particle versus causal predicates | `is_physical` checks finite, future-directed particle momenta and a configurable invariant tolerance. `is_timelike`, `is_spacelike`, and `is_lightlike` classify the invariant; they are separate operations. |
| Tolerances | Comparison tolerances must be finite and nonnegative. Causal tolerance has invariant-squared units. At zero causal tolerance, exact null vectors are lightlike; this intentionally differs from vector 1.9.0's strict-zero tolerance boundary. |
| Unit vectors | Euclidean normalization handles overflow/underflow in magnitudes. Zero vectors remain zero. Nonzero null 4-vectors cannot have finite unit normalization; nonzero components become signed infinity and zero components remain zero. NaN input is not converted into a valid zero vector. |
| Boost domains | Beta boosts require finite speed below c. p4 boost references must be finite, future-directed and timelike. Signed-gamma axis boosts require finite absolute gamma at least one. |
| Active and passive transforms | Boosts are active. `to_frame(Frame(...))` uses the negative frame velocity. Euler signs and orders follow ROOT/vector; ordinary axis rotations are active and right-handed. |
| Quaternions and matrices | Quaternions are scalar-first and not automatically normalized. General matrices can change norms. Only rotations and Lorentz matrices have the corresponding invariant-preservation contract. |
| Coordinate embeddings | Dimension-only methods add Cartesian zeros. Named representation methods add zeros in the named coordinate, as vector does: adding tau=0 differs from adding t=0. Cartesian storage preserves existing energy signs during representation conversions. |
| Mutation and NumPy options | No mutable component assignment, writable ufunc `out`, or reduction `dtype`/`out`/`initial`/`where`. Create a new collection from changed components. Reductions support axis and keepdims. |

These policies preserve existing LVec behavior while making the common physical
analysis domain familiar. Selected cancellation and range-risk cases receive
compensated, exact, or Decimal recovery. Float64 input rounding and conditioning
still limit every calculation; there is no universal correct-rounding guarantee.

## Choosing a performance path

Construct the collection once when its inputs are reused. Read derived
properties from that collection so their cached values can be reused. Original
`LVec(px,py,pz,E)`, Cartesian arithmetic, axis rotations, and component-beta
boosts continue to use their established kernels. Familiar names delegate to
those kernels where the calculation is identical.

Shared scalar rotation parameters avoid broadcasting coefficient arrays.
Euler and arbitrary-axis rotations keep up to 128 immutable scalar coefficient
sets per rotation family; changing orientations beyond that bound exercises
fresh coefficient evaluation. Per-particle parameters use fused numerical
leaves, while large scalar transforms and boosts use bounded blocks. Neither
path retains mutable parameter arrays. Both keep numerical risk detection and
selective recovery; the benchmark includes repeated orientations, 257 varying
orientations, and per-particle parameters separately.

Use p4 or signed-gamma boosts when those are the parameters available. They avoid
losing information by converting an ultrarelativistic reference to beta. Use
native records when nested Awkward behavior is needed; measure the complete
workflow, including conversion and ownership costs. Inspect the benchmark's
fresh/cached boundaries before comparing a timing to another package.

Numba compilation, symbolic SymPy expressions, PyTree registration, vector's
coordinate container classes, and mutable ndarray-subclass behavior are outside
this numerical core. Code using those interfaces needs a separate integration
rather than an import rename.
