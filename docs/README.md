# LVec API and usage guide

Start with the [quick start](../README.md#quick-start). The
[migration guide](migration.md) covers compatibility changes in the unreleased
update, and the [analysis guide](analysis_features.md) develops event sums,
matching, and decay-angle conventions.

## Installation

Python 3.10+ and NumPy 2.2.4+ are required. Optional Awkward support requires
Awkward 2.7.4+.

| Install | Includes |
| --- | --- |
| `pip install lvec` | NumPy vector calculations |
| `pip install "lvec[awkward]"` | Jagged arrays and Awkward records |
| `pip install "lvec[test]"` | pytest |
| `pip install "lvec[examples]"` | Awkward, uproot, matplotlib |
| `pip install "lvec[benchmark]"` | Comparison, plotting, and memory benchmark dependencies |
| `pip install -e ".[dev]"` | All development extras from a checkout |

`requirements.txt` pins the minimum runtime versions for compatibility checks;
install the package extras to obtain the corresponding tools. The repository's
unreleased changes are available from a checkout before publication to PyPI.

## Construction and component ownership

```python
import numpy as np
from lvec import LVec, Frame, Vector2D, Vector3D

v = LVec(px=1., py=2., pz=3., E=5.)
by_coordinates = LVec.from_ptepm(pt=2., eta=.3, phi=.4, m=.13957)
from_components = LVec.from_p4(1., 2., 3., 5.)
from_mapping = LVec.from_ary({"px": 1., "py": 2., "pz": 3., "E": 5.})
```

`LVec`, `from_p4`, and `from_ptepm` accept `max_cache_size=None`,
`default_ttl=None`, and keyword-only `validate=False`,
`cache_intermediates=False`, `cache_stats=True`.
`from_vec(obj)` reads the object's `px`, `py`, `pz`, and `E` attributes.
`from_ary` and `from_vec` use default cache policies.

Construction copies components into float64 snapshots and marks their NumPy
buffers read-only, including NumPy buffers inside Awkward arrays. Caller input
mutation cannot alter existing vectors or stale their caches. Ordinary in-place
component edits are rejected; construct a new vector for changed components.
Manipulating private attributes, replacing low-level Awkward layouts, or forcing
buffer writeability is outside the supported API. Complex components are rejected.

Python/NumPy scalars and zero-dimensional NumPy arrays broadcast to the collection.
Non-scalar component arrays must match shape, particle multiplicity, and missing
positions. A length-one array is not a scalar. Arrays may mix regular and variable
Awkward layouts when their logical structures match. Invalid shapes raise
`ShapeError`; unavailable optional backends raise `DependencyError` from
`lvec.exceptions`.

## Observables and arithmetic

Use the same units for momentum, energy, and mass, with `c=1` and metric
`(+,-,-,-)`.

| API | Meaning |
| --- | --- |
| `px`, `py`, `pz`, `E` | Read-only Cartesian components |
| `pt`, `pt2` | Transverse momentum and its square |
| `p`, `p2` | Spatial momentum magnitude and its square |
| `phi` | Azimuthal angle in radians |
| `eta` | Pseudorapidity `asinh(pz/pt)` |
| `mass2` | Signed invariant `E²-|p|²`, without clipping |
| `mass` | `sign(mass2)*sqrt(abs(mass2))` |
| `rapidity` | Longitudinal rapidity; distinct from eta for massive particles |
| `beta`, `gamma` | Particle speed and Lorentz factor with domain rules below |
| `dot(other)` | Elementwise Minkowski scalar product |
| `delta_phi(other)` | Signed azimuthal difference in `[-pi, pi)` |
| `delta_r2(other)`, `delta_r(other)` | Squared separation and separation in eta-phi |
| `delta_r(other, rapidity=True)` | Separation using rapidity instead of eta |

`eta` is signed infinity on the beam axis and zero for a zero spatial vector.
`beta` is NaN for nonpositive energy and can exceed one for spacelike vectors.
`gamma` is infinite for positive-energy lightlike vectors and NaN for spacelike
or nonpositive-energy vectors. Undefined coordinate/direction differences produce
NaN. See the [analysis guide](analysis_features.md#kinematics-and-matching) for
rapidity limits and missing-value behavior.

Addition, subtraction, and scalar multiplication return new vectors. Negative
energy and spacelike mathematical vectors are supported. Indexing or boolean
selection applies the same index to all components:

```python
tracks = LVec.from_ptepm(np.array([2., 3.]), .2, .4, .13957)
selected = tracks[tracks.pt > 2.5]
pair = tracks[0] + tracks[1]
```

## Particle validation

```python
particle = LVec(1., 2., 3., 5., validate=True)
mask = particle.is_physical(rtol=1e-12, atol=0.)
particle.validate_particle(rtol=1e-12, atol=0.)  # returns self, or raises InputError
```

Physical particles require finite components, `E >= 0`, and nonnegative mass
squared within tolerance. Relative tolerance scales with `max(E², |p|²)`;
`atol` has units of energy squared. Validation does not change the raw invariant.
Missing entries propagate through the mask and are skipped by aggregate
validation. `from_ptepm(..., validate=True)` also rejects negative pt or mass.
Use `rtol=0, atol=0` for strict classification; the sign of negative squared
invariants is retained even if they round to negative zero.

## Rotations and reference frames

```python
v = LVec(1., 2., 3., 5.)
rotated = v.rotate(np.pi/2, axis="y")
rotated_x = v.rotx(.2)
rotated_y = v.roty(.3)
rotated_z = v.rotz(.4)
boosted = v.boost(.1, .2, 0.)
boosted_z = v.boostz(.2)

lab = Frame.rest(name="lab")
cm = Frame.from_lvec(v, name="cm")
in_cm = v.to_frame(cm)
from_lab_to_cm = v.transform_frame(lab, cm)
back_in_lab = in_cm.transform_frame(cm, lab)
```

Angles are radians and rotation axes are the strings `"x"`, `"y"`, or `"z"`.
`boost(bx, by, bz)` is active: boosting a rest particle gives momentum along beta.
Velocities must be finite and subluminal. A `Frame` stores physical velocity
relative to the lab, so `Frame.from_lvec(v)` uses `+p/E`. `to_frame(frame)` assumes
lab input and applies `-beta_frame`. `transform_frame(source, target)` first
applies `+beta_source`, then `-beta_target`. It follows these canonical boost
conventions; boost/frame operations do not erase floating-point conditioning.

`Frame(bx, by, bz, name=None)` also accepts arrays. `Frame.center_of_mass(vectors)`
uses the total four-vector; `beta_x`, `beta_y`, `beta_z`, and `gamma` expose its
parameters. A rest frame requires a finite timelike system with nonzero energy;
`LVec.boost_to_rest(reference=None)` further requires a future timelike reference.
Omitting that reference uses the vector itself.

Opening and helicity angles are described in the
[decay-angle guide](analysis_features.md#rest-frames-and-decay-angles), including
the required common input frame, helicity-axis sign, and undefined directions.

## Arrays, reductions, and conversion

```python
import awkward as ak

particles = LVec(ak.Array([[1., 2.], [], [3., None]]), 0., 0., 5.)
events = particles.sum(axis=-1)
all_particles = particles.sum()  # axis=None
keep_axis = particles.sum(axis=-1, keepdims=True)
records = particles.to_ak_record()
restored = LVec.from_ak_record(records)
```

Sums combine four-momentum components before deriving mass; they do not sum
particle masses. Empty groups yield zero vectors. Missing particles are skipped
across all components; missing events remain missing when reducing their
particle axis. NumPy reductions support NumPy axis/keepdims conventions,
including axis tuples. Precision recovery selects affected reduction groups.

| Conversion | Output |
| --- | --- |
| `to_p4()` | `(px, py, pz, E)` tuple |
| `to_ptepm()` | `(pt, eta, phi, mass)` tuple |
| `to_np()` | Dictionary of NumPy component arrays; irregular jagged arrays cannot become rectangular NumPy arrays |
| `to_ak()` | Dictionary of Awkward component arrays; requires the optional backend |
| `to_ak_record()` | Awkward records with `px`, `py`, `pz`, `E`; scalar input yields an `ak.Record` |
| `from_ak_record(records)` | LVec from the four required fields; additional fields are ignored |
| `to_root_dict()` | NumPy arrays keyed `fX`, `fY`, `fZ`, `fE`, suitable for writing with uproot |

Reading or writing ROOT files is handled by tools such as uproot, rather than a
mandatory ROOT/PyROOT dependency. The [examples](../examples/README.md) cover that
workflow.

## Spatial vectors

`Vector2D(x, y, *, cache_intermediates=False, cache_stats=True)` provides `x`, `y`,
`r`, `phi`, arithmetic, indexing, `dot(other)`, and `rotate(angle)`.

`Vector3D(x, y, z, *, cache_intermediates=False, cache_stats=True)` additionally
provides `z`, `rho`, `theta`, `cross(other)`, and `rotx`, `roty`, `rotz`,
`rotate(angle, axis="z")`. The same float64 ownership and shape rules apply.

```python
transverse = Vector2D(3., 4.)
rotated_2d = transverse.rotate(np.pi/4)
spatial = Vector3D(1., 2., 3.)
rotated_3d = spatial.rotate(np.pi/2, axis="y")
normal = spatial.cross(Vector3D(0., 0., 1.))
```

## Cache policies

Derived results are cached lazily, and cached numeric arrays are read-only.
Intermediate squared arrays are temporary by default; enable
`cache_intermediates=True` to retain them when repeated reuse justifies memory.

```python
v = LVec(1., 2., 3., 5., max_cache_size=16, default_ttl=60.,
         cache_intermediates=False, cache_stats=False)
v.set_ttl("mass", 10.)  # may be set before the first calculation
mass = v.mass
stats = v.cache_stats
retained_bytes = stats["overall"]["retained_bytes"]
v.set_ttl("mass", None)  # disable expiry for this property
expired_count = v.clear_expired()
v.clear_cache()
uncached = LVec(1., 2., 3., 5., max_cache_size=0)
```

`max_cache_size` counts entries, not bytes. `None` is unlimited; `0` retains no
derived arrays. Bounded caches evict least recently used entries. TTLs use a
monotonic clock, and configured property policies survive eviction, expiration,
and cache invalidation. `cache_stats=False` disables hit/miss counting while
retained-byte and cache-size information remain available. Use `cache_hit_ratio`,
`cache_size`, and `reset_cache_stats()` for inspection; `touch()` remains a
compatibility invalidation hook and does not make private component edits safe.

Factory cache options are explicit. Newly created arithmetic/transformation
results use their default cache settings rather than inheriting a parent's TTL
or retention limit.

## Numerical accuracy

The [accuracy guide](numerical_accuracy.md) explains ordinary vectorized paths,
selective compensated/exact recovery, float64 limits, and measured detection
costs. Recovery is selective within a batch and preserves missing-value layouts.
There is no promise of exact rounding or reversible boosts for every input.

## Validation and benchmarks

From a checkout with test extras installed:

```bash
python -m pytest -W error::RuntimeWarning
python -m pytest lvec/tests/test_stress_physics.py -W error
python -m pytest lvec/tests/test_extreme_invariants.py lvec/tests/test_adaptive_precision.py -W error
```

CI checks Python 3.10 with minimum NumPy/Awkward versions, current dependencies
on Python 3.11–3.14, NumPy-only installation, and wheel/source distributions.
The current local suites have 324 passing tests with both minimum and current
runtime versions; the NumPy-only suite has 212 passing and 84 skipped tests.
Counts reflect the recorded environment and can grow as tests are added.

```bash
PYTHONPATH=. python benchmarks/benchmark_core.py --sizes 1000 1000000 --jagged-events 10000
PYTHONPATH=. python benchmarks/stress_lvec.py --sizes 1000 100000 --repeats 5
PYTHONPATH=. python benchmarks/adaptive_precision.py --before /path/to/baseline --after .
```

Benchmarks use seeded data and separate cold calculations, cached reads, complete
workflows, retained buffers, traced allocations, and process memory where
applicable. Compare the same script and environment across checkouts; cached
reads do not measure arithmetic throughput. Timings are not portable CI gates.
See [recorded reports and reproduction details](numerical_accuracy.md).
