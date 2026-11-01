# LVec user guide

LVec provides Lorentz four-vectors and two- and three-dimensional spatial
vectors for physics analysis. Calculations use NumPy or optional Awkward arrays.
Angles are radians; momentum, energy, and mass must use consistent units with
`c=1` and the Minkowski metric `(+,-,-,-)`.

## Contents

- [Installation](#installation)
- [Constructing vectors](#constructing-vectors)
- [Observables and arithmetic](#observables-and-arithmetic)
- [Particle validation](#particle-validation)
- [Angular matching](#angular-matching)
- [Rotations and reference frames](#rotations-and-reference-frames)
- [Rest frames and decay angles](#rest-frames-and-decay-angles)
- [Event collections and reductions](#event-collections-and-reductions)
- [Records and conversion](#records-and-conversion)
- [Spatial vectors](#spatial-vectors)
- [Cache controls](#cache-controls)
- [Numerical accuracy](#numerical-accuracy)
- [Examples](#examples)
- [Development and benchmarks](#development-and-benchmarks)
- [Workflow dependencies](#workflow-dependencies)
- [Building and releasing](#building-and-releasing)
- [Citation](#citation)

## Installation

NumPy is the only mandatory runtime dependency. Awkward support is optional.
Supported Python versions and dependency constraints are declared in
[pyproject.toml](pyproject.toml).

| Install | Purpose |
| --- | --- |
| `pip install lvec` | NumPy calculations |
| `pip install "lvec[awkward]"` | Jagged collections and Awkward records |
| `pip install "lvec[test]"` | pytest |
| `pip install "lvec[examples]"` | Awkward, uproot, and matplotlib examples |
| `pip install "lvec[benchmark]"` | Comparison, plotting, and memory benchmark tools |
| `pip install -e ".[dev]"` | All development extras from a checkout |

Install a checkout to work with a release candidate before its publication.
`requirements.txt` pins the minimum NumPy/Awkward versions for compatibility
checks; it is not the full development environment.

## Constructing vectors

```python
import numpy as np
from lvec import LVec, Frame, Vector2D, Vector3D

v = LVec(px=1., py=2., pz=3., E=5.)
coordinates = LVec.from_ptepm(pt=2., eta=.3, phi=.4, m=.13957)
cartesian = LVec.from_p4(1., 2., 3., 5.)
mapping = LVec.from_ary({"px": 1., "py": 2., "pz": 3., "E": 5.})
```

`LVec`, `from_p4`, and `from_ptepm` accept `max_cache_size=None`,
`default_ttl=None`, and keyword-only `validate=False`,
`cache_intermediates=False`, `cache_stats=True`. `from_vec(obj)` reads the
object's `px`, `py`, `pz`, and `E` attributes. `from_vec` and `from_ary` use default
cache policies.

Construction copies real components into float64 snapshots and marks their
NumPy buffers read-only, including buffers inside Awkward arrays. Mutating the
original input cannot change a vector or invalidate a cached observable.
Construct a new vector to change its components. Complex and nonnumeric
components are rejected. Private attribute changes, low-level layout replacement,
and forcing buffer writeability are outside the supported API.

Python/NumPy scalars and zero-dimensional NumPy arrays broadcast to a collection.
Non-scalar arrays must match in shape, particle counts, and missing-value
positions. A length-one array is not a scalar. Regular and variable Awkward
layouts may be combined when their logical structures match.

Shape mismatches raise `ShapeError`, available from `lvec`. Other errors,
including `InputError`, `BackendError`, and `DependencyError`, are available
from `lvec.exceptions`.

## Observables and arithmetic

| API | Meaning |
| --- | --- |
| `px`, `py`, `pz`, `E` | Read-only Cartesian components |
| `pt`, `pt2` | Transverse momentum and its square |
| `p`, `p2` | Spatial momentum magnitude and its square |
| `phi` | Azimuthal angle in radians |
| `eta` | Pseudorapidity `asinh(pz/pt)` |
| `mass2` | Signed invariant `E²-|p|²`, without clipping |
| `mass` | Signed square root of the invariant |
| `rapidity` | Longitudinal rapidity |
| `beta`, `gamma` | Particle speed and Lorentz factor |
| `dot(other)` | Elementwise Minkowski scalar product |

Spacelike vectors have negative `mass2` and negative signed `mass`. Mathematical
four-vectors may have negative energy. Addition, subtraction, and scalar
multiplication return new vectors; indexing applies to all four components:

```python
tracks = LVec.from_ptepm(np.array([2., 3.]), .2, .4, .13957)
selected = tracks[tracks.pt > 2.5]
pair = tracks[0] + tracks[1]
scaled = 2. * pair
difference = pair - tracks[0]
metric_product = tracks[0].dot(tracks[1])
```

`eta` is signed infinity on the beam axis and zero for zero spatial momentum.
Rapidity uses a difference of logarithms near the light cone. Its beam-axis
lightlike limits are signed infinity; it is NaN for the zero four-vector or
when `abs(pz) > abs(E)`.

`beta` is NaN for `E <= 0` and can exceed one for a spacelike vector. `gamma` is
infinite for positive-energy lightlike vectors and NaN for spacelike or
nonpositive-energy vectors. Missing values propagate. Undefined expressions
such as differences of identical infinite coordinates produce NaN.

## Particle validation

```python
particle = LVec(1., 2., 3., 5., validate=True)
physical = particle.is_physical(rtol=1e-12, atol=0.)
particle.validate_particle(rtol=1e-12, atol=0.)
```

A physical particle requires finite components, `E >= 0`, and nonnegative mass
squared within tolerance. Relative tolerance scales with `max(E², |p|²)`;
absolute tolerance has units of energy squared. Validation never clips or changes
the raw invariant. `validate_particle` returns the vector or raises `InputError`.
Missing entries remain missing in the mask and are skipped in aggregate checks.

Use `rtol=0, atol=0` for strict classification. The sign of a negative squared
invariant is retained even when the result rounds to negative zero.
`from_ptepm(..., validate=True)` also rejects negative pt or mass.

## Angular matching

```python
reference = LVec.from_ptepm(pt=4., eta=.1, phi=-3.14, m=.49367)
tracks = LVec.from_ptepm(
    pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
    phi=np.array([3.13, -3.13]), m=.13957,
)
dphi = tracks.delta_phi(reference)
dr2 = tracks.delta_r2(reference)
dr = tracks.delta_r(reference)
rapidity_distance = tracks.delta_r(reference, rapidity=True)
matched = tracks[dr2 < .4**2]
```

`delta_phi` is signed `self.phi - other.phi`, wrapped to `[-pi, pi)`.
`delta_r2` is `delta_eta² + delta_phi²`, and `delta_r` is its square root.
`rapidity=True` substitutes longitudinal rapidity for eta on either method.
Squared distances avoid a square root when applying cuts.

The rapidity version is invariant under a common longitudinal boost of physical
four-vectors. Eta separation generally changes under such a boost for massive
objects. Undefined coordinate differences produce NaN.

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
also_in_cm = v.transform_frame(lab, cm)
back_in_lab = in_cm.transform_frame(cm, lab)
np.testing.assert_allclose(in_cm.to_p4(), also_in_cm.to_p4())
np.testing.assert_allclose(back_in_lab.to_p4(), v.to_p4())
```

Rotation axes are the strings `"x"`, `"y"`, or `"z"`. Rotations leave energy
unchanged. Boost velocities must be finite and strictly subluminal.

`boost(bx, by, bz)` is active: a rest particle acquires momentum along beta.
A `Frame` stores its physical velocity relative to the lab, so
`Frame.from_lvec(v)` uses `+p/E`. `to_frame(frame)` assumes lab input and applies
`-beta_frame`. `transform_frame(source, target)` applies `+beta_source`, followed
by `-beta_target`. These methods define the sequence of canonical boosts.

`Frame(bx, by, bz, name=None)` accepts scalars or arrays. Use `Frame.rest()` for a
stationary lab frame, `Frame.from_lvec(total)` for a system's rest frame, or
`Frame.center_of_mass(vectors)` to construct it from a collection of LVecs.
The properties `beta_x`, `beta_y`, `beta_z`, and `gamma` expose frame parameters.
A frame from a four-vector requires a finite timelike system with nonzero energy.

## Rest frames and decay angles

```python
parent = LVec(1., 2., 3., 8.)
parent_rest = parent.boost_to_rest()
track_rest = reference.boost_to_rest(parent)
angle = reference.opening_angle(tracks)
```

`boost_to_rest(reference=None)` uses the vector itself when the reference is
omitted. Its reference must be finite, future timelike, and satisfy `E > 0` and
`E²-|p|² > 0`. Invalid present entries raise `InputError`; missing entries remain
missing. Lightlike vectors have no rest frame.

`opening_angle(other)` is the spatial angle in `[0, pi]` in the current frame.
A zero spatial vector has no direction and produces NaN.

For a chain `P -> R + ...`, `R -> d + ...`, provide the resonance and its mother
explicitly. This example constructs a massless daughter in the resonance rest
frame and boosts it into the mother's frame:

```python
mother = LVec(0., 0., 0., 10.)
resonance = LVec(0., 0., 3., 5.)
daughter_rest = LVec(1., 0., np.sqrt(3.), 2.)
daughter = daughter_rest.boost(0., 0., .6)
theta = daughter.helicity_angle(parent=resonance, reference=mother)
cos_theta = np.cos(theta)
np.testing.assert_allclose(theta, np.pi/6, atol=1e-14)
```

All three vectors must be expressed in the same input frame. The daughter and
mother are boosted into the resonance rest frame, and the helicity axis points
opposite the mother's momentum there. The angle is in `[0, pi]` and is invariant
under a common Lorentz transformation of the inputs. An undefined daughter or
reference direction gives NaN.

## Event collections and reductions

```python
import awkward as ak  # install lvec[awkward]

particles = LVec(ak.Array([[1., 2.], [], [3., None]]), 0., 0., 5.)
events = particles.sum(axis=-1)
all_particles = particles.sum()
keep_axis = particles.sum(axis=-1, keepdims=True)
assert ak.to_list(events.px) == [3., 0., 3.]
assert ak.to_list(events.E) == [10., 0., 5.]
```

A sum combines components before deriving mass. It does not add individual
particle masses. `axis=None` reduces all collection dimensions; `axis=-1`
reduces the particle axis in a one-level jagged collection. NumPy arrays support
NumPy axis/keepdims conventions, including axis tuples.

Empty groups yield zero vectors. Missing particles are skipped consistently
across components; missing events remain missing when reducing their particle
axis. Scalar components broadcast into the same masks as array components.
Selected reduction groups receive additional numerical recovery when needed.

## Records and conversion

```python
records = particles.to_ak_record()
restored = LVec.from_ak_record(records)
components = restored.to_p4()
```

| Conversion | Output |
| --- | --- |
| `to_p4()` | `(px, py, pz, E)` tuple |
| `to_ptepm()` | `(pt, eta, phi, mass)` tuple |
| `to_np()` | Dictionary of NumPy component arrays |
| `to_ak()` | Dictionary of Awkward component arrays |
| `to_ak_record()` | Records with `px`, `py`, `pz`, `E` fields |
| `from_ak_record(records)` | LVec from the four required named fields |
| `to_root_dict()` | NumPy arrays keyed `fX`, `fY`, `fZ`, `fE` |

Awkward conversion requires the optional backend. Scalar record export yields an
`ak.Record`; collections yield an `ak.Array`, and missing particles become missing
records. Import ignores additional fields. Irregular jagged collections cannot
be converted directly into rectangular NumPy arrays.

ROOT file I/O is handled by tools such as uproot rather than a mandatory
ROOT/PyROOT dependency. `to_root_dict()` provides component arrays suitable for
writing through uproot.

## Spatial vectors

`Vector2D(x, y, *, cache_intermediates=False, cache_stats=True)` provides `x`, `y`,
`r`, `phi`, arithmetic, indexing, `dot(other)`, and `rotate(angle)`.

`Vector3D(x, y, z, *, cache_intermediates=False, cache_stats=True)` adds `z`, `rho`,
`theta`, `cross(other)`, `rotx`, `roty`, `rotz`, and `rotate(angle, axis="z")`.
The same float64 ownership and shape rules apply.

```python
transverse = Vector2D(3., 4.)
rotated_2d = transverse.rotate(np.pi/4)
spatial = Vector3D(1., 2., 3.)
rotated_3d = spatial.rotate(np.pi/2, axis="y")
normal = spatial.cross(Vector3D(0., 0., 1.))
```

## Cache controls

Derived observables are cached lazily and cached numeric arrays are read-only.
Squared-component intermediates are temporary by default; set
`cache_intermediates=True` when repeated reuse warrants retaining them.

```python
v = LVec(1., 2., 3., 5., max_cache_size=16, default_ttl=60.,
         cache_intermediates=False, cache_stats=False)
v.set_ttl("mass", 10.)
mass = v.mass
retained_bytes = v.cache_stats["overall"]["retained_bytes"]
v.set_ttl("mass", None)
expired_count = v.clear_expired()
v.clear_cache()
uncached = LVec(1., 2., 3., 5., max_cache_size=0)
```

`max_cache_size` counts entries, not bytes. `None` permits unlimited entries;
`0` retains no derived arrays. Bounded caches evict least recently used entries.
TTLs use a monotonic clock. Property policies may be set before first access and
survive eviction, expiration, and invalidation. `set_ttl(name, None)` disables
expiration for that property.

`cache_stats=False` disables hit/miss counting while cache size and retained
bytes remain inspectable. `cache_hit_ratio`, `cache_size`, and
`reset_cache_stats()` support monitoring. `touch()` clears component-dependent
cache entries; it does not make private component edits safe. New arithmetic
and transformation results use default cache settings rather than inheriting
source TTLs or retention limits.

## Numerical accuracy

Calculations use float64 inputs and outputs. Ordinary buffers use vectorized
kernels; selected entries with cancellation or range risk receive additional
arithmetic without promoting an entire batch to high precision.

| Calculation | Recovery for selected cases |
| --- | --- |
| Magnitudes and coordinate energy | Scaled/hypot evaluation to avoid intermediate square overflow/underflow |
| Invariant mass, mass squared, Minkowski products | Compensated products/sums, then exact binary accumulation for extreme or unresolved entries |
| Boosts and frame gamma | Accurate `1-beta²` near the light-speed boundary; local Decimal recovery for exceptional boosts |
| Rotations | Compensated/exact products for cancelled or nonfinite linear combinations |
| Event sums | Compensated short groups, fsum for long/extreme groups, exact binary recovery from intermediate overflow |

```python
extreme = LVec(1e200, 1., 0., 1e200)
assert extreme.mass2 == -1.
assert extreme.mass == -1.
assert extreme.dot(extreme) == -1.
```

Signed mass is rounded directly from the exact invariant in the extreme path,
so it can remain finite when mass squared itself overflows or underflows.
Opening and helicity angles use atan2-based geometry to retain small collinear
angles. Float32 angle arrays are evaluated at float64 component precision.
Awkward list structure and missing values are preserved.

The cancellation checks use a scale of roughly `sqrt(eps)` of the sum bound.
Extreme component hierarchies and near-light-speed boosts receive additional
checks. These are numerical risk triggers, not a guarantee of exact rounding or
reversible transforms for every input. Precision already lost in supplied
components cannot be recovered. An unrepresentable squared observable may be
infinite or zero even when its square root is representable.

Detection has overhead even when a precision fallback does not run. In the
recorded million-entry workload, NumPy boost/rotation/sum times were about
17.0/6.6/12.7 ms, compared with 10.9/2.6/6.4 ms without the adaptive checks.
Jagged boosts and rotations were within about 7% of that baseline; jagged sums
took about twice as long. These are single-machine measurements, not timing
guarantees. The [adaptive precision report](benchmarks/results/adaptive-precision-2026-11-01.json)
records the environment, workload, and mixed-batch recovery counts.

Compensated products/sums follow
[Ogita, Rump and Oishi (2005)](https://doi.org/10.1137/030601818).

## Examples

Install the examples extra, then run scripts from the `examples` directory.

| Script | Purpose | Input |
| --- | --- | --- |
| `01_basic_reading.py` | Read ROOT branches | Local ROOT fixture |
| `02_decay_reconstruction.py` | Combine daughter four-vectors | Local ROOT fixture |
| `03_advanced_selections.py` | Apply array selections | Local ROOT fixture |
| `04_boost_frame.py` | Boost toward a parent rest frame | Local ROOT fixture |
| `05_2d_vectors.py` | 2D magnitudes and vector operations | Generated in script |
| `06_3d_vectors.py` | 3D vector algebra | Generated in script |
| `07_jagged_arrays.py` | Variable-length collections | Generated in script |
| `08_lhcb_data.py` | ROOT data and two-body mass plots | Network download if absent |
| `09_cache_performance.py` | Inspect cached access | Generated in script |
| `10_reference_frames.py` | CM transformations | Generated in script |
| `11_cache_optimization_demo.py` | Cache policies | Generated in script |

```bash
pip install -e ".[examples]"
cd examples
python 05_2d_vectors.py
python 06_3d_vectors.py
python 07_jagged_arrays.py
python 10_reference_frames.py
```

For the ROOT fixture examples, create the input from the examples directory:

```bash
mkdir -p samples
python create_test_data.py
python 01_basic_reading.py
```

That fixture demonstrates I/O: its daughter three-momenta sum to the stored
mother's momentum, but their energies do not sum to its stored energy. It is
not an energy-conserving Z-decay reference. Physical two-body decay checks are
in [the stress suite](lvec/tests/test_stress_physics.py).

The LHCb example may download data and write plots in the current directory.
Check the dataset's component units against the script's mass/energy assumptions
before using its histograms for analysis. It is outside the offline test suite.

## Development and benchmarks

From the repository root:

```bash
pip install -e ".[awkward,test]"
python -m pytest -W error::RuntimeWarning
python -m pytest lvec/tests/test_stress_physics.py -W error
python -m pytest lvec/tests/test_extreme_invariants.py lvec/tests/test_adaptive_precision.py -W error
```

CI checks the supported Python versions with minimum and current dependencies,
NumPy-only installations, and wheel/source distributions. The current matrix is
defined in the [CI workflow](.github/workflows/Build_Package.yml).
Independent checks use Fraction, integer, and high-precision Decimal oracles;
physics stress cases cover random transformations, two-body decays, massless
vectors, nested missing data, numerical extremes, and cache lifecycle.

```bash
PYTHONPATH=. python benchmarks/benchmark_core.py --sizes 1000 1000000 --jagged-events 10000
PYTHONPATH=. python benchmarks/stress_lvec.py --sizes 1000 10000 100000 1000000 --repeats 5
PYTHONPATH=. python benchmarks/stress_lvec.py --kinds extreme --sizes 1 1000 10000
PYTHONPATH=. python benchmarks/adaptive_precision.py --before /path/to/baseline --after .
PYTHONPATH=. python benchmarks/adaptive_precision.py --probe --repeats 3
```

Benchmarks use seeded data and distinguish cold calculations, cached reads,
complete workflows, retained buffers, traced allocations, and process memory
where applicable. Compare the same script and environment across checkouts;
cached reads do not measure arithmetic throughput. Timings are not CI gates.
Missing/deep/heavy-tail stress cases are capped at 100,000 events, and explicit
extreme-cancellation batches at 10,000 entries.

Recorded data are available in the
[scaling report](benchmarks/results/stress-2026-11-01.json),
[invariant accuracy report](benchmarks/results/invariant-accuracy-2026-11-01.json),
and [adaptive report](benchmarks/results/adaptive-precision-2026-11-01.json).

## Workflow dependencies

[Sanad](https://github.com/MohamedElashri/sanad) manages the native GitHub Actions
lockfile at `.github/workflows/actions.lock`. Its policy lives in `.sanad.toml`.
The lockfile covers both CI and publishing workflows, including transitive action
dependencies, and the CI check verifies the frozen dependency graph.

After editing action references or adding a workflow, review and reconcile the
lockfile with:

```bash
sanad sync
sanad sync --write
sanad check
```

Use `sanad plan` to review available dependency updates and `sanad update --write`
to apply eligible changes. Commit the lockfile together with workflow changes;
do not edit its generated contents by hand.

Markdown-only and documentation-directory changes skip CI. A change that also
touches code, package metadata, dependencies, workflow files, or the action
lockfile runs CI normally. Release publishing is event-driven and remains
independent of these path filters.

## Building and releasing

Build and inspect distributions from a clean checkout:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The package version in `pyproject.toml` and `lvec.__version__` must agree.
Release tags use `v` followed by the package version. The release workflow checks
that the tag and package versions match, runs tests, builds both distributions,
and validates their metadata.
Publishing a GitHub release triggers the separate PyPI publishing job using
Trusted Publishing and the `pypi` environment. A draft release does not trigger
publication. See the [release workflow](.github/workflows/python-publish.yml),
[GitHub release events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release),
and [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Citation

Record `lvec.__version__` with analysis outputs when an exact software version
is needed.

```bibtex
@software{lvec,
  author = {Mohamed Elashri},
  title = {LVec: Lorentz and spatial vectors for physics analysis},
  url = {https://github.com/MohamedElashri/lvec}
}
```

LVec is distributed under the [GNU General Public License v3](LICENSE).
