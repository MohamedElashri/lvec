# Migrating to the unreleased update

These changes are pending release. The package version remains `0.1.4` until a
release is prepared; use a checkout to exercise the new behavior.

## Installation and supported versions

The supported baseline is Python 3.10+, NumPy 2.2.4+, and optional Awkward 2.7.4+.
NumPy is the only mandatory runtime dependency. Install `lvec[awkward]` for jagged
arrays and records, `lvec[examples]` for ROOT-reading/plotting examples,
`lvec[benchmark]` for benchmark tools, or `lvec[dev]` for all development extras.
If code previously relied on LVec installing unrelated analysis tools, declare
those tools explicitly or select the appropriate extra.

## Signed invariant mass and particle checks

Previously, taking the absolute value of mass squared hid spacelike vectors.
`mass2` now retains its sign and `mass` uses a signed square root:

```python
from lvec import LVec

spacelike = LVec(2., 0., 0., 1.)
assert spacelike.mass2 == -3.
assert spacelike.mass < 0.
```

Review selections that assumed mass was always nonnegative. Mathematical vector
algebra now accepts negative energy; use `validate=True`, `is_physical`, or
`validate_particle` where a vector represents a physical particle. The default
relative tolerance is `1e-12`; absolute tolerance has units of energy squared.
Checks never clip or change the raw invariant. Strict zero-tolerance checks
retain a negative invariant's sign even when it rounds to negative zero.

`eta` uses `asinh(pz/pt)`: it returns signed infinity on the beam axis and zero
for zero spatial momentum. See the [analysis guide](analysis_features.md) for
rapidity, beta/gamma, and undefined direction rules.

## Active boosts and passive frame transforms

A `Frame` stores its physical velocity relative to the lab. `Frame.from_lvec(v)`
now uses `+p/E`, correcting the earlier inconsistent sign. `boost(bx, by, bz)`
is an active boost, while `to_frame(frame)` assumes lab coordinates and applies
`-beta_frame`. A transform between frames applies `+beta_source`, followed by
`-beta_target`.

```python
import numpy as np
from lvec import Frame, LVec

parent = LVec(1., 2., 3., 8.)
lab = Frame.rest(name="lab")
cm = Frame.from_lvec(parent, name="cm")
np.testing.assert_allclose(parent.to_frame(cm).to_p4()[:3], 0., atol=1e-14)
np.testing.assert_allclose(
    parent.to_frame(cm).to_p4(), parent.transform_frame(lab, cm).to_p4(),
)
```

Remove manual sign reversals added to compensate for the old frame behavior.
Raw calls such as `particle.boost(-parent.px/parent.E, ...)` retain the active
boost convention, but `particle.boost_to_rest(parent)` expresses that intent
more clearly. Rest-frame helpers reject references with no valid rest frame.

Boost velocities must be finite and strictly subluminal. Near the light-speed
boundary, validation now checks an accurate `1-beta²` rather than rejecting a
valid velocity merely because rounded squares sum to one.

## Immutable float64 components and broadcasting

Construction snapshots external NumPy/Awkward components and promotes real
integer/float32 data to float64. Public component buffers are read-only. Code
that mutates source arrays to update an existing vector must instead construct
a new vector:

```python
import numpy as np
from lvec import LVec

px = np.array([1., 2.])
v = LVec(px, 0., 0., 5.)
px[0] = 3.
assert v.px[0] == 1.
updated = LVec(px, 0., 0., 5.)
assert updated.px[0] == 3.
```

Only scalars, including zero-dimensional NumPy arrays, broadcast to a collection.
Length-one arrays do not silently expand. Non-scalar shapes, nested particle
counts, and missing-value positions must match across components. Complex and
nonnumeric components are rejected. Low-level buffer/layout mutation is outside
the supported API; `touch()` is a compatibility cache hook, not a mutable-input
mode.

## Caches and memory

Final observables are cached by default; squared-component intermediates are
now temporary. Set `cache_intermediates=True` if repeated reuse warrants the
additional memory. `max_cache_size=0` disables retention, `None` is unlimited,
and finite limits count entries rather than bytes. Cached numeric arrays are
read-only. `cache_stats=False` disables hit/miss counting while size and retained
bytes remain inspectable.

Per-property TTL policies can be set before first access and survive eviction,
expiration, and invalidation. `set_ttl(name, None)` disables that property's
expiration. TTLs use a monotonic clock. Arithmetic/transform results use their
default cache settings; they do not inherit a source vector's TTL or cache size.

## Analysis helpers and numerical recovery

The [analysis guide](analysis_features.md) covers kinematics, matching, event
sums, records, rest-frame boosts, and decay angles. `to_ak()` still returns a
dictionary; use `to_ak_record()` for records. Sums add components before deriving
mass and consistently skip missing particles.

Severe cancellation and range extremes receive selective compensated, exact
binary, or Decimal recovery. For example:

```python
from lvec import LVec

v = LVec(1e200, 1., 0., 1e200)
assert v.mass2 == -1.
assert v.mass == -1.
assert v.dot(v) == -1.
```

The [accuracy guide](numerical_accuracy.md) documents limits, regression oracles,
and measurements. Detection has overhead even when no fallback runs; at one
million entries, recorded NumPy boost/rotation/sum costs were roughly
1.6/2.5/2.0 times the pre-adaptive baseline. Jagged boosts/rotations were within
about 7% of baseline, while jagged sums were roughly twice as costly. These are
single-machine measurements. Float64 input rounding and ill-conditioned inverse
transforms still limit recoverable precision.
