# Physics analysis helpers

All components must use the same momentum/energy units; natural units `c = 1`
are assumed. Angles are in radians. The Minkowski metric is `(+,-,-,-)`.

## Kinematics and matching

```python
import numpy as np
from lvec import LVec

tracks = LVec.from_ptepm(
    pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
    phi=np.array([3.13, -3.13]), m=.13957,
)
reference = LVec.from_ptepm(pt=4., eta=.1, phi=-3.14, m=.49367)

tracks.pt2                 # px**2 + py**2
tracks.p2                  # px**2 + py**2 + pz**2
tracks.dot(reference)      # E1*E2 - p1.p2, element by element
tracks.rapidity            # y = atanh(pz/E), not pseudorapidity eta
tracks.beta                # |p|/E in units of c
tracks.gamma               # E/sqrt(E**2-|p|**2)
tracks.delta_phi(reference) # signed phi_tracks - phi_reference in [-pi, pi)
tracks.delta_r2(reference)  # (delta_eta)**2 + (delta_phi)**2
tracks.delta_r(reference)   # sqrt(delta_r2), conventional eta-phi matching
tracks.delta_r(reference, rapidity=True)  # use delta_y instead of delta_eta
```

The rapidity version of `delta_r` is invariant under a common longitudinal boost
of physical four-vectors. The eta version generally is not, for massive objects.
Squared quantities avoid square roots when applying cuts.

Rapidity uses a difference of logarithms to retain precision near the light cone.
Its beam-axis lightlike limits are signed infinity. Rapidity is NaN for the zero
four-vector or when `abs(pz) > abs(E)`. `beta` is NaN when `E <= 0`, and can exceed
one for a spacelike mathematical vector. `gamma` is infinite for positive-energy
lightlike vectors and NaN for spacelike or nonpositive-energy vectors.
Undefined differences such as `inf - inf` in matching produce NaN.
Missing Awkward values propagate through these quantities.

## Broadcasting and event sums

Python/NumPy scalars and zero-dimensional NumPy arrays broadcast to the structure
of array components. Non-scalar arrays must match shapes, particle counts, and
missing-value structure. In particular, a length-one component array is not a
scalar: it cannot silently expand to a longer collection. This rule prevents
accidentally associating different particles' components.

```python
import awkward as ak  # Install lvec[awkward] when using the optional backend.

px = ak.Array([[1., 2.], [], [3., None]])
particles = LVec(px=px, py=0., pz=0., E=5.)
event_totals = particles.sum(axis=-1)
# px: [3., 0., 3.], E: [10., 0., 5.]
all_events_total = particles.sum()  # axis=None reduces all collection dimensions
retained_particle_axis = particles.sum(axis=-1, keepdims=True)
```

Summation combines components, then derives quantities such as invariant mass
from the resulting four-vector. It does not add individual masses. Empty
collections yield zero four-vectors. Missing particles are skipped consistently
across the four components; missing events remain missing when reducing their
particle axis. NumPy arrays follow `numpy.sum` axis/keepdims conventions.

## Awkward records

```python
records = particles.to_ak_record()  # jagged records with px, py, pz, E fields
restored = LVec.from_ak_record(records)
```

`to_ak_record()` produces an `ak.Record` for a scalar vector and an `ak.Array`
for arrays. Missing particles become missing records. Existing `to_ak()` still
returns a dictionary of component arrays, so existing callers remain compatible.
The importer requires all four named fields and ignores additional fields.

## Rest frames and decay angles

```python
parent = LVec(px=1., py=2., pz=3., E=8.)
parent_rest = parent.boost_to_rest()  # zero spatial momentum
track_rest = reference.boost_to_rest(parent)
angle = reference.opening_angle(tracks)  # angle in the current frame
```

A rest frame requires a finite, future timelike reference: `E > 0` and
`E**2 - |p|**2 > 0`. Invalid present entries raise `InputError`, including
lightlike vectors (which have no rest frame). Missing entries remain missing.

`opening_angle(other)` returns the spatial angle in `[0, pi]`. It depends on the
chosen frame. Either vector having zero spatial momentum makes its direction
undefined, giving NaN.

For a decay chain `P -> R + ...`, `R -> d + ...`, calculate:

```python
theta = daughter.helicity_angle(parent=R, reference=P)
cos_theta = np.cos(theta)
```

The three vectors must be expressed in the same input frame. Both the daughter
and reference are boosted to `R`'s rest frame. The helicity axis points opposite
to `P`'s momentum in that frame; the method measures the daughter's angle to
that axis, in radians `[0, pi]`. This sign convention matches the parent's flight
direction in the mother's rest frame. Supplying the mother explicitly removes
an ambiguity that would otherwise depend on the chosen laboratory frame.
An undefined daughter or reference direction yields NaN. This angle is invariant
under a common Lorentz transformation of all three input four-vectors.

This convention is used in the
[Belle II helicity-angle definition](https://software.belle2.org/release-05-01-25/doxygen/classBelle2_1_1Particle.html).
For comparison of rapidity and eta separation conventions, see the
[Scikit-HEP Vector interface](https://vector.readthedocs.io/en/stable/src/vector4d.html).
