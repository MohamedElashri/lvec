# Physics guide

[Overview](README.md) · [API reference](docs/api-reference.md) · [Migration](docs/vector-migration.md)

[Momenta](#constructing-particle-momenta) · [Mass and selections](#invariant-mass-and-selections) ·
[Events](#jagged-events-and-reductions) · [Frames](#boosts-and-reference-frames) ·
[Decay angles](#rotations-and-decay-angles) · [Physical domains](#physical-domains-and-numerical-accuracy)

LVec uses natural units with $c=1$, Cartesian four-momenta

$$
p^\mu=(E,p_x,p_y,p_z),\qquad p^2=E^2-p_x^2-p_y^2-p_z^2,
$$

and metric $(+,-,-,-)$. **Constructor and export order is `(px, py, pz, E)`.**
Choose one consistent unit for momentum, energy, and mass; angles are radians.

## Constructing particle momenta

Use `LVec` for Cartesian components and `obj` for other coordinates, such as
transverse momentum, pseudorapidity, azimuth, and mass; both accept scalars and arrays:

```python
import numpy as np
import lvec

cartesian = lvec.LVec(px=1., py=2., pz=3., E=5., validate=True)
tracks = lvec.obj(
    pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
    phi=np.array([.4, -.5]), mass=.13957,
)
print(np.round(tracks.mass, 5))  # [0.13957 0.13957]
```

`obj` selects 2D, 3D, or 4D from the supplied coordinates. All combinations of
`(x,y)` or `(rho,phi)`, then `z`/`theta`/`eta`, then `t`/`tau` are supported;
momentum aliases include `pt`, `pz`, `E`, and `mass`. See the
[constructor reference](docs/api-reference.md#construction-and-coordinates).

Owned collections copy inputs into read-only float64 snapshots. Scalars
broadcast; non-scalar components must have matching shapes and missing positions.
A length-one array is an array, not a scalar. Use `lvec.zip` for explicit
Awkward event-to-particle broadcasting.

## Invariant mass and selections

For a reconstructed parent, add daughter four-momenta **before** calculating
mass: $m_{12}^2=(p_1+p_2)^2$. The sum of daughter masses is a different quantity.

```python
import numpy as np
import lvec

tracks = lvec.obj(
    pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
    phi=np.array([.4, -.5]), mass=.13957,
)
selected = tracks[tracks.pt > 2.5]
parent = tracks[0] + tracks[1]
print(selected.pt)  # [3.]
print(f"Pair mass: {parent.mass:.3f}")  # Pair mass: 2.481
```

| Observable | Definition |
| --- | --- |
| `pt`, `p` | Transverse and spatial momentum magnitudes |
| `mass2` | Signed $p^2$, without clipping |
| `mass` | $\mathrm{sgn}(p^2)\sqrt{\lvert p^2\rvert}$ |
| `eta` | Pseudorapidity $\mathrm{arsinh}(p_z/p_T)$ |
| `rapidity` | Longitudinal rapidity $y=\frac12\log[(E+p_z)/(E-p_z)]$ for physical momenta |
| `dot(q)` or `p @ q` | Minkowski product $E_pE_q-\boldsymbol p\cdot\boldsymbol q$ |
| `Et`, `Mt2` | $E\sin\theta$ and $E^2-p_z^2$; `Mt` is the signed square root |

`Mt` is a single-vector transverse mass, not the two-object missing-energy
transverse mass used in some collider analyses.

`delta_phi(q)` is signed $\phi_{\mathrm{self}}-\phi_q$, wrapped to $[-\pi,\pi)$.
`delta_r2(q)` calculates $(\Delta\eta)^2+(\Delta\phi)^2$; use the squared
distance for matching cuts. `delta_r(q, rapidity=True)` substitutes rapidity,
whose separation is invariant under a common longitudinal boost of physical
momenta. Pseudorapidity separation generally is not for massive particles.

## Jagged events and reductions

Install `lvec[awkward]` to work with variable-length particle collections.

```python
import awkward as ak
from lvec import LVec

particles = LVec(ak.Array([[1., 2.], [], None, [3., None]]), 0., 0., 5.)
totals = particles.sum(axis=-1)
print(ak.to_list(totals.px))  # [3.0, 0.0, None, 3.0]
print(ak.to_list(totals.E))   # [10.0, 0.0, None, 5.0]
```

`axis=-1` reduces particles within each event; `sum()` reduces all collection
dimensions. Empty groups give zero vectors, missing particles are skipped as
whole vectors, and missing events remain missing when reducing their particle
axis. `keepdims=True` retains the reduced axis. NumPy arrays also accept axis
tuples. Form event totals from selected particles before reading their masses.

For vectors inside larger Awkward records, use
`particles.to_ak_record(with_behavior=True)` or the native factories in
`lvec.awkward`. Native records have ordinary view semantics and no property
cache; Cartesian float64 factories can share external numerical buffers.
Convert with `LVec.from_ak_record(records)` for an immutable owned collection.
See [array interfaces](docs/api-reference.md#arrays-and-interchange).

## Boosts and reference frames

`boost(bx, by, bz)` is **active**: a particle at rest gains momentum along
$\boldsymbol\beta$. With $\gamma=(1-\beta^2)^{-1/2}$,

$$
E'=\gamma(E+\boldsymbol\beta\cdot\boldsymbol p),\qquad
\boldsymbol p'=\boldsymbol p+
\left[\frac{\gamma^2}{\gamma+1}(\boldsymbol\beta\cdot\boldsymbol p)
+\gamma E\right]\boldsymbol\beta.
$$

To express a daughter in its parent's rest frame, use `boost_to_rest(parent)`.
A reference must be finite, future-directed, and timelike; a lightlike system
has no rest frame.

```python
from lvec import Frame, LVec

parent = LVec(1., 2., 3., 8.)
daughter = LVec(1., 0., 0., 2.)
cm = Frame.from_lvec(parent)
daughter_cm = daughter.to_frame(cm)
print(daughter_cm.allclose(daughter.boost_to_rest(parent)))  # True
print(f"CM momentum: {parent.to_frame(cm).p:.3f}")  # CM momentum: 0.000
```

A `Frame` stores velocity relative to the lab. `to_frame(frame)` assumes lab
input and applies the **negative** frame velocity. `transform_frame(source,
target)` applies $+\boldsymbol\beta_{\mathrm{source}}$, then
$-\boldsymbol\beta_{\mathrm{target}}$, defining a sequence of
canonical boosts; non-collinear boosts can include a spatial rotation.

Use `boost_p4(reference)` or `boostX(gamma=...)` when momentum or $\gamma$ is
already available, avoiding conversion of an ultrarelativistic reference to
$\boldsymbol\beta$. Velocity inputs must be finite with $\beta^2<1$;
signed $\gamma$ requires $\lvert\gamma\rvert\ge1$, with its sign selecting
the boost direction.

## Rotations and decay angles

`rotateX/Y/Z` and `rotate_axis(axis, angle)` are active, right-handed rotations
and leave energy unchanged. `rotate_euler(phi, theta, psi)` defaults to `zxz`.
An order `abc` denotes ROOT's convention $R_a(-\psi)R_b(-\theta)R_c(-\phi)$,
with the rightmost rotation acting first. All 12 Euler/Tait–Bryan orders are
supported. Quaternions are scalar-first and require unit norm to preserve
lengths; general matrices need not preserve any physical invariant.

For $P\to R+\cdots$, $R\to d+\cdots$, the helicity axis is opposite the
mother's momentum in the resonance rest frame:

```python
import numpy as np
from lvec import LVec

mother = LVec(0., 0., 0., 10.)
resonance = LVec(0., 0., 3., 5.)
daughter_rest = LVec(1., 0., np.sqrt(3.), 2.)
daughter = daughter_rest.boost(0., 0., .6)
theta = daughter.helicity_angle(parent=resonance, reference=mother)
print(f"Helicity angle: {theta:.3f} rad")  # Helicity angle: 0.524 rad
```

Supply all three vectors in the same input frame. The helicity angle is in
$[0,\pi]$ and invariant under a common Lorentz transformation. `opening_angle(q)`
is the spatial angle in the current frame; a zero direction gives NaN.

## Physical domains and numerical accuracy

General four-vectors may be spacelike or have negative energy. For particle
data, `LVec(..., validate=True)` or `validate_particle(rtol=1e-12, atol=0.)`
requires finite components, $E\ge0$, and nonnegative invariant within tolerance.
Relative tolerance scales with $\max(E^2,\lvert\boldsymbol p\rvert^2)$;
absolute tolerance has units of energy squared. `is_physical(...)` returns a
mask. Validation preserves the raw
invariant; use zero tolerances for strict classification. Missing entries are
excluded from aggregate validation.

Negative `mass`/`tau` in coordinate factories denotes a spacelike signed
invariant. If its magnitude exceeds the spatial momentum, no real energy exists:
the factory sets the energy to zero, as vector does, and issues
`lvec.CoordinateWarning`. The legacy `LVec.from_ptepm(..., m=...)` squares its mass argument;
`validate=True` rejects negative $p_T$ or mass there.

| Limit | Result |
| --- | --- |
| Zero spatial momentum | $\eta=0$; opening/helicity angles with a zero direction are NaN |
| Nonzero beam-axis momentum | `eta` is signed infinity |
| Zero four-vector | `beta`, `gamma`, and `rapidity` are NaN |
| Positive-energy lightlike vector | $\gamma=\infty$; beam-axis rapidity is signed infinity |
| Spacelike vector | Signed negative mass; `gamma` is NaN and `beta` may exceed one |
| Nonpositive energy | `beta` and `gamma` are NaN |
| $\lvert p_z\rvert>\lvert E\rvert$ | `rapidity` is NaN |
| Zero / nonzero null vector normalization | Zero remains zero; a nonzero null four-vector has no finite unit normalization |

Calculations use float64. Scaled magnitudes, compensated or exact accumulation,
and selective Decimal recovery address detected cancellation and range risks.
Boost recovery also covers subnormal reference masses and proper velocities;
parameter validation ignores masked and unused Awkward storage.
Masses, scalar products, and $\gamma$ use ordinary float64 evaluation by
default. `with lvec.invariant_policy('accurate'):` recovers cancelling and
range-extreme invariants. Signed-mass recovery then activates for $m/E<1/64$,
so light particles usually take the slower compensated path. Use it when the
Cartesian components are the data or near the float64 range limits, not when
the supplied components already limit the attainable accuracy.
These checks cannot recover precision lost in inputs or guarantee correct
rounding or reversible transformations for every input. A squared observable
may overflow or underflow even when its square root is representable. NaN and
missing values have distinct meanings; missing positions retain their array structure.

Reuse owned collections to benefit from cached observables. Cache limits and
native view boundaries are described in the [API reference](docs/api-reference.md);
[benchmarks](benchmarks/README.md) separate cached reads from fresh calculations.
Independent Fraction/Decimal references, Lorentz invariants, and decay
conservation are checked in the [test suite](lvec/tests).

## Further examples

[Jagged events](examples/07_jagged_arrays.py),
[reference frames](examples/10_reference_frames.py), and
[spatial vectors](examples/06_3d_vectors.py) run on generated data.
ROOT I/O uses optional uproot rather than a required ROOT dependency; see
[reading branches](examples/01_basic_reading.py) and the
[fixture instructions](CONTRIBUTING.md#examples).
