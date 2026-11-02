# Observables and vector algebra

An observable is a scalar for one vector, a NumPy array for a dense collection,
or an Awkward array with the collection's missing structure. Momentum and
geometric aliases refer to the same calculation. The [API index](../api-reference.md)
lists all spellings.

## Momentum and invariant mass

```{py:property} lvec.LVec.mass2
:no-index:
The signed Minkowski norm $p^2=E^2-p_x^2-p_y^2-p_z^2$, without clipping.
```

```{py:property} lvec.LVec.mass
:no-index:
The signed mass $\mathrm{sgn}(p^2)\sqrt{\lvert p^2\rvert}$. A spacelike
four-vector therefore has negative mass. Aliases include `m`, `M`, and `tau`.
```

```{py:property} lvec.LVec.pt
:no-index:
The transverse magnitude $p_T=\sqrt{p_x^2+p_y^2}$; `rho` is an alias.
```

| Observable | Definition |
| --- | --- |
| `p`, `mag` | $\lvert\boldsymbol p\rvert$ |
| `eta` | $\mathrm{arsinh}(p_z/p_T)$ for nonzero transverse momentum |
| `phi` | Azimuthal angle about the longitudinal axis |
| `rapidity` | $y=\frac12\log[(E+p_z)/(E-p_z)]$ for physical momenta |
| `beta` | $\lvert\boldsymbol p\rvert/E$ for positive energy |
| `gamma` | $E/m$ for positive-energy timelike momenta |
| `Et` | $E\sin\theta$ |
| `Mt2`, `Mt` | $E^2-p_z^2$ and its signed square root |

`Mt` is a single-vector transverse mass. It is not the two-object
missing-energy transverse mass used in some collider analyses. Consult the
[physical-domain table](../physics.md#physical-domains-and-numerical-accuracy)
for zeros, null vectors, spacelike inputs, infinities, and NaN results.

## Arithmetic and comparison

`+`, `-`, scalar multiplication, and scalar division act elementwise.
`add`, `subtract`, and `scale` provide method forms. Binary algebra and
equality require the same dimension; project explicitly before combining
spatial vectors and four-vectors.

```{py:method} lvec.LVec.dot(other)
:no-index:
Return the Minkowski product $E_pE_q-\boldsymbol p\cdot\boldsymbol q$.
The operator `p @ q` is equivalent. Spatial classes use the Euclidean product.
```

```{py:method} lvec.LVec.unit()
:no-index:
Return a normalized vector. Timelike and spacelike four-vectors are scaled
by the magnitude of their signed mass, preserving the invariant's sign.
Zero remains zero; a nonzero null four-vector has no finite unit normalization:
its nonzero components become signed infinities and its zero components stay
zero. Spatial classes normalize by their Euclidean magnitude.
```

`abs(p)` gives signed mass in 4D and Euclidean magnitude in 2D/3D.
`scale2D`, `scale3D`, and `scale4D` scale only the named components;
`neg2D`, `neg3D`, and `neg4D` are corresponding negation properties.

```{py:method} lvec.LVec.isclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)
:no-index:
Compare corresponding components and return one Boolean entry per vector.
`==` and `equal` instead require exact component equality.
```

```{py:method} lvec.LVec.allclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)
:no-index:
Return a single Boolean requiring all visible component comparisons to pass.
Missing entries must have matching structure. Use explicit tolerances
appropriate to the momentum scale of the analysis.
```

## Angular separation

```{py:method} lvec.LVec.delta_phi(other)
:no-index:
Return $\phi_{\mathrm{self}}-\phi_{\mathrm{other}}$, wrapped to $[-\pi,\pi)$.
`deltaphi` is an alias.
```

```{py:method} lvec.LVec.delta_r(other, *, rapidity=False)
:no-index:
Return $\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}$. With `rapidity=True`,
substitute longitudinal rapidity for pseudorapidity. `delta_r2` avoids
the square root, for example in matching cuts.
```

`deltaRapidityPhi` and `deltaRapidityPhi2` are named rapidity variants.
`deltaR` and `deltaR2` use pseudorapidity. A rapidity difference is invariant
under a common longitudinal boost of physical momenta; a pseudorapidity
difference generally is not for massive particles.

`cross` returns a spatial 3D vector. `deltaangle` and `opening_angle` measure
spatial angles in the current frame. Direction predicates `is_parallel`,
`is_antiparallel`, and `is_perpendicular` use a direction-cosine tolerance,
defaulting to $10^{-5}$.

## Particle validation

```{py:method} lvec.LVec.is_physical(rtol=1e-12, atol=0.0)
:no-index:
Return a mask for finite momenta with $E\ge0$ and nonnegative invariant
within tolerance. Relative tolerance scales with
$\max(E^2,\lvert\boldsymbol p\rvert^2)$; absolute tolerance has units of
energy squared. Tolerances must be finite and nonnegative.
```

```{py:method} lvec.LVec.validate_particle(rtol=1e-12, atol=0.0)
:no-index:
Apply the same particle check and return `self`. Raise `InputError` if any
visible momentum fails. Missing Awkward entries are excluded from the check.
`LVec(..., validate=True)` applies the default tolerances during construction.
```

Validation leaves the original signed invariant intact. Zero tolerances
request strict classification. General four-vectors may legitimately have
negative energy or a spacelike invariant, so validation is opt-in.

`is_timelike(tolerance=0)`, `is_spacelike(tolerance=0)`, and
`is_lightlike(tolerance=1e-5)` classify the signed invariant; their tolerance
has invariant-squared units, unlike the direction predicates.

## Errors

```{py:exception} lvec.ShapeError
:no-index:
Non-scalar component shapes or missing structures do not agree.
```

`lvec.exceptions.InputError` reports invalid physical parameters.
`BackendError` wraps failures during backend operations, including some
constructor errors. Unsupported operations raise `TypeError`;
invalid transformation parameters can raise `ValueError`.
