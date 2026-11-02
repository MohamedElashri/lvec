# Boosts, rotations, and frames

All operations return new vectors. Parameters may be scalars or arrays
compatible with the collection. Energies, momenta, and reference masses
must use consistent units; angles are radians.

## Lorentz boosts

```{py:method} lvec.LVec.boost(bx=None, by=None, bz=None, *, booster=None)
:no-index:
Apply an active boost with dimensionless velocity $\boldsymbol\beta$.
A particle at rest gains momentum along the velocity. Components must
be finite with $\beta^2<1$.

Supply either all three velocity components or one `booster`: a 3D velocity
vector or a future-directed timelike four-momentum. A single positional
booster is also accepted. Do not mix booster and component forms.
```

```{py:method} lvec.LVec.boost_p4(p4)
:no-index:
Apply an active boost defined by reference four-momentum `p4`, using its
proper velocity without converting to $\boldsymbol\beta$.

:param p4: Finite, future-directed timelike `LVec`, scalar or collection.
:returns: Boosted four-momenta with the input dimension and collection structure.
```

A zero, lightlike, spacelike, or nonpositive-energy reference raises
`InputError`. This reference domain is stricter than the domain of a
mathematical four-vector. `boost_beta3(beta3)` accepts a spatial 3D velocity.

```{py:method} lvec.LVec.boostX(beta=None, gamma=None)
:no-index:
Boost along $x$. Supply exactly one of `beta` or signed `gamma`.
The signed parameter must be finite with $\lvert\gamma\rvert\ge1$;
its sign selects the boost direction.
```

`boostY` and `boostZ` apply the same interface on the other axes.
`boostz(bz)` is a velocity-only alias. Prefer `gamma` or a reference momentum
when a velocity representation would round to the speed of light.

```{py:method} lvec.LVec.boost_to_rest(reference=None)
:no-index:
Express the input in the reference's rest frame. If `reference` is `None`,
use `self`. The reference has the same physical requirements as `boost_p4`.
```

`boostCM_of_p4(p4)` is the explicit four-momentum form;
`boostCM_of_beta3(beta3)` uses a velocity, and `boostCM_of(booster)` dispatches
by dimension. These methods include the inverse-boost sign.
`p.boost_p4(p)` is an active boost away from rest, not a CM transformation.
Negating all four components is not the inverse-reference operation.

## Named reference frames

```{py:class} lvec.Frame(bx, by, bz, name=None)
:no-index:
Store a scalar or array velocity relative to the lab, with an optional label.
Velocities must be finite and subluminal. Properties `beta_x`, `beta_y`,
`beta_z`, and `gamma` describe the frame.
```

`Frame.rest(name='lab')` constructs the lab frame.
`Frame.from_lvec(total_lvec, name='cm')` uses a total momentum, and
`Frame.center_of_mass(lvec_list, name='cm')` first sums its input vectors.

```{py:method} lvec.LVec.to_frame(frame)
:no-index:
Express lab-frame input in `frame`, applying the negative frame velocity.
```

```{py:method} lvec.LVec.transform_frame(current_frame, target_frame)
:no-index:
Apply $+\boldsymbol\beta_{\mathrm{current}}$, followed by
$-\boldsymbol\beta_{\mathrm{target}}$. This specifies a sequence of canonical
boosts; non-collinear boosts can include a spatial rotation.
```

## Spatial rotations

`rotateX(angle)`, `rotateY(angle)`, and `rotateZ(angle)` are active,
right-handed rotations. They leave energy unchanged. `rotx`, `roty`, and
`rotz` are aliases; `rotate(angle, axis='z')` selects a Cartesian axis.
For 2D vectors, `rotate(angle)` rotates the transverse plane.

```{py:method} lvec.LVec.rotate_axis(axis, angle)
:no-index:
Rotate about a nonzero spatial direction. `axis` is a 3D vector or the
spatial part of a 4D vector. The direction is normalized; scalar or
per-particle axes and angles are accepted.
```

```{py:method} lvec.LVec.rotate_euler(phi, theta, psi, order='zxz')
:no-index:
Apply the ROOT Euler convention: for an order `abc`, the matrix is
$R_a(-\psi)R_b(-\theta)R_c(-\phi)$, with the rightmost rotation acting first.
All twelve proper Euler and Tait–Bryan orders are supported.
```

`rotate_nautical(yaw, pitch, roll)` is equivalent to
`rotate_euler(roll, pitch, yaw, order='zyx')`.

```{py:method} lvec.LVec.rotate_quaternion(u, i, j, k)
:no-index:
Apply a scalar-first quaternion $(u,i,j,k)$. Unit norm is required to
preserve spatial lengths; non-unit inputs are not silently normalized.
Parameters may vary per particle.
```

## Matrix transformations

```{py:method} lvec.LVec.transform4D(obj)
:no-index:
Apply a linear transformation in component order $(x,y,z,t)$.
Supply a mapping with keys `xx`, `xy`, through `tt`, or an array of shape
$(\ldots,4,4)$. Coefficients may be scalar or per particle.
```

`transform2D` and `transform3D` act on the spatial components of those
dimensions, leaving higher components unchanged. General matrices need not
preserve a Lorentz invariant or a Euclidean norm. Check the intended physical
identity when supplying a matrix.

## Decay geometry

```{py:method} lvec.LVec.helicity_angle(parent, reference)
:no-index:
Return the helicity angle in $[0,\pi]$. Here `self` is the daughter,
`parent` is its resonance, and `reference` is the resonance's mother.
Supply all three in the same input frame.

In the resonance rest frame, the helicity axis is opposite the mother's
momentum. A zero direction gives NaN. The angle is invariant under a common
Lorentz transformation of the three momenta.
```

`opening_angle(other)` instead measures the spatial angle in the current
frame. For complete conservation and rest-frame examples, see the
[rest-frame tutorial](../tutorials/rest-frames.md).
