# Coordinates and conventions

LVec stores Cartesian float64 components. Coordinate keywords describe an
input representation; named conversions select coordinates or a dimension
without changing the stored representation.

## Coordinate families

Choose one row from each applicable part of the following table. The temporal
part requires a longitudinal coordinate, so it constructs a 4D vector.

| Part | Keyword choices | Definition |
| --- | --- | --- |
| Transverse | `x`, `y` | Cartesian components |
| Transverse | `rho`, `phi` | $x=\rho\cos\phi$, $y=\rho\sin\phi$ |
| Longitudinal | `z` | Cartesian component |
| Longitudinal | `theta` | $z=\rho\cot\theta$ |
| Longitudinal | `eta` | $z=\rho\sinh\eta$ |
| Temporal | `t` | Cartesian temporal component |
| Temporal | `tau` | Signed invariant with $t^2=\lvert\boldsymbol x\rvert^2+\mathrm{sgn}(\tau)\tau^2$ |

There are two transverse, six spatial, and twelve Lorentz coordinate families.
Momentum synonyms are available on the same classes:

| Geometric name | Momentum names |
| --- | --- |
| `x`, `y`, `z` | `px`, `py`, `pz` |
| `rho` | `pt` |
| `t` | `E`, `e`, `energy` |
| `tau` | `mass`, `m`, `M` |
| `eta` | `pseudorapidity` |

Angles are radians. Physical transverse momentum is nonnegative, and the
usual polar angle is in $[0,\pi]$. Coordinate factories do not replace
particle validation; call `validate_particle` when interpreting their result
as physical particle momenta.

## Signed invariant coordinates

Positive mass supplies a timelike invariant and chooses nonnegative energy.
Negative mass denotes a spacelike invariant. A real temporal component
requires $\lvert\boldsymbol p\rvert^2+\mathrm{sgn}(m)m^2\ge0$.
If the requested signed invariant would require an imaginary energy, the
factory clips the energy to zero; it cannot represent that requested mass.

```{doctest}
>>> import lvec
>>> timelike = lvec.obj(px=3., py=0., pz=0., mass=4.)
>>> timelike.E, float(timelike.mass2)
(5.0, 16.0)
>>> spacelike = lvec.obj(px=3., py=0., pz=0., mass=-2.)
>>> round(float(spacelike.mass2), 12)
-4.0
```

The legacy `LVec.from_ptepm` squares its mass argument and has a different
negative-mass policy. Its entry in the [4D reference](../reference/vector4d.md)
documents that behavior explicitly.

## Projections and embeddings

`to_2D`, `to_3D`, and `to_4D` project or embed by dimension. Added Cartesian
coordinates are zero unless supplied as keywords. Named conversions instead
impute zero in the named coordinate. For example, adding $t=0$ gives a
different invariant from adding $\tau=0$.

```{doctest}
>>> spatial = lvec.Vector3D(3., 0., 0.)
>>> float(spatial.to_4D().mass2), float(spatial.to_xyztau().mass2)
(-9.0, 0.0)
>>> float(spatial.to_4D(E=5.).mass2)
16.0
```

The [dimensional interfaces](../api-reference.md) list every generated
`from_<coordinates>` and `to_<coordinates>` method, including momentum
spellings. Constructor arguments follow the order in the method name.

## Rotation conventions

Cartesian-axis rotations are active and right-handed. Euler rotations follow
the ROOT convention: order `abc` denotes

$$
R_a(-\psi)R_b(-\theta)R_c(-\phi),
$$

with the rightmost rotation applied first. This is not a sequence of positive
active rotations by the three named angles. All six proper Euler and six
Tait–Bryan orders are accepted. Nautical rotation is equivalent to
`rotate_euler(roll, pitch, yaw, order='zyx')`.

Quaternions are scalar-first. For $q=(u,i,j,k)$, unit norm preserves spatial
length; a non-unit quaternion also scales the spatial result by its squared
norm. LVec does not normalize it implicitly. General matrices have no
invariant-preservation requirement.
