# Numerical accuracy and physical domains

All stored components and calculated outputs use float64. Scaled magnitudes,
compensated or exact accumulation, and selective Decimal recovery address
detected cancellation and range risks. They do not recover information
already rounded away in the supplied components.

## Invariants near cancellation

An ordinary evaluation of $E^2-p_x^2-p_y^2-p_z^2$ can overflow or cancel.
LVec detects risky cases and uses a more accurate accumulation path.

```{doctest}
>>> from lvec import LVec
>>> p = LVec(1e200, 1., 0., 1e200)
>>> float(p.mass2)
-1.0
```

A squared quantity can exceed float64 range while its square root is still
representable. Read the observable required by the calculation directly.

```{doctest}
>>> import math
>>> rest = LVec(0., 0., 0., 1e200)
>>> math.isinf(rest.mass2), float(rest.mass)
(True, 1e+200)
```

These recovery paths are selective, not a guarantee of correctly rounded
results for every combination of inputs. Transforms are not guaranteed to
be exactly reversible after floating-point rounding. Reference tests use
independent analytic, Fraction, Decimal, and matrix calculations.

## Physical validation and classification

Particle validation requires finite components, $E\ge0$, and

$$
p^2\ge-\mathrm{atol}-\mathrm{rtol}\max(E^2,\lvert\boldsymbol p\rvert^2).
$$

The relative tolerance is dimensionless; absolute tolerance has energy-squared
units. Defaults allow rounding near a lightlike boundary. Validation does
not clip the signed invariant, and zero tolerances request strict classification.

```{doctest}
>>> near_null = LVec(1., 0., 0., 1. - 1e-14)
>>> bool(near_null.mass2 < 0), bool(near_null.is_physical())
(True, True)
>>> bool(near_null.is_physical(rtol=0., atol=0.))
False
```

Causal predicates have a separate invariant-squared tolerance. Direction
predicates instead compare direction cosine. A tolerance chosen for one
kind of predicate cannot be transferred to the other without considering
its units and definition.

## Boost references and derived velocities

An active boost parameter must define a finite, subluminal velocity. A
reference four-momentum must be finite, future-directed, and timelike.
Lightlike vectors have no rest frame. Zero energy can make `to_beta3`
produce NaN or infinite component ratios; a spacelike momentum can produce
a velocity magnitude greater than one. The ratio conversion does not itself
validate a boost.

Near the light cone, conversion to velocity may round to unit magnitude and
derived Lorentz factors may lose accuracy or overflow. Use `boost_p4` or
`boost_to_rest` when the reference momentum is available. Signed-gamma axis
boosts avoid a velocity conversion when gamma is the supplied quantity.

## Missing values, NaN, and ownership

Missing Awkward entries describe absent particles or events. NaN is a
floating-point value describing an undefined or invalid numerical result.
These meanings are distinct. Physical validation ignores missing entries;
it rejects visible nonfinite momenta. Transform-parameter validation ignores
masked and unused Awkward storage.

Owned collections snapshot numerical and structural buffers, preventing
external mutation from invalidating a cached observable. Native records use
view semantics and do not retain property caches. For measured costs of
these boundaries and of recovery checks, see [benchmarks](../benchmarks.md).
