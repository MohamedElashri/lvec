# Migrating from vector

[Overview](../README.md) · [Physics conventions](../user-guide.md) · [API reference](api-reference.md)

LVec provides familiar numerical vector operations with Cartesian float64
storage, immutable owned collections, and cached observables.

```python
import numpy as np
import lvec

p = lvec.array({"pt": np.array([2., 3.]), "eta": .2, "phi": .4, "mass": .13957})
selected = p[p.pt > 2.5]
total = np.sum(selected)
rest = total.boostCM_of_p4(total)
np.testing.assert_allclose(rest.p, 0., atol=1e-14)
```

| Existing pattern | LVec |
| --- | --- |
| `vector.obj(**coordinates)` | `lvec.obj(**coordinates)` |
| `vector.array(columns)` | `lvec.array(columns)` returns an owned collection |
| `vector.zip(columns)` | `lvec.zip(columns)` returns an owned collection |
| Native Awkward vector records | `lvec.awkward.zip` / `Array`, or `to_ak_record(with_behavior=True)` |
| Existing four-vector records | `LVec.from_ak_record(records)` returns an owned snapshot |
| Coordinate changes and projections | `to_<coordinates>()`, `to_2D/3D/4D()` |
| Algebra, rotations, boosts, angular distances | Familiar method names; see the [API tables](api-reference.md) |

Owned collections copy inputs and cache properties. Native Awkward records
retain view semantics and calculate properties without a retained cache.
Choose the boundary that matches how your analysis uses and mutates its inputs.

## Check these differences

- **Shapes and missing data.** Direct constructors require matching non-scalar
  shapes and masks; use `lvec.zip` for event-to-particle broadcasting. A missing
  particle is missing in every component. Normalize partially missing records
  before comparing reductions.
- **Array behavior.** Collections are vector objects rather than ndarray
  subclasses. Use components for unsupported ndarray operations. Components
  are immutable; writable `out` and extra NumPy reduction options are unsupported.
- **Physics conventions.** Check the [physical domains](../user-guide.md#physical-domains-and-numerical-accuracy),
  especially nonpositive energy, undefined $\gamma$ for spacelike momenta,
  zero vectors, signed mass, and physical boost-reference requirements.
  At zero causal tolerance, exact null vectors are classified as lightlike.
- **Dimensions.** Mixed-dimension algebra requires explicit projection or
  embedding. Adding $t=0$ differs from adding $\tau=0$.
- **Retained vector conventions.** Several conventions intentionally match
  vector: a spacelike signed mass larger than the spatial momentum gives zero
  energy (LVec also issues `CoordinateWarning`), `np.sqrt` of a four-vector
  returns the square root of its signed norm, `is_lightlike` defaults to an
  absolute invariant-squared tolerance of `1e-5`, and a non-unit quaternion
  scales spatial components by its squared norm.
- **Invariant evaluation.** Like vector, LVec evaluates masses and scalar
  products with ordinary float64 arithmetic by default. Use
  `lvec.invariant_policy('accurate')` to recover ill-conditioned and
  range-extreme invariants.
- **Scope.** Numba registration, symbolic SymPy, PyTrees, coordinate dispatch
  containers, and mutable ndarray-subclass behavior are outside this API.

The shared numerical domain is checked against `vector==1.9.0` in the
[reference tests](../lvec/tests/test_vector_reference.py), alongside independent
physics tests. Matching names do not imply identical rounding or policies in
undefined or nonphysical domains. [Benchmarks](../benchmarks/README.md) compare
fresh calculations and equivalent construction boundaries separately.
