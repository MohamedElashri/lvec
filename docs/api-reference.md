# API reference

The dimensional interfaces list every public property, method, alias, and
coordinate conversion. Scalars, owned NumPy collections, and owned Awkward
collections use the same numerical interface. Native records expose the
subset described in [Awkward construction](construction/awkward.md).

## Complete interfaces

| Interface | Geometry |
| --- | --- |
| [2D vectors](reference/vector2d.md) | Transverse Euclidean vectors; `Vector2D`, `Momentum2D` |
| [3D vectors](reference/vector3d.md) | Spatial Euclidean vectors; `Vector3D`, `Momentum3D` |
| [4D vectors](reference/vector4d.md) | Lorentz vectors; `LVec`, `Vector4D`, `Momentum4D` |
| [Reference frames](reference/frame.md) | Frame velocities and CM factories |
| [Operators and NumPy](reference/operators.md) | Python operators, reductions, functions, and ufuncs |
| [Exceptions](reference/exceptions.md) | Shape, input, backend, and dependency errors |

```{toctree}
:hidden:

reference/vector2d
reference/vector3d
reference/vector4d
reference/frame
reference/operators
reference/exceptions
```

## Construction and coordinates

Choose [scalar](construction/scalars.md), [NumPy](construction/numpy.md),
or [Awkward](construction/awkward.md) input. The
[constructor parameters](reference/constructors.md) define the public factories
and cache options. The [coordinate guide](guides/coordinates.md) gives the
mathematical definitions, signed invariant policy, and embedding conventions.

## Observables

The [observable guide](reference/vectors.md) explains masses, rapidity,
transverse quantities, and physical validation. Individual definitions and
all aliases appear in the dimensional interfaces above.

## Algebra and geometry

Use the [operator reference](reference/operators.md) for Python and NumPy
syntax. Geometry methods, direction tests, and angular separations are
listed individually in each applicable dimensional interface.

## Transformations

The [transformation guide](reference/transformations.md) explains active
boosts, CM transforms, Euler conventions, quaternions, and decay geometry.
The [4D interface](reference/vector4d.md) includes every axis-boost spelling
and the complete frame-conversion signatures.

## Arrays and interchange

See [arrays, ownership, and caching](reference/arrays.md) for shapes,
reductions, snapshots, native views, and exports. The
[NumPy tutorial](tutorials/numpy.md) demonstrates regular collections;
[particle combinations](tutorials/particle-combinations.md) demonstrates
native Awkward records inside event structures.

## Cache options and errors

Cache options are part of the [constructors](reference/constructors.md).
[Cache controls](reference/arrays.md#property-caches) describe lifetimes,
retained entries, counters, and invalidation. The [exception reference](reference/exceptions.md)
and [numerical guide](guides/numerics.md) define failure modes and physical domains.

```{toctree}
:hidden:
:caption: Topic guides

reference/constructors
reference/vectors
reference/transformations
reference/arrays
```
