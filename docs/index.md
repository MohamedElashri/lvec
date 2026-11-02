# LVec

LVec provides Lorentz four-vectors and spatial vectors for particle-physics
analysis with NumPy and Awkward Array. The same operations apply to a single
particle, a dense collection, or a sample of variable-length events.

## Installation

```bash
python -m pip install lvec
python -m pip install "lvec[awkward]"  # include jagged event data
```

Requires Python 3.10 or later, NumPy 2.2.4 or later, and optionally
Awkward 2.7.4 or later. No ROOT installation is required.

## Start with an analysis

- [Scalar vectors](tutorials/scalars.md): spatial algebra, signed norms,
  rotations, and explicit dimensional embeddings.
- [NumPy collections](tutorials/numpy.md): regular shapes, broadcasting,
  event reductions, indexing, and exports.
- [Reconstruct a particle pair](tutorials/first-analysis.md): construct momenta,
  select particles, and calculate an invariant mass.
- [Analyze jagged events](tutorials/jagged-events.md): apply particle selections,
  reduce events, and handle missing data.
- [Work in a rest frame](tutorials/rest-frames.md): transform daughter momenta
  and calculate a decay angle.
- [Particle combinations](tutorials/particle-combinations.md): form candidates
  inside events using native Awkward records.

## Coordinates and supported data

Input coordinates combine a transverse pair (`x`, `y` or `rho`, `phi`), a
longitudinal coordinate (`z`, `theta`, or `eta`), and for 4D a temporal
coordinate (`t` or signed `tau`). Momentum aliases include `pt`, `pz`, `E`,
and `mass`. These give two planar, six spatial, and twelve Lorentz coordinate
families; storage remains Cartesian.

| Data | Construction guide | Ownership |
| --- | --- | --- |
| Single vectors | [Scalar construction](construction/scalars.md) | Owned values |
| Regular arrays | [NumPy construction](construction/numpy.md) | Immutable snapshots with property caches |
| Jagged arrays | [Awkward construction](construction/awkward.md) | Owned collections or native views |

The [coordinate guide](guides/coordinates.md) defines the mathematical
representations. Every property and method is listed in the
[2D](reference/vector2d.md), [3D](reference/vector3d.md), and
[4D](reference/vector4d.md) interfaces, with momentum aliases on the same classes.

## Conventions and reference

Use consistent momentum and energy units, with $c=1$ and metric $(+,-,-,-)$.
The constructor order is `(px, py, pz, E)`. Owned collections store immutable
float64 components and cache observables; native Awkward records provide views.

The [physics guide](physics.md) defines signs, physical domains, and numerical
limits. The [API reference](api-reference.md) describes constructors, operations,
and array behavior. For existing analyses, see [migration from vector](vector-migration.md).
Measured performance and reproduction instructions are in [benchmarks](benchmarks.md).

```{toctree}
:hidden:
:caption: Tutorials

tutorials/scalars
tutorials/numpy
tutorials/jagged-events
tutorials/first-analysis
tutorials/particle-combinations
tutorials/rest-frames
```

```{toctree}
:hidden:
:caption: Construction

construction/scalars
construction/numpy
construction/awkward
```

```{toctree}
:hidden:
:caption: User guide

physics
guides/coordinates
guides/numerics
vector-migration
benchmarks
```

```{toctree}
:hidden:
:caption: Reference

api-reference
```

```{toctree}
:hidden:
:caption: Development

development
```

## Citation

Please cite LVec using [CITATION.cff](https://github.com/MohamedElashri/lvec/blob/main/CITATION.cff).

## Licence

LVec is licensed under the [GNU General Public License v3](https://github.com/MohamedElashri/lvec/blob/main/LICENSE).
