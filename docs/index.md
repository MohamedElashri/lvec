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

- [Reconstruct a particle pair](tutorials/first-analysis.md): construct momenta,
  select particles, and calculate an invariant mass.
- [Analyze jagged events](tutorials/jagged-events.md): apply particle selections,
  reduce events, and handle missing data.
- [Work in a rest frame](tutorials/rest-frames.md): transform daughter momenta
  and calculate a decay angle.

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

tutorials/first-analysis
tutorials/jagged-events
tutorials/rest-frames
```

```{toctree}
:hidden:
:caption: User guide

physics
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
