# LVec

[![CI](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml/badge.svg)](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml)
[![PyPI](https://badge.fury.io/py/lvec.svg)](https://pypi.org/project/lvec/)

**Lorentz vectors for particle physics with NumPy and Awkward Array.**

Reconstruct invariant masses, select particles, transform reference frames,
and calculate decay angles on scalars, dense arrays, or jagged event data.
LVec also provides two- and three-dimensional spatial vectors.

## Install

Requires Python ≥3.10 and NumPy ≥2.2.4. Jagged collections additionally require
Awkward ≥2.7.4.

```bash
python -m pip install lvec
python -m pip install "lvec[awkward]"  # for jagged event data
```

## Reconstruct a two-particle system

Use consistent energy and momentum units, with `c=1` and metric `(+,-,-,-)`.

```python
from lvec import LVec

daughter1 = LVec(px=3., py=0., pz=0., E=5.)
daughter2 = LVec(px=0., py=4., pz=0., E=5.)
parent = daughter1 + daughter2

print(f"Pair mass: {parent.mass:.3f}")  # Pair mass: 8.660
daughter1_cm = daughter1.boost_to_rest(parent)
```

The same operations act elementwise on arrays. Components are independent,
read-only float64 snapshots, so editing an input array cannot change a vector.
Use `validate=True` when constructing physical particle momenta. For event
samples, start with [jagged events and reductions](user-guide.md#jagged-events-and-reductions).

## Documentation

| Start here | Contents |
| --- | --- |
| [Physics guide](user-guide.md) | Coordinates, invariant mass, event selections, rest frames, decay angles, and numerical conventions |
| [API reference](docs/api-reference.md) | Constructors, observables, transformations, array operations, and cache options |
| [Migrating from vector](docs/vector-migration.md) | Familiar interfaces and scientific differences to check |
| [Benchmarks](benchmarks/README.md) | Representative measurements, ownership boundaries, and reproduction |
| [Contributing](CONTRIBUTING.md) | Testing, numerical bug reports, and releases |

## Citation

Please cite LVec using [CITATION.cff](CITATION.cff).

## Licence

LVec is licensed under the [GNU General Public License v3](LICENSE).
