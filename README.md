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

Examples describe this source revision; use the matching release tag for an
installed release.

## Reconstruct a two-particle system

Use consistent energy and momentum units, with `c=1` and metric `(+,-,-,-)`.

```python
import numpy as np
from lvec import LVec

daughter1 = LVec(px=3., py=0., pz=0., E=5.)
daughter2 = LVec(px=0., py=4., pz=0., E=5.)
parent = daughter1 + daughter2

np.testing.assert_allclose(parent.mass2, 75.)
daughter1_cm = daughter1.boost_to_rest(parent)
daughter2_cm = daughter2.boost_to_rest(parent)
np.testing.assert_allclose((daughter1_cm + daughter2_cm).p, 0., atol=1e-14)
```

The same operations act elementwise on arrays. Collection constructors retain
independent, read-only float64 components; derived observables are cached.
Signed invariants and selective numerical recovery support lightlike,
spacelike, and extreme inputs. Physical-particle validation is opt-in.

## Documentation

| Start here | Contents |
| --- | --- |
| [Physics guide](https://github.com/MohamedElashri/lvec/blob/main/user-guide.md) | Coordinates, invariant mass, event selections, rest frames, decay angles, and numerical conventions |
| [API reference](https://github.com/MohamedElashri/lvec/blob/main/docs/api-reference.md) | Constructors, observables, transformations, array operations, and cache options |
| [Migrating from vector](https://github.com/MohamedElashri/lvec/blob/main/docs/vector-migration.md) | Familiar interfaces and scientific differences to check |
| [Benchmarks](https://github.com/MohamedElashri/lvec/blob/main/benchmarks/README.md) | Representative measurements, ownership boundaries, and reproduction |
| [Contributing](https://github.com/MohamedElashri/lvec/blob/main/CONTRIBUTING.md) | Testing, numerical bug reports, and releases |

## Citation

Use [CITATION.cff](https://github.com/MohamedElashri/lvec/blob/main/CITATION.cff) to cite the software. For reproducible research,
record the release or Git commit, Python and dependency versions, and relevant
numerical settings. A development checkout may share version metadata with a
release.

Also cite the dependencies used in your analysis: [NumPy](https://numpy.org/citing-numpy/)
and, where applicable, [Awkward Array](https://github.com/scikit-hep/awkward/blob/main/CITATION.cff).

LVec is distributed under the [GNU General Public License v3](https://github.com/MohamedElashri/lvec/blob/main/LICENSE).
