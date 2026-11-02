# LVec

[![CI](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml/badge.svg)](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml)
[![PyPI](https://badge.fury.io/py/lvec.svg)](https://pypi.org/project/lvec/)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-blue)](https://github.com/MohamedElashri/lvec/blob/main/pyproject.toml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/MohamedElashri/lvec/blob/main/LICENSE)

**Lorentz and spatial vectors for array-based physics analysis.**

LVec provides relativistic kinematics for NumPy arrays and optional Awkward
Array collections. It combines four-momentum arithmetic, reference-frame
transformations, event reductions, angular observables, and decay angles with
explicit component ownership and selective numerical precision recovery.

The package is intended for calculations on scalars, dense arrays, and jagged
event data. NumPy is the only required runtime dependency. Awkward support is
optional. Selected examples demonstrate ROOT I/O with uproot and plotting.

[User guide](https://github.com/MohamedElashri/lvec/blob/main/user-guide.md)
· [Migrating from vector](https://github.com/MohamedElashri/lvec/blob/main/docs/vector-migration.md)
· [Benchmarks](https://github.com/MohamedElashri/lvec/blob/main/benchmarks/README.md)
· [Examples](https://github.com/MohamedElashri/lvec/tree/main/examples)
· [Citation](#citation)

## Capabilities

| Area | Supported operations |
| --- | --- |
| Coordinates | All 20 Cartesian/polar coordinate families across 2D/3D/4D; momentum aliases; projections and embeddings |
| Algebra | Arithmetic, Euclidean/Minkowski products, unit vectors, comparisons, and direction/causal predicates |
| Event collections | NumPy and jagged Awkward arrays; event-wise/total reductions; optional native record behavior |
| Transformations | Beta/p4/gamma Lorentz boosts; arbitrary-axis, Euler and quaternion rotations; general matrices; named frames |
| Analysis observables | Invariant and transverse mass, transverse energy, rapidity, beta/gamma, angular distances, and helicity angles |
| Spatial vectors | Complete numerical 2D/3D geometry through `Vector2D` and `Vector3D` |
| Derived properties | Cached results with optional size limits, TTLs, statistics, and intermediate retention |

## Installation

Python **3.10 or newer** is required. The minimum runtime versions are NumPy
2.2.4 and, when enabled, Awkward 2.7.4. Supported versions and optional extras
are defined in [pyproject.toml](https://github.com/MohamedElashri/lvec/blob/main/pyproject.toml).

```bash
python -m pip install lvec
python -m pip install "lvec[awkward]"  # optional event-array support
```

To work from source:

```bash
git clone https://github.com/MohamedElashri/lvec.git
cd lvec
python -m pip install -e ".[awkward,test]"
```

This README describes the source tree on `main`. For a released installation,
consult the corresponding release tag. Benchmark reports identify the exact
source revision and dependencies used for their measurements.

## Quick start

Construct a collection of four-momenta, form its total, and transform that total
to its center-of-momentum frame:

```python
import numpy as np
from lvec import Frame, LVec

tracks = LVec(
    px=np.array([3., 0.]),
    py=np.array([0., 4.]),
    pz=np.array([0., 0.]),
    E=np.array([5., 5.]),
    validate=True,
)
np.testing.assert_allclose(tracks.mass, [4., 3.])

total = tracks.sum()
np.testing.assert_allclose(total.mass2, 75.)

cm = Frame.from_lvec(total)
rest = total.to_frame(cm)
np.testing.assert_allclose(rest.to_p4()[:3], 0., atol=1e-14)
```

Alternative constructors include `LVec.from_ptepm(pt, eta, phi, m)` and
`LVec.from_p4(px, py, pz, E)`. Components are independent, read-only float64
snapshots of public constructor inputs. Changing an input array afterwards
cannot alter a vector or invalidate its cached observables.

### Jagged event data

With the Awkward extra installed, the same API operates on variable-length
particle collections:

```python
import awkward as ak
from lvec import LVec

events = LVec(
    px=ak.Array([[3., 0., None], [], None, [1.]]),
    py=ak.Array([[0., 4., None], [], None, [0.]]),
    pz=ak.Array([[0., 0., None], [], None, [0.]]),
    E=ak.Array([[5., 5., None], [], None, [2.]]),
)
event_totals = events.sum(axis=-1)
print(ak.to_list(event_totals.mass))
```

Missing particles are skipped by reductions, empty events give zero
four-momentum, and missing events remain missing. Component arrays must have
matching logical shapes and missing-value positions; scalar components can
broadcast to that structure. The [event-collection guide](https://github.com/MohamedElashri/lvec/blob/main/user-guide.md#event-collections-and-reductions)
explains reductions and conversion to and from Awkward records.

## Scientific conventions and numerical scope

- Four-momenta use `(px, py, pz, E)`, natural units with `c = 1`, and the
  Minkowski metric `(+,-,-,-)`. Use consistent momentum, energy, and mass units.
- `mass2` is the signed invariant `E² − px² − py² − pz²`; `mass` is its signed
  square root. Spacelike vectors therefore have negative mass values.
- Angles are in radians. `boost` applies an active boost; `to_frame` uses the
  passive transformation into the specified frame.
- Physical-particle validation is opt-in with `validate=True` or
  `validate_particle()`. General mathematical four-vectors can have negative
  energy or spacelike invariants.

Ordinary calculations use vectorized float64 arithmetic. Selected cancellation
and range-risk cases receive compensated arithmetic, exact binary accumulation,
or Decimal recovery as appropriate. Detection is heuristic: it does not promise
correct rounding or reversible transformations for every input. Input rounding,
float64 representability, and the conditioning of the requested calculation
remain limitations. See [numerical accuracy](https://github.com/MohamedElashri/lvec/blob/main/user-guide.md#numerical-accuracy)
for the algorithms, conventions, and boundaries of these checks.

## Validation and performance

The test suite includes independent Fraction and Decimal references, randomized
Lorentz and frame invariants, two-body decay conservation, near-null and extreme
inputs, nested missing data, cache lifecycle checks, and buffer-ownership tests.
CI covers Python 3.10–3.15, minimum and current dependencies, NumPy-only
installations, and built distributions. The exact matrix is maintained in the
[CI workflow](https://github.com/MohamedElashri/lvec/blob/main/.github/workflows/Build_Package.yml).

The [benchmark report](https://github.com/MohamedElashri/lvec/blob/main/benchmarks/README.md)
records LVec scaling, cache behavior, allocation measurements, numerical recovery
cost, and selected comparisons with Scikit-HEP `vector`. It separates cached
reads, fresh calculations, construction, and complete workflows. Results include
hardware, source revision, dependency versions, measurement boundaries, and
observed dispersion; they describe particular workloads on one machine.

The [kernel optimization study](https://github.com/MohamedElashri/lvec/blob/main/benchmarks/README.md#kernel-optimization-update)
compares shared and per-particle transformation parameters, small arrays,
complete workflows, and native record construction. Optimized kernels avoid
repeated parameter broadcasting and process large operations in bounded
blocks, retaining selective numerical recovery and owned collection snapshots.
Native Awkward record factories provide ordinary view semantics when shared
buffers are appropriate.

## Development

The [feature parity audit](https://github.com/MohamedElashri/lvec/blob/main/docs/vector-feature-parity.md) compares the public
mathematical capabilities of Scikit-HEP `vector` with LVec, records semantic
differences, and identifies the numerical core implemented by LVec. The
[migration guide](https://github.com/MohamedElashri/lvec/blob/main/docs/vector-migration.md) explains familiar constructors,
array operations, scientific policies, and performance paths.

```bash
python -m pip install -e ".[awkward,test,benchmark]"
python -m pytest -W error::RuntimeWarning
python -m benchmarks --check --sizes 1 17
```

Benchmarks run from a checkout and print Markdown reports. For a full run,
use `python -m benchmarks`; consult the benchmark README for optional workloads
and reproduction commands. Generated results stay outside version control.

Contributions and numerical counterexamples are welcome through
[issues](https://github.com/MohamedElashri/lvec/issues) and pull requests.
Include a minimal reproducer, Python and dependency versions, the relevant input
layout, and an independent expected result when reporting numerical behavior.

## Citation

If you use LVec in research, cite the software and record the exact release or
Git commit used, together with the relevant dependencies. Machine-readable
attribution is provided in [CITATION.cff](https://github.com/MohamedElashri/lvec/blob/main/CITATION.cff).
A general software citation is:

```bibtex
@software{lvec,
  author = {Elashri, Mohamed},
  title = {LVec: Lorentz and spatial vectors for physics analysis},
  url = {https://github.com/MohamedElashri/lvec}
}
```

Also acknowledge the dependencies used in an analysis according to the
[NumPy citation guidance](https://numpy.org/citing-numpy/) and
[Awkward Array citation metadata](https://github.com/scikit-hep/awkward/blob/main/CITATION.cff).

## License

LVec is distributed under the [GNU General Public License, version 3](https://github.com/MohamedElashri/lvec/blob/main/LICENSE).
