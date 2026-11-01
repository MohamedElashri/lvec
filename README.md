[![lvec CI](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml/badge.svg)](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml)
[![PyPI version](https://badge.fury.io/py/lvec.svg)](https://badge.fury.io/py/lvec)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

# LVec

LVec provides Lorentz, 2D, and 3D vectors for physics analysis with NumPy and
optional Awkward arrays. It supports event kinematics, angular matching,
reference-frame transforms, event sums, and decay angles.

The changes described in [Unreleased](CHANGELOG.md#unreleased) are pending the
next release. Installing from PyPI gives the currently published version; use a
checkout to try these changes before release.

## Installation

Python 3.10+ and NumPy 2.2.4+ are required. NumPy is the only mandatory runtime
dependency. Awkward 2.7.4+ is optional.

```bash
pip install lvec
pip install "lvec[awkward]"  # jagged arrays and Awkward records
```

From a checkout:

```bash
pip install -e ".[awkward,test]"
python -m pytest -W error::RuntimeWarning
```

Use `lvec[examples]` for uproot/plotting examples, `lvec[benchmark]` for comparison
and memory benchmarks, or `lvec[dev]` for all development extras.

## Quick start

```python
import numpy as np
from lvec import LVec, Frame, Vector2D, Vector3D

tracks = LVec.from_ptepm(
    pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
    phi=np.array([3.13, -3.13]), m=.13957,
)
pair = tracks[0] + tracks[1]
print(pair.mass, pair.pt, pair.rapidity)
separation = tracks[0].delta_r(tracks[1])

# A Frame stores its physical velocity relative to the lab.
cm = Frame.from_lvec(pair, name="cm")
first_in_cm = tracks[0].to_frame(cm)
pair_in_cm = pair.to_frame(cm)  # spatial momentum is close to zero

# Rotation angles are radians; 3D axes are 'x', 'y', or 'z'.
transverse = Vector2D(3., 4.).rotate(np.pi/4)
spatial = Vector3D(1., 2., 3.).rotate(np.pi/2, axis="y")
```

For variable numbers of particles per event:

```python
import awkward as ak  # install lvec[awkward]
from lvec import LVec

particles = LVec(ak.Array([[1., 2.], [], [3., None]]), 0., 0., 5.)
totals = particles.sum(axis=-1)
assert ak.to_list(totals.px) == [3., 0., 3.]
assert ak.to_list(totals.E) == [10., 0., 5.]
records = particles.to_ak_record()
restored = LVec.from_ak_record(records)
```

## Physics and precision

Use consistent momentum, mass, and energy units with `c=1` and metric
`(+,-,-,-)`. `mass2` is `E²-|p|²`; `mass` is its signed square root, so spacelike
vectors have negative mass. General mathematical four-vectors may have negative
energy. Physical-particle checks are opt-in:

```python
particle = LVec(1., 2., 3., 5., validate=True)
mask = particle.is_physical(rtol=1e-12, atol=0.)
particle.validate_particle()
```

`boost(bx, by, bz)` is active. `Frame.from_lvec(v)` uses `+p/E`, and
`to_frame(frame)` assumes lab coordinates and applies `-beta`. See the
[migration guide](docs/migration.md) before updating existing analyses: these
conventions correct earlier inconsistent behavior.

Inputs are copied into owned, read-only float64 components. Scalars broadcast;
array components must agree in shape and missing-value structure. Construct a
new vector to change its components.

Numerically risky entries receive additional arithmetic for invariants,
transforms, and event sums. For example, `LVec(1e200, 1., 0., 1e200).mass2` is
`-1`, rather than losing the residual to cancellation. This cannot restore
precision lost in supplied inputs or represent outputs outside float64's range.
Detection also has measurable cost: see the
[accuracy policy and measured overhead](docs/numerical_accuracy.md).

## Cache controls

Derived observables are cached on first access. Intermediate arrays are
temporary by default; enable `cache_intermediates=True` when reuse warrants the
extra retained memory.

```python
v = LVec(1., 2., 3., 5., max_cache_size=16, cache_stats=False)
v.set_ttl("mass", 60.)  # seconds; policy can be set before first access
mass = v.mass
retained_bytes = v.cache_stats["overall"]["retained_bytes"]
v.clear_cache()
```

`max_cache_size=None` permits unlimited entries; `0` disables retention. Limits
count entries, rather than bytes. See [cache policies](docs/README.md#cache-policies)
and the [benchmark instructions](docs/README.md#validation-and-benchmarks).

## Documentation

- [API and usage guide](docs/README.md)
- [Physics analysis helpers and decay-angle conventions](docs/analysis_features.md)
- [Migration and compatibility changes](docs/migration.md)
- [Numerical accuracy, stress tests, and performance measurements](docs/numerical_accuracy.md)
- [Examples and data requirements](examples/README.md)
- [Unreleased changes](CHANGELOG.md)

## Citation

```bibtex
@software{lvec2024,
  author = {Mohamed Elashri},
  title = {LVec: A Python package for handling Lorentz vectors},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MohamedElashri/lvec}
}
```

Contributions are welcome through GitHub pull requests. LVec is distributed
under the [GNU General Public License v3](LICENSE).
