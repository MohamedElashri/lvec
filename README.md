# LVec

[![CI](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml/badge.svg)](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml)
[![PyPI](https://badge.fury.io/py/lvec.svg)](https://pypi.org/project/lvec/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/MohamedElashri/lvec/blob/main/LICENSE)

Lorentz and spatial vectors for physics analysis, built on NumPy with optional
Awkward Array support. LVec provides event kinematics, angular matching,
reference-frame transformations, event reductions, and decay angles.

## Installation

NumPy is required; Awkward support is optional. Supported Python and dependency
versions are declared in [pyproject.toml](https://github.com/MohamedElashri/lvec/blob/main/pyproject.toml).

```bash
pip install lvec
pip install "lvec[awkward]"  # optional jagged arrays and records
```

## Quick start

```python
import numpy as np
from lvec import LVec, Frame

tracks = LVec.from_ptepm(
    pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
    phi=np.array([.4, -.5]), m=.13957,
)
pair = tracks[0] + tracks[1]
print(pair.mass, pair.pt, tracks[0].delta_r(tracks[1]))

cm = Frame.from_lvec(pair)
pair_in_cm = pair.to_frame(cm)
np.testing.assert_allclose(pair_in_cm.to_p4()[:3], 0., atol=1e-14)
```

Use consistent momentum, energy, and mass units with `c=1` and metric
`(+,-,-,-)`. Components are owned, read-only float64 arrays. Physical-particle
validation is optional, and numerically risky calculations receive selective
precision recovery.

## Documentation

The [user guide](https://github.com/MohamedElashri/lvec/blob/main/user-guide.md) covers the full API, NumPy and Awkward workflows,
physics conventions, cache controls, numerical accuracy, examples, and benchmarks.

The [benchmark framework](benchmarks/README.md) documents reproducible runs,
LVec performance and memory results, and optional comparisons with `vector`.

For development, install a checkout with `pip install -e ".[dev]"` and run
`python -m pytest -W error::RuntimeWarning`.

## License

LVec is distributed under the [GNU General Public License v3](https://github.com/MohamedElashri/lvec/blob/main/LICENSE).
See the [citation](https://github.com/MohamedElashri/lvec/blob/main/user-guide.md#citation) when using LVec in research.
