# LVec

[![PyPI version](https://badge.fury.io/py/lvec.svg)](https://badge.fury.io/py/lvec)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> ⚠️ This project is a work in progress

A Python package for seamless handling of Lorentz vectors in High Energy Physics analysis, bridging the gap between Scikit-HEP and ROOT CERN ecosystems.

## Motivation

LVec aims to simplify High Energy Physics analysis by providing a unified interface for working with Lorentz vectors across different frameworks. It seamlessly integrates with both the Scikit-HEP ecosystem (uproot, vector, awkward) and ROOT/PyROOT, enabling physicists to write more maintainable and efficient analysis code.

## Installation

```bash
pip install lvec
```

## Quick Start

```python
from lvec import LVec
import numpy as np

# Create a single Lorentz vector
v = LVec(px=1.0, py=2.0, pz=3.0, E=4.0)

# Access properties
print(f"Mass: {v.mass}")
print(f"pt: {v.pt}")

# Create from pt, eta, phi, mass
v2 = LVec.from_ptepm(pt=5.0, eta=0.0, phi=0.0, m=1.0)

# Vector operations
v3 = v1 + v2
v4 = v1 * 2.0

# Works with numpy arrays
px = np.array([1.0, 2.0, 3.0])
py = np.array([2.0, 3.0, 4.0])
pz = np.array([3.0, 4.0, 5.0])
E = np.array([4.0, 5.0, 6.0])
vectors = LVec(px, py, pz, E)

# Works with awkward arrays
import awkward as ak
vectors_ak = LVec(ak.Array(px), ak.Array(py), ak.Array(pz), ak.Array(E))
```

## Available Methods

### Constructors

| Method | Description |
|--------|-------------|
| `LVec(px, py, pz, E)` | Create from Cartesian components |
| `from_p4(px, py, pz, E)` | Alternative constructor using Cartesian components |
| `from_ptepm(pt, eta, phi, m)` | Create from pt, eta, phi, mass |
| `from_ary(ary_dict)` | Create from dictionary with px, py, pz, E keys |
| `from_vec(vobj)` | Create from another vector-like object |

### Properties

| Property | Description |
|----------|-------------|
| `px, py, pz` | Momentum components |
| `E` | Energy |
| `pt` | Transverse momentum |
| `p` | Total momentum |
| `mass` | Invariant mass |
| `phi` | Azimuthal angle |
| `eta` | Pseudorapidity |

### Operations

| Operation | Description |
|-----------|-------------|
| `+` | Vector addition |
| `-` | Vector subtraction |
| `*` | Scalar multiplication |
| `[]` | Array indexing |

### Transformations

| Method | Description |
|--------|-------------|
| `boost(bx, by, bz)` | General Lorentz boost |
| `boostz(bz)` | Boost along z-axis |
| `rotx(angle)` | Rotation around x-axis |
| `roty(angle)` | Rotation around y-axis |
| `rotz(angle)` | Rotation around z-axis |

### Conversions

| Method | Description |
|--------|-------------|
| `to_p4()` | Get (px, py, pz, E) tuple |
| `to_ptepm()` | Get (pt, eta, phi, mass) tuple |
| `to_np()` | Convert to NumPy arrays |
| `to_ak()` | Convert to Awkward arrays |
| `to_root_dict()` | Convert to ROOT-compatible dictionary |

## Advanced Usage

LVec supports both NumPy and Awkward array backends, automatically choosing the appropriate backend based on input types. It provides efficient caching of derived properties and handles array broadcasting.

Examples of advanced usage can be found in the [examples](examples/) directory.

## Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Submit pull requests
- Suggest new features
- Share feedback

## Requirements

- Python 3.8+
- NumPy
- Awkward Array (optional, for Awkward array support)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citing

If you use LVec in your research, please cite:

```bibtex
@software{lvec,
  author = {Mohamed Elashri},
  title = {LVec: A Python Package for Lorentz Vector Analysis},
  year = {2024},
  url = {https://github.com/MohamedElashri/lvec}
}
```