[![lvec CI](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml/badge.svg)](https://github.com/MohamedElashri/lvec/actions/workflows/Build_Package.yml)

# LVec

[![PyPI version](https://badge.fury.io/py/lvec.svg)](https://badge.fury.io/py/lvec)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> ⚠️ This project is a work in progress

A Python package for seamless handling of Lorentz vectors, 2D vectors, and 3D vectors in HEP analysis, bridging the gap between Scikit-HEP and ROOT ecosystems.

## Motivation

LVec aims to simplify HEP analysis by providing a unified interface for working with various vector types across different frameworks. It seamlessly integrates with both the Scikit-HEP ecosystem (uproot, vector, awkward) and ROOT/PyROOT, enabling physicists to write more maintainable and efficient analysis code.

## Installation

```bash
pip install lvec
```

For development installation:
```bash
git clone https://github.com/MohamedElashri/lvec
cd lvec
pip install -e ".[dev]"
```

## Quick Start

### Lorentz Vectors
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
```

### 2D Vectors
```python
from lvec import Vector2D

# Create a 2D vector
vec2d = Vector2D(x=3.0, y=4.0)

# Access properties
print(f"Magnitude: {vec2d.r}")
print(f"Angle (phi): {vec2d.phi}")

# Rotate vector
rotated = vec2d.rotate(angle=np.pi/4)  # 45 degrees rotation
```

### 3D Vectors
```python
from lvec import Vector3D

# Create a 3D vector
vec3d = Vector3D(x=1.0, y=2.0, z=3.0)

# Access properties
print(f"Magnitude: {vec3d.r}")
print(f"Theta: {vec3d.theta}")
print(f"Phi: {vec3d.phi}")

# Rotate around axis
rotated = vec3d.rotate(theta=np.pi/2, axis=[0, 1, 0])  # 90 degrees around y-axis
```

### Array Operations
```python
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

### Lorentz Vector (LVec) Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|----------|
| `__init__` | Create a Lorentz vector | `px, py, pz, E` | `LVec` |
| `from_ptepm` | Create from pt, eta, phi, mass | `pt, eta, phi, m` | `LVec` |
| `from_p4` | Create from px, py, pz, E | `px, py, pz, E` | `LVec` |
| `from_ary` | Create from dictionary | `ary_dict` with px, py, pz, E keys | `LVec` |
| `from_vec` | Create from vector-like object | `vobj` with px, py, pz, E attributes | `LVec` |
| `boost` | Boost vector to new frame | `bx, by, bz` | `LVec` |
| `mass` | Get invariant mass | - | `float` |
| `pt` | Get transverse momentum | - | `float` |
| `eta` | Get pseudorapidity | - | `float` |
| `phi` | Get azimuthal angle | - | `float` |
| `E` | Get energy | - | `float` |
| `p` | Get total momentum | - | `float` |

### 2D Vector (Vector2D) Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|----------|
| `__init__` | Create a 2D vector | `x, y` | `Vector2D` |
| `x` | Get x component | - | `float` |
| `y` | Get y component | - | `float` |
| `r` | Get vector magnitude | - | `float` |
| `phi` | Get azimuthal angle | - | `float` |
| `dot` | Compute dot product | `other` | `float` |

### 3D Vector (Vector3D) Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|----------|
| `__init__` | Create a 3D vector | `x, y, z` | `Vector3D` |
| `x` | Get x component | - | `float` |
| `y` | Get y component | - | `float` |
| `z` | Get z component | - | `float` |
| `r` | Get vector magnitude | - | `float` |
| `rho` | Get cylindrical radius | - | `float` |
| `phi` | Get azimuthal angle | - | `float` |
| `theta` | Get polar angle | - | `float` |
| `dot` | Compute dot product | `other` | `float` |
| `cross` | Compute cross product | `other` | `Vector3D` |

## Requirements

- Python >= 3.10
- NumPy >= 1.20.0
- Awkward >= 2.0.0

For development:
- pytest >= 7.0.0
- uproot >= 5.0.0

## Citation

If you use LVec in your research, please cite:

```bibtex
@software{lvec2024,
  author       = {Mohamed Elashri},
  title        = {LVec: A Python package for handling Lorentz vectors},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/MohamedElashri/lvec}
}
```

## Documentation

For detailed documentation and examples, visit our [documentation page](https://github.com/MohamedElashri/lvec/tree/main/docs).

## Examples

Check out our [examples directory](https://github.com/MohamedElashri/lvec/tree/main/examples) for comprehensive examples including:
- Basic vector operations
- Decay reconstructions
- Frame transformations
- 2D vector manipulations
- 3D spatial analysis
- Advanced selections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
