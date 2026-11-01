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
# Include jagged-array support:
pip install "lvec[awkward]"
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
v3 = v + v2
v4 = v * 2.0
```

### Memory Optimization
```python
# Create vector with cache size limit (LRU eviction)
v = LVec(px=1.0, py=2.0, pz=3.0, E=4.0, max_cache_size=100)

# Create vector with TTL for all properties
v_ttl = LVec(px=1.0, py=2.0, pz=3.0, E=4.0, default_ttl=60)  # 60 second TTL

# Set TTL for specific property
v.set_ttl('pt', 10)  # 10 second TTL for transverse momentum

# Get cache statistics
stats = v.cache_stats
print(f"Cache hit ratio: {v.cache_hit_ratio}")
```

### Reference Frames
```python
from lvec import LVec, Frame

# Create particles in the lab frame
p1_lab = LVec(px=0.0, py=0.0, pz=20.0, E=25.0)
p2_lab = LVec(px=0.0, py=0.0, pz=-15.0, E=20.0)
total_lab = p1_lab + p2_lab

# Create reference frames
lab_frame = Frame.rest(name="lab")  # Stationary frame
cm_frame = Frame.from_lvec(total_lab, name="cm")  # Center-of-mass frame

# Transform particles to center-of-mass frame
p1_cm = p1_lab.transform_frame(lab_frame, cm_frame)
p2_cm = p2_lab.transform_frame(lab_frame, cm_frame)

# Verify momentum conservation in CM frame
total_cm = p1_cm + p2_cm
print(f"Total momentum in CM: ({total_cm.px:.3f}, {total_cm.py:.3f}, {total_cm.pz:.3f})")
# Should output values close to (0, 0, 0)

# Alternative: directly transform to a frame
p1_cm_alt = p1_lab.to_frame(cm_frame)
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

See the [physics analysis guide](docs/analysis_features.md) for kinematics,
angular matching, event sums, Awkward records, and decay-angle conventions.

### Lorentz Vector (LVec) Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|----------|
| `__init__` | Create a Lorentz vector | `px, py, pz, E, max_cache_size=None, default_ttl=None` | `LVec` |
| `from_ptepm` | Create from pt, eta, phi, mass | `pt, eta, phi, m` | `LVec` |
| `from_p4` | Create from px, py, pz, E | `px, py, pz, E` | `LVec` |
| `from_ary` | Create from dictionary | `ary_dict` with px, py, pz, E keys | `LVec` |
| `from_vec` | Create from vector-like object | `vobj` with px, py, pz, E attributes | `LVec` |
| `boost` | Boost vector to new frame | `bx, by, bz` | `LVec` |
| `to_frame` | Transform to specified frame | `frame` | `LVec` |
| `transform_frame` | Transform between frames | `current_frame, target_frame` | `LVec` |
| `set_ttl` | Set TTL for property | `property_name, ttl_seconds` | None |
| `clear_expired` | Remove expired cache values | - | `int` (count) |
| `cache_stats` | Get cache statistics | - | `dict` |
| `cache_hit_ratio` | Get cache hit ratio | - | `float` |
| `reset_cache_stats` | Reset cache statistics | - | None |
| `cache_size` | Get current cache size | - | `int` |
| `mass` | Get invariant mass | - | `float` |
| `pt` | Get transverse momentum | - | `float` |
| `eta` | Get pseudorapidity | - | `float` |
| `phi` | Get azimuthal angle | - | `float` |
| `E` | Get energy | - | `float` |
| `p` | Get total momentum | - | `float` |

### Frame Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|----------|
| `__init__` | Create a reference frame | `bx, by, bz, name` | `Frame` |
| `rest` | Create a stationary frame | `name` | `Frame` |
| `from_lvec` | Create frame in which a vector is at rest | `total_lvec, name` | `Frame` |
| `center_of_mass` | Create center-of-mass frame from vectors | `lvec_list, name` | `Frame` |

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
- NumPy >= 2.2.4
- Awkward >= 2.7.4 (optional, install `lvec[awkward]`)

Install `lvec[test]` for pytest, `lvec[examples]` for ROOT-reading and plotting
examples, or `lvec[benchmark]` for comparison and memory benchmarks.
`lvec[dev]` includes all of these. NumPy is the only mandatory runtime dependency.

## Physics conventions and validation

Use consistent momentum, mass, and energy units throughout an analysis, with
c=1 and metric (+, -, -, -). `mass2` returns E²-|p|² unchanged; `mass` is
sign(mass2) sqrt(abs(mass2)), so spacelike vectors have negative signed mass.
This corrects the earlier behavior that took the absolute value of negative
mass squared. General mathematical four-vectors, including negative energy,
are accepted so subtraction and negative scaling remain valid.

For physical particles, opt in to checks:

```python
particle = LVec(1., 2., 3., 5., validate=True)
mask = vectors.is_physical()  # finite, E >= 0, timelike/lightlike within tolerance
particle.validate_particle(rtol=1e-12, atol=0.)
```

The relative tolerance scales with max(E², |p|²); `atol` has units of energy
squared. Validation never changes raw mass squared. Missing Awkward entries
remain missing and are excluded from the aggregate particle check.

`eta` uses asinh(pz/pt), with signed infinity on the beam axis and zero for
the zero spatial vector. Boost velocities must be finite with |beta| < 1.
`boost(beta)` is active: a rest particle acquires momentum in the direction
of beta. A `Frame` stores its physical velocity relative to the lab;
`Frame.from_lvec(v)` now uses +p/E. `to_frame(frame)` assumes lab coordinates
and applies -beta. `transform_frame(source, target)` first applies +beta_source,
then -beta_target. These sign corrections make both CM transformation paths
agree and are a change from the earlier inconsistent frame convention.

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
- Reference frame transformations
- Decay reconstructions
- Center-of-mass calculations
- 2D vector manipulations
- 3D spatial analysis
- Advanced selections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Component ownership

Construction snapshots NumPy and Awkward component arrays and marks their
NumPy buffers read-only. Changing caller-owned inputs cannot change a vector
or leave its cached observables stale. Public component arrays reject ordinary
in-place edits; create a new vector for changed components. This adds a copy at
construction. Deliberate manipulation of private buffers, Awkward `.layout` replacement,
or forcibly changing writeability is outside the supported API.

Components are promoted to float64 before physics arithmetic, including integer
and float32 arrays. This avoids integer overflow in squared momenta and provides
the same precision policy for scalar, NumPy, and Awkward inputs. Complex-valued
components are not supported.

Internal NumPy arithmetic reuses freshly allocated results and existing owned
read-only components instead of snapshotting them again. Public constructors
continue to snapshot external inputs; this private optimization does not expose
a borrowed mutable-input mode.
