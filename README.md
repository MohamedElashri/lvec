# LVec

A Python package for handling Lorentz vectors with support for both NumPy and Awkward array backends.

## Installation

From the project directory:

```bash
pip install -e .
```

For development installation with test dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from lvec import LVec

# Create a single vector
v1 = LVec(px=1.0, py=2.0, pz=3.0, E=4.0)

# Access properties
print(f"Mass: {v1.mass}")
print(f"pt: {v1.pt}")

# Create from pt, eta, phi, mass
v2 = LVec.from_ptepm(pt=5.0, eta=0.0, phi=0.0, m=1.0)

# Vector operations
v3 = v1 + v2
v4 = v1 * 2.0
```

## Testing

Run the tests using pytest:

```bash
pytest
```