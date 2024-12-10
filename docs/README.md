# LVec Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Backend System](#backend-system)
7. [Advanced Usage](#advanced-usage)
8. [Performance Considerations](#performance-considerations)


## Overview

LVec is a Python package designed for High Energy Physics (HEP) analysis, providing a unified interface for handling Lorentz vectors. It bridges the gap between different HEP ecosystems (Scikit-HEP and ROOT) and supports both NumPy and Awkward array backends.

### Dependencies
- Python 3.8+
- NumPy (required)
- Awkward Array (optional, for Awkward array support)

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

## API Reference

### LVec Class

```python
class LVec:
    """
    A class representing a Lorentz vector with support for both NumPy and Awkward array backends.
    
    The LVec class provides a complete set of methods and properties for handling 4-vectors
    in particle physics analysis.
    """
```

#### Constructors

```python
def __init__(self, px, py, pz, E):
    """
    Initialize a Lorentz vector from its Cartesian components.

    Parameters
    ----------
    px : float, array_like
        x-component of momentum
    py : float, array_like
        y-component of momentum
    pz : float, array_like
        z-component of momentum
    E : float, array_like
        Energy

    Returns
    -------
    LVec
        Initialized Lorentz vector instance

    Notes
    -----
    - If any input is an Awkward array, all inputs are converted to Awkward arrays
    - Otherwise, inputs are converted to NumPy arrays
    - All inputs must have compatible shapes
    """
```

```python
@classmethod
def from_ptepm(cls, pt, eta, phi, m):
    """
    Create a Lorentz vector from pt, eta, phi, mass coordinates.

    Parameters
    ----------
    pt : float, array_like
        Transverse momentum
    eta : float, array_like
        Pseudorapidity
    phi : float, array_like
        Azimuthal angle
    m : float, array_like
        Mass

    Returns
    -------
    LVec
        New Lorentz vector instance

    Examples
    --------
    >>> vec = LVec.from_ptepm(50.0, 0.0, 0.0, 91.2)  # Z boson at rest in eta
    >>> print(f"pT: {vec.pt:.1f}, mass: {vec.mass:.1f}")
    pT: 50.0, mass: 91.2
    """
```

```python
@classmethod
def from_p4(cls, px, py, pz, E):
    """
    Create a Lorentz vector from Cartesian components (alternative constructor).

    Parameters
    ----------
    px, py, pz : float, array_like
        Momentum components
    E : float, array_like
        Energy

    Returns
    -------
    LVec
        New Lorentz vector instance
    """
```
## Properties

All properties are cached for performance optimization. The cache is automatically invalidated when the vector is modified.

### Kinematic Properties

```python
@property
def pt(self):
    """
    Transverse momentum (pT).

    Returns
    -------
    array_like
        The magnitude of the momentum in the transverse plane
        sqrt(px² + py²)

    Notes
    -----
    - Result is cached until vector is modified
    - Always non-negative
    """

@property
def eta(self):
    """
    Pseudorapidity.

    Returns
    -------
    array_like
        η = -ln[tan(θ/2)] where θ is the polar angle
        
    Notes
    -----
    - Cached property
    - Undefined for pt = 0 (returns ±inf)
    - Independent of energy/mass
    """

@property
def phi(self):
    """
    Azimuthal angle.

    Returns
    -------
    array_like
        φ = atan2(py, px)
        
    Notes
    -----
    - Range: [-π, π]
    - Cached property
    """

@property
def mass(self):
    """
    Invariant mass.

    Returns
    -------
    array_like
        m = sqrt(E² - p²)
        
    Notes
    -----
    - Returns real part only if E² < p²
    - For virtual particles, can be imaginary
    - Cached property
    """

@property
def p(self):
    """
    Total momentum magnitude.

    Returns
    -------
    array_like
        |p| = sqrt(px² + py² + pz²)
        
    Notes
    -----
    - Always non-negative
    - Cached property
    """
```

### Component Properties

```python
@property
def px(self):
    """x-component of momentum (read-only)."""
    return self._px

@property
def py(self):
    """y-component of momentum (read-only)."""
    return self._py

@property
def pz(self):
    """z-component of momentum (read-only)."""
    return self._pz

@property
def E(self):
    """Energy (read-only)."""
    return self._E
```

## Operations

### Arithmetic Operations

```python
def __add__(self, other):
    """
    Add two Lorentz vectors.

    Parameters
    ----------
    other : LVec
        Vector to add

    Returns
    -------
    LVec
        New vector with summed components

    Examples
    --------
    >>> v1 = LVec(1.0, 0.0, 0.0, 2.0)
    >>> v2 = LVec(0.0, 1.0, 0.0, 2.0)
    >>> v3 = v1 + v2
    >>> print(f"pT: {v3.pt:.1f}, mass: {v3.mass:.1f}")
    """

def __sub__(self, other):
    """
    Subtract two Lorentz vectors.

    Parameters
    ----------
    other : LVec
        Vector to subtract

    Returns
    -------
    LVec
        New vector with subtracted components
    """

def __mul__(self, scalar):
    """
    Multiply vector by scalar.

    Parameters
    ----------
    scalar : float, array_like
        Scalar multiplication factor

    Returns
    -------
    LVec
        New scaled vector
    """

def __rmul__(self, scalar):
    """Right multiplication by scalar."""
    return self.__mul__(scalar)
```

### Indexing and Selection

```python
def __getitem__(self, idx):
    """
    Index or mask the vector.

    Parameters
    ----------
    idx : int, slice, array_like
        Index, slice, or boolean mask

    Returns
    -------
    LVec
        New vector with selected components

    Examples
    --------
    >>> v = LVec([1,2,3], [4,5,6], [7,8,9], [10,11,12])
    >>> first = v[0]  # First event
    >>> mask = v.pt > 50  # High-pT selection
    >>> high_pt = v[mask]
    """
```

## Transformations

### Rotations

```python
def rotx(self, angle):
    """
    Rotate vector around x-axis.

    Parameters
    ----------
    angle : float, array_like
        Rotation angle in radians

    Returns
    -------
    LVec
        New rotated vector

    Notes
    -----
    - Energy component remains unchanged
    - Rotation matrix:
      [1      0           0     ]
      [0   cos(θ)   -sin(θ)]
      [0   sin(θ)    cos(θ)]
    """

def roty(self, angle):
    """
    Rotate vector around y-axis.

    Parameters
    ----------
    angle : float, array_like
        Rotation angle in radians

    Returns
    -------
    LVec
        New rotated vector

    Notes
    -----
    - Energy component remains unchanged
    - Rotation matrix:
      [ cos(θ)   0   sin(θ)]
      [   0      1     0   ]
      [-sin(θ)   0   cos(θ)]
    """

def rotz(self, angle):
    """
    Rotate vector around z-axis.

    Parameters
    ----------
    angle : float, array_like
        Rotation angle in radians

    Returns
    -------
    LVec
        New rotated vector

    Notes
    -----
    - Energy component remains unchanged
    - Rotation matrix:
      [cos(θ)   -sin(θ)   0]
      [sin(θ)    cos(θ)   0]
      [  0         0      1]
    """
```

### Lorentz Boosts

```python
def boost(self, bx, by, bz):
    """
    Apply general Lorentz boost.

    Parameters
    ----------
    bx, by, bz : float, array_like
        Boost velocity components (β) in units of c
        Each component should be in range [-1, 1]

    Returns
    -------
    LVec
        New boosted vector

    Notes
    -----
    - Preserves invariant mass
    - γ = 1/sqrt(1 - β²)
    - For numerical stability, treats very small boosts as zero

    Examples
    --------
    >>> v = LVec(0, 0, 0, 1.0)  # Particle at rest
    >>> v_boosted = v.boost(0, 0, 0.5)  # Boost along z with β=0.5
    """

def boostz(self, bz):
    """
    Apply Lorentz boost along z-axis.

    Parameters
    ----------
    bz : float, array_like
        Boost velocity (β) along z-axis in units of c
        Should be in range [-1, 1]

    Returns
    -------
    LVec
        New boosted vector

    Notes
    -----
    - Specialized, faster version of general boost
    - Useful for collider physics where boost is often along beam axis
    """
```

## Conversion Methods

```python
def to_numpy(self):
    """
    Convert to NumPy arrays.

    Returns
    -------
    dict
        Dictionary containing arrays for each component
        {'px': array, 'py': array, 'pz': array, 'E': array}
    """

def to_awkward(self):
    """
    Convert to Awkward arrays.

    Returns
    -------
    dict
        Dictionary containing awkward arrays for each component
        {'px': ak.Array, 'py': ak.Array, 'pz': ak.Array, 'E': ak.Array}

    Raises
    ------
    DependencyError
        If awkward package is not installed
    """

def to_root_dict(self):
    """
    Convert to ROOT-compatible dictionary format.

    Returns
    -------
    dict
        Dictionary with ROOT-style keys
        {'fX': px, 'fY': py, 'fZ': pz, 'fE': E}

    Notes
    -----
    - Compatible with ROOT TLorentzVector convention
    - Useful for writing back to ROOT files
    """

def to_ptepm(self):
    """
    Convert to (pt, eta, phi, mass) representation.

    Returns
    -------
    tuple
        (pt, eta, phi, mass) components
    """
```

## Backend System

The backend system handles the transition between NumPy and Awkward arrays seamlessly.

### Backend Detection and Conversion

```python
# backends.py
def is_ak(x):
    """
    Check if input is an Awkward array.

    Parameters
    ----------
    x : object
        Input to check

    Returns
    -------
    bool
        True if x is an Awkward array
    """

def is_np(x):
    """
    Check if input is a NumPy array.

    Parameters
    ----------
    x : object
        Input to check

    Returns
    -------
    bool
        True if x is a NumPy array
    """

def to_ak(x):
    """
    Convert input to Awkward array.

    Parameters
    ----------
    x : array_like
        Input to convert

    Returns
    -------
    ak.Array
        Converted array

    Raises
    ------
    DependencyError
        If awkward package is not installed
    """

def to_np(x):
    """
    Convert input to NumPy array.

    Parameters
    ----------
    x : array_like
        Input to convert

    Returns
    -------
    np.ndarray
        Converted array
    """
```

### Backend Mathematical Operations

```python
def backend_sqrt(x, lib):
    """
    Compute square root using appropriate backend.

    Parameters
    ----------
    x : array_like
        Input array
    lib : str
        Backend library ('np' or 'ak')

    Returns
    -------
    array_like
        Square root computed with appropriate backend
    """

def backend_sin(x, lib):
    """Sine with backend dispatch."""

def backend_cos(x, lib):
    """Cosine with backend dispatch."""

def backend_sinh(x, lib):
    """Hyperbolic sine with backend dispatch."""

def backend_cosh(x, lib):
    """Hyperbolic cosine with backend dispatch."""

def backend_atan2(y, x, lib):
    """Arctangent2 with backend dispatch."""
```

## Advanced Usage

### Caching System

The LVec class implements an efficient caching system for derived properties:

```python
class LVec:
    def _get_cached(self, key, func):
        """
        Get cached value or compute and cache it.

        Parameters
        ----------
        key : str
            Cache key
        func : callable
            Function to compute value if not cached

        Returns
        -------
        array_like
            Cached or computed value

        Notes
        -----
        - Cache is version-controlled to ensure consistency
        - Cache is cleared when vector is modified
        """

    def touch(self):
        """
        Invalidate cache by incrementing version.
        
        Called automatically when vector is modified.
        """
```

### Working with Batch Data

```python
# Example of batch processing
def analyze_batch(vectors):
    """
    Process multiple vectors efficiently.

    Parameters
    ----------
    vectors : LVec
        Vector with array components

    Returns
    -------
    dict
        Analysis results

    Examples
    --------
    >>> data = uproot.open("events.root")["Events"].arrays()
    >>> vectors = LVec(data["px"], data["py"], data["pz"], data["E"])
    >>> high_pt = vectors[vectors.pt > 50]  # High-pT selection
    >>> masses = high_pt.mass  # Vectorized mass calculation
    """
```

### Integration with HEP Tools

#### Using with Uproot

```python
import uproot
import numpy as np
from lvec import LVec

# Reading from ROOT file
file = uproot.open("events.root")
tree = file["Events"]
data = tree.arrays(["px", "py", "pz", "E"], library="np")

# Create LVec object
vectors = LVec(data["px"], data["py"], data["pz"], data["E"])

# Analysis
mask = vectors.pt > 30
selected = vectors[mask]
```

#### Converting to ROOT Format

```python
vectors = LVec(px, py, pz, E)
root_dict = vectors.to_root_dict()
# Can be written back to ROOT file
with uproot.recreate("output.root") as f:
    f["Events"] = root_dict
```

## Performance Considerations

1. **Caching Strategy**
   - Derived properties are cached on first access
   - Cache is invalidated only when necessary
   - Version control prevents stale cache usage

2. **Memory Management**
   - Large arrays are handled efficiently
   - Backend operations avoid unnecessary copies
   - Lazy evaluation where possible

3. **Optimization Tips**
   ```python
   # Efficient for large datasets
   vectors = LVec(px, py, pz, E)
   masses = vectors.mass  # Cached after first calculation
   
   # Avoid repeated calculations
   pt = vectors.pt  # Store if used multiple times
   
   # Efficient filtering
   mask = (vectors.pt > 20) & (np.abs(vectors.eta) < 2.5)
   selected = vectors[mask]
   ```

