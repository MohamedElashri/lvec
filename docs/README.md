# LVec Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Backend System](#backend-system)
7. [Advanced Usage](#advanced-usage)
8. [Performance Considerations](#performance-considerations)
9. [Reference Frames](#reference-frames)

## Overview

LVec is a Python package designed for High Energy Physics (HEP) analysis, providing a unified interface for handling Lorentz vectors. It bridges the gap between different HEP ecosystems (Scikit-HEP and ROOT) and supports both NumPy and Awkward array backends.

### Dependencies
- Python 3.10+ (due to numpy requirement)
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
def to_np(self):
    """
    Convert to NumPy arrays.

    Returns
    -------
    dict
        Dictionary containing arrays for each component
        {'px': array, 'py': array, 'pz': array, 'E': array}
    """

def to_ak(self):
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

### Memory Optimization Features

LVec includes a sophisticated caching system with memory optimization features to enhance performance and resource management, particularly valuable when working with large datasets in High Energy Physics analysis.

#### Cache Size Limits with LRU Eviction

You can limit the maximum number of cached properties to prevent memory bloat:

```python
# Create a vector with max cache size of 100 entries
v = LVec(px=1.0, py=2.0, pz=3.0, E=4.0, max_cache_size=100)

# When cache exceeds this limit, least recently used properties are evicted
```

When the cache reaches the size limit, the Least Recently Used (LRU) eviction policy automatically removes the oldest accessed entries to make room for new calculations.

#### Time-To-Live (TTL) Expiration

Control how long cached values remain valid with TTL settings:

```python
# Create a vector with default TTL of 60 seconds for all properties
v = LVec(px=1.0, py=2.0, pz=3.0, E=4.0, default_ttl=60)

# Set different TTL values for specific properties
v.set_ttl('pt', 10)      # Set pt to expire after 10 seconds
v.set_ttl('eta', 30)     # Set eta to expire after 30 seconds
v.set_ttl('mass', None)  # Never expire mass (unless cache size limit is reached)

# Clear all expired cache entries
expired_count = v.clear_expired()
print(f"Removed {expired_count} expired properties")
```

TTL functionality is particularly useful for:
- Ensuring calculations use fresh values in long-running analyses
- Managing memory in workflows where property relevance changes over time
- Forcing recalculation of properties that may have numerical instability

#### Cache Statistics and Monitoring

Monitor cache performance with built-in statistics:

```python
# Access detailed cache statistics
stats = v.cache_stats
print(f"Total cache hits: {sum(prop['hits'] for prop in stats['properties'].values())}")
print(f"Total cache misses: {sum(prop['misses'] for prop in stats['properties'].values())}")

# Get overall cache hit ratio (higher is better)
hit_ratio = v.cache_hit_ratio
print(f"Cache hit ratio: {hit_ratio:.2f}")

# Reset statistics to measure performance of a specific section of code
v.reset_cache_stats()
```

#### Combining Features

You can combine size limits and TTL for comprehensive memory management:

```python
# Create a vector with both memory optimization features
v = LVec(
    px=1.0, py=2.0, pz=3.0, E=4.0,
    max_cache_size=100,   # Limit cache size
    default_ttl=300       # Default 5-minute expiration
)

# Properties will be removed if either:
# 1. They exceed their TTL expiration time
# 2. The cache exceeds max_cache_size (LRU eviction)
```

#### Implementation Details

The caching system tracks:
- Dependencies between properties and vector components
- Access timestamps for TTL expiration
- Access order for LRU eviction
- Hit/miss statistics per property

Cache invalidation happens automatically when vector components are modified, ensuring calculations always use correct values.

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

## Reference Frames

The `Frame` class provides a powerful abstraction for handling reference frame transformations in relativistic calculations.

```python
from lvec import Frame, LVec

# Create a stationary lab frame
lab_frame = Frame.rest(name="lab")

# Create a frame moving with velocity β=(0.5, 0, 0) relative to lab frame
moving_frame = Frame(bx=0.5, by=0.0, bz=0.0, name="moving")

# Create a center-of-mass frame from a collection of particles
particles = [
    LVec(px=1.0, py=0.0, pz=2.0, E=5.0),
    LVec(px=-0.5, py=0.0, pz=-1.0, E=3.0)
]
cm_frame = Frame.center_of_mass(particles, name="cm")

# Access frame properties
print(f"Frame velocity: β=({cm_frame.beta_x}, {cm_frame.beta_y}, {cm_frame.beta_z})")
print(f"Gamma factor: γ={cm_frame.gamma}")
```

### Frame Transformations

LVec provides two methods for transforming vectors between reference frames:

1. **to_frame**: Transform directly to a specific frame
   ```python
   # Transform a vector to the center-of-mass frame
   p_cm = p_lab.to_frame(cm_frame)
   ```

2. **transform_frame**: Transform from one specific frame to another
   ```python
   # Transform a vector from lab frame to moving frame
   p_moving = p_lab.transform_frame(lab_frame, moving_frame)
   ```

The key difference is that `to_frame` assumes the vector is in some unspecified frame and applies a direct boost using the negative of the target frame's velocity, while `transform_frame` explicitly specifies both the source and target frames.

### Physics Validation

A key property of reference frame transformations is that they preserve invariant quantities like mass:

```python
# Verify mass invariance
print(f"Mass in lab frame: {particle.mass}")
print(f"Mass in CM frame: {particle.to_frame(cm_frame).mass}")
# These should be identical
```

In the center-of-mass frame, the total momentum should be zero:

```python
# Get total momentum in CM frame
total_p_cm = sum(p.to_frame(cm_frame) for p in particles)
print(f"Total momentum in CM: ({total_p_cm.px}, {total_p_cm.py}, {total_p_cm.pz})")
# Should be very close to (0, 0, 0)
```
