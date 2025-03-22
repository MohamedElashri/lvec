# Examples - LVec Package

This directory contains example scripts demonstrating how to use the LVec package for High Energy Physics analysis.

## Prerequisites
Make sure you have LVec installed along with its dependencies:
```bash
pip install lvec
pip install uproot numpy awkward
```

## Data Sample
The examples use a simulated Z→μμ decay sample. To generate the sample:

```bash
python create_test_data.py
```

This will create `samples/physics_data.root` containing:
- Mother particle (Z boson): px, py, pz, E
- Daughter 1 (μ⁺): px, py, pz, E
- Daughter 2 (μ⁻): px, py, pz, E

## Available Examples

### 1. Basic Reading (`01_basic_reading.py`)
Demonstrates how to:
- Read ROOT files using uproot
- Create LVec objects from branches
- Access basic kinematic properties

```python
from lvec import LVec
mother = LVec(data["m_px"], data["m_py"], data["m_pz"], data["m_E"])
print(f"Average pt: {np.mean(mother.pt):.2f} GeV")
```

### 2. Decay Reconstruction (`02_decay_reconstruction.py`)
Shows how to:
- Handle multiple particles
- Perform vector addition
- Calculate derived quantities
- Validate reconstructed masses

```python
reconstructed = daughter1 + daughter2
print(f"Original mass: {np.mean(mother.mass):.2f} GeV")
print(f"Reconstructed mass: {np.mean(reconstructed.mass):.2f} GeV")
```

### 3. Physics Selections (`03_advanced_selections.py`)
Demonstrates:
- Making physics selections (pt, eta cuts)
- Applying masks to vectors
- Calculating derived quantities for selected events

```python
mask = (muon1.pt > 20) & (muon2.pt > 20) & \
       (np.abs(muon1.eta) < 2.4) & (np.abs(muon2.eta) < 2.4)
muon1_selected = muon1[mask]
```

### 4. Reference Frames (`04_boost_frame.py`)
Shows advanced operations:
- Calculating boost vectors
- Performing Lorentz boosts
- Working in different reference frames
- Validating frame transformations

```python
beta_x = -Z.px/Z.E
muon1_rest = muon1.boost(beta_x, beta_y, beta_z)
```

### 5. 2D Vectors (`05_2d_vectors.py`)
Demonstrates operations with 2D vectors:
- Creating and manipulating 2D vectors
- Calculating angles and rotations in 2D
- Vector addition and scalar multiplication
- Computing dot products and cross products

```python
# Create 2D vectors
vec1 = Vector2D(x1, y1)
vec2 = Vector2D(x2, y2)

# Calculate angle between vectors
angle = vec1.angle(vec2)
```

### 6. 3D Vectors (`06_3d_vectors.py`)
Shows how to work with 3D vectors:
- Basic 3D vector operations
- 3D rotations and transformations
- Vector algebra in 3D space
- Spatial geometry calculations

```python
# Create and manipulate 3D vectors
vec3d = Vector3D(x, y, z)
rotated = vec3d.rotate(theta, axis)
```

### 7. Jagged Arrays handling (`07_jagged_arrays.py`)
Shows how to:
- Work with jagged arrays in LVec
- Handle variable-length datasets

### 8. LHCb Analysis (`08_lhcb_data.py`)
Demonstrates how to:
- Work with real LHCb open data (B→hhh decay)
- Calculate two-body invariant masses
- Create publication-quality plots with LHCb style
- Handle multiple particle combinations

```python
# Create Lorentz vectors for each particle
h1 = LVec(h1_px, h1_py, h1_pz, calculate_energy(h1_px, h1_py, h1_pz))
h2 = LVec(h2_px, h2_py, h2_pz, calculate_energy(h2_px, h2_py, h2_pz))

# Calculate two-body invariant masses
m12 = (h1 + h2).mass  # Invariant mass of particles 1 and 2
```

### 9. Cache Performance (`09_cache_performance.py`)
Demonstrates how to:
- Analyze cache hit ratio
- Measure performance improvements due to caching

```python
# Measure cache hit ratio
cache_hit_ratio = vec.cache_hit_ratio
print(f"Cache hit ratio: {cache_hit_ratio:.2f}")
```

#### Dependencies
Additional dependencies required for this example:
```bash
pip install uproot matplotlib
```

#### Data
Uses LHCb open data from:
- Dataset: B2HHH_MagnetDown.root (just a sample)
- Source: [CERN Open Data](https://opendata.cern.ch/record/4900)
- Description: B→hhh decay data collected by LHCb in 2011 at √s = 7 TeV

#### Running the Example
```bash
python lhcb_data.py
```

This will:
1. Download the LHCb data file
2. Calculate two-body invariant masses
3. Create publication-style mass distribution plots
4. Save plots as 'mass_distributions.pdf'

## Running the Examples

Run each example individually:
```bash
python 01_basic_reading.py
python 02_decay_reconstruction.py
python 03_advanced_selections.py
python 04_boost_frame.py
python 08_lhcb_data.py
python 09_cache_performance.py
```

## Expected Output

### Basic Reading
```
Mother particle properties:
Average pt: 60.89 GeV
Average mass: 91.20 GeV
Average eta: 1.58
```

### Decay Reconstruction
```
Decay reconstruction validation:
Original mass: 91.20 GeV
Reconstructed mass: 0.22 GeV
Mass resolution: 0.017 GeV

Average ΔR between daughters: 0.000
```

### Physics Selections
```
Selection results:
Total events: 1000
Selected events: 625

Selected Z properties:
Mass mean: 0.22 ± 0.02 GeV
pT mean: 75.25 ± 15.55 GeV
```

### Reference Frames
```
Rest frame validation:
Original Z pT: 60.89 GeV
Boosted Z pT: 20.37 GeV
Original Z mass: 91.20 GeV
Boosted Z mass: 0.22 GeV

Mean cos(theta) in rest frame: -0.007
```

### 2D Vectors

```
=== 2D Vector Operations with NumPy Backend ===

Vector components (x, y): ([1. 2. 3.], [4. 5. 6.])
Magnitude: [4.12310563 5.38516481 6.70820393]
Azimuthal angle (phi): [1.32581766 1.19028995 1.10714872]

Vector Operations:
v1 + v2: ([6. 8.], [10. 12.])
v1 · v2 (dot product): [26. 44.]

=== 2D Vector Operations with Awkward Backend ===

Vector components (x, y): ([1, 2, 3], [4, 5, 6])
Magnitude: [4.12, 5.39, 6.71]
Azimuthal angle (phi): [1.33, 1.19, 1.11]

Vector Operations:
v1 + v2: ([6, 8], [10, 12])
v1 · v2 (dot product): [26, 44]
```

### 3D Vectors

```
=== 3D Vector Operations with NumPy Backend ===

Vector components (x, y, z): ([1. 2. 3.], [4. 5. 6.], [7. 8. 9.])
Magnitude: [ 8.1240384   9.64365076 11.22497216]
Cylindrical radius (rho): [4.12310563 5.38516481 6.70820393]
Azimuthal angle (phi): [1.32581766 1.19028995 1.10714872]
Polar angle (theta): [0.5323032  0.59247462 0.64052231]

Vector Operations:
v1 + v2: ([ 8. 10.], [12. 14.], [16. 18.])
v1 · v2 (dot product): [ 89. 128.]
v1 × v2 (cross product): ([-12. -12.], [24. 24.], [-12. -12.])

=== 3D Vector Operations with Awkward Backend ===

Vector components (x, y, z): ([1, 2, 3], [4, 5, 6], [7, 8, 9])
Magnitude: [8.12, 9.64, 11.2]
Cylindrical radius (rho): [4.12, 5.39, 6.71]
Azimuthal angle (phi): [1.33, 1.19, 1.11]
Polar angle (theta): [0.532, 0.592, 0.641]

Vector Operations:
v1 + v2: ([8, 10], [12, 14], [16, 18])
v1 · v2 (dot product): [89, 128]
v1 × v2 (cross product): ([-12, -12], [24, 24], [-12, -12])
```

### LHCb Analysis

The output will be a PDF file named `mass_distributions.pdf` containing three mass distribution plots:

```
Two-body invariant mass statistics:
m12 mean: 1803.56 GeV
m23 mean: 2513.31 GeV
m13 mean: 2951.84 GeV

Mass distribution plots have been saved as 'mass_distributions.pdf'

```
This is the plot you should get is:

![Mass Distributions](plots/mass_distributions.png)

## Additional Usage Tips

1. Working with different backends:
```python
# NumPy arrays
data_np = tree.arrays(branches, library="np")
vec_np = LVec(data_np["px"], data_np["py"], data_np["pz"], data_np["E"])

# Awkward arrays
data_ak = tree.arrays(branches, library="ak")
vec_ak = LVec(data_ak["px"], data_ak["py"], data_ak["pz"], data_ak["E"])
```

2. Caching behavior:
```python
# First access calculates and caches
pt = vec.pt
# Second access uses cached value
pt_again = vec.pt
```

3. Performing transformations:
```python
# Rotations
rotated = vec.rotz(np.pi/4)  # 45-degree rotation around z
rotated = vec.rotx(angle)    # rotation around x
rotated = vec.roty(angle)    # rotation around y

# Boosts
boosted = vec.boostz(0.5)    # boost along z with β=0.5
boosted = vec.boost(bx, by, bz)  # general boost
```

## Contributing New Examples
If you have interesting use cases, consider contributing:
1. Create a new Python file in the examples directory
2. Follow the naming convention: `XX_descriptive_name.py`
3. Include detailed comments and documentation
4. Demonstrate practical physics use cases
5. Submit a pull request

For more information, see the main [README](../README.md) or open an issue on GitHub.