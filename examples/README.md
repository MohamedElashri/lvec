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

## Running the Examples

Run each example individually:
```bash
python 01_basic_reading.py
python 02_decay_reconstruction.py
python 03_advanced_selections.py
python 04_boost_frame.py
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