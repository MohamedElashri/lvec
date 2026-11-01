# LVec examples

Install from the checkout when testing the unreleased changes:

```bash
pip install -e ".[examples]"
cd examples
```

The examples extra supplies Awkward, uproot, and matplotlib. Pure vector examples
can also run with the base NumPy install. Use consistent momentum/energy units
throughout; the package assumes `c=1` and angles in radians.

## Script index

| Script | Purpose | Input |
| --- | --- | --- |
| `01_basic_reading.py` | Read ROOT branches and construct vectors | Local ROOT fixture |
| `02_decay_reconstruction.py` | Add daughter four-vectors and inspect masses | Local ROOT fixture |
| `03_advanced_selections.py` | Apply array selections | Local ROOT fixture |
| `04_boost_frame.py` | Apply an active boost toward a parent rest frame | Local ROOT fixture |
| `05_2d_vectors.py` | 2D components, magnitudes, arithmetic, and dot products | Generated in script |
| `06_3d_vectors.py` | 3D magnitudes, arithmetic, dot and cross products | Generated in script |
| `07_jagged_arrays.py` | Variable-length particle collections | Generated in script |
| `08_lhcb_data.py` | Download ROOT data and plot two-body mass combinations | Network download if absent |
| `09_cache_performance.py` | Inspect cached property access | Generated in script |
| `10_reference_frames.py` | Transform particles and verify a CM system | Generated in script |
| `11_cache_optimization_demo.py` | Demonstrate cache policies | Generated in script |

Run the self-contained vector/frame examples with:

```bash
python 05_2d_vectors.py
python 06_3d_vectors.py
python 07_jagged_arrays.py
python 10_reference_frames.py
```

The ROOT fixture scripts expect `samples/physics_data.root` relative to the
examples directory. To create it:

```bash
mkdir -p samples
python create_test_data.py
python 01_basic_reading.py
```

The legacy fixture is an I/O demonstration: its daughter three-momenta sum to the
stored mother's momentum, but their energies do not sum to the stored mother
energy. Its reconstructed pair is therefore not an energy-conserving Z decay.
For physical two-body decay validation, use the deterministic cases in
[`test_stress_physics.py`](../lvec/tests/test_stress_physics.py). Random fixture
outputs are not fixed numerical reference results.

`08_lhcb_data.py` may download a ROOT file and create plots in the current
working directory. Check the dataset's component units against the script's
energy/mass assumptions before using its histograms for analysis. It is not
part of the offline test suite.

## Current API examples

```python
import numpy as np
from lvec import LVec, Frame, Vector2D, Vector3D

transverse = Vector2D(3., 4.)
rotated_2d = transverse.rotate(np.pi/4)
spatial = Vector3D(1., 2., 3.)
rotated_3d = spatial.rotate(np.pi/2, axis="y")
normal = spatial.cross(Vector3D(0., 0., 1.))

parent = LVec(1., 2., 3., 8.)
cm = Frame.from_lvec(parent, name="cm")
parent_in_cm = parent.to_frame(cm)
np.testing.assert_allclose(parent_in_cm.to_p4()[:3], 0., atol=1e-14)
np.testing.assert_allclose(parent_in_cm.mass, parent.mass)
```

`boost(bx, by, bz)` is active. A `Frame` stores physical lab velocity `+p/E`,
while `to_frame` applies the negative boost. A Lorentz boost preserves invariant
mass within scale-appropriate floating-point tolerances. It does not turn a
91 GeV parent into a 0.22 GeV parent. See the
[migration guide](../docs/migration.md#active-boosts-and-passive-frame-transforms)
for the corrected sign convention.

```python
import awkward as ak
from lvec import LVec

particles = LVec(ak.Array([[1., 2.], [], [3., None]]), 0., 0., 5.)
events = particles.sum(axis=-1)
assert ak.to_list(events.px) == [3., 0., 3.]
assert ak.to_list(events.E) == [10., 0., 5.]
records = particles.to_ak_record()
```

Event sums combine components and then calculate observables; adding particle
masses is not a reconstruction of the total invariant mass. Missing particles
are skipped consistently, empty groups produce zero vectors, and missing events
remain missing when reducing their particle axis.

## Validation and measurements

From the repository root:

```bash
python -m pytest -W error::RuntimeWarning
PYTHONPATH=. python benchmarks/benchmark_core.py --sizes 1000 1000000 --jagged-events 10000
```

Use the [analysis guide](../docs/analysis_features.md) for matching and decay
angles, the [API guide](../docs/README.md) for cache settings, and the
[accuracy guide](../docs/numerical_accuracy.md) for stress tests and recorded
performance costs. Example cache-hit demonstrations do not replace measurements
of uncached arithmetic or complete workflows.
