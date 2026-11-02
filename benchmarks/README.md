# Benchmarks

[Overview](../README.md) · [Physics guide](../user-guide.md)

The suite measures construction, observables, transformations, reductions,
and complete analysis workflows. Scikit-HEP vector is an optional comparison.
Each workload checks its result against an independent formula before and
after timing, including the positions of missing values.

## Representative results

**Milliseconds per call, median ± median absolute deviation (MAD).** NumPy
inputs contain 100,000 particles; jagged inputs contain 100,000 events and
401,402 particles. These are fresh calculations on existing collections;
repeated p4 boosts reuse the LVec reference's cached mass.

| Workload | NumPy LVec | NumPy vector | Jagged LVec | Jagged vector |
| --- | ---: | ---: | ---: | ---: |
| Uncached mass | 0.316 ± 0.003 | 0.431 ± 0.038 | 3.269 ± 0.434 | 4.869 ± 0.958 |
| Four-vector normalization | 0.517 ± 0.010 | 1.809 ± 0.072 | 5.164 ± 0.177 | 10.061 ± 1.063 |
| Beta boost | 0.795 ± 0.056 | 1.627 ± 0.068 | 4.523 ± 0.341 | 7.530 ± 0.098 |
| Gamma boost, per-particle parameters | 0.889 ± 0.028 | 0.978 ± 0.046 | 6.754 ± 0.443 | 5.348 ± 0.232 |
| p4 boost, shared reference | 1.336 ± 0.013 | 1.560 ± 0.019 | 6.371 ± 0.295 | 7.133 ± 0.241 |
| p4 boost, per-particle references | 1.768 ± 0.123 | 3.399 ± 0.055 | 10.110 ± 0.315 | 15.104 ± 0.753 |
| Euler rotation, per-particle angles | 1.466 ± 0.052 | 2.122 ± 0.062 | 9.438 ± 0.439 | 10.206 ± 0.208 |
| Quaternion rotation, per-particle parameters | 1.925 ± 0.046 | 2.091 ± 0.041 | 9.431 ± 0.709 | 9.044 ± 0.202 |
| Dense matrix, per-particle coefficients | 1.480 ± 0.062 | 1.609 ± 0.002 | 9.868 ± 0.285 | 8.233 ± 0.605 |
| Add, boost, calculate mass | 1.322 ± 0.036 | 2.674 ± 0.048 | 8.829 ± 0.392 | 14.535 ± 0.478 |

Native jagged record construction measured 0.243 ± 0.001 ms
for LVec versus 0.261 ± 0.013 ms for vector. **Owned**
LVec construction costs 1.668 ± 0.165 ms because it snapshots
numerical buffers. A cached mass read is another distinct boundary.
Compare the path your analysis actually uses.

Parameterized jagged gamma boosts and dense matrices remain slower than vector;
quaternion rotations are close within the observed dispersion. The dense-matrix
fixture measured 9.868 ± 0.285 ms versus vector's
8.233 ± 0.605 ms. MAD is dispersion, not a
confidence interval; results do not establish a universal ordering across
operations or layouts.

## Measurement record

The post-merge run contains 198 timing rows, including the parameterized boost,
quaternion, dense-matrix, and native-record cases added during the review.

| Item | Value |
| --- | --- |
| Measured source | `3cd0fdb18aa2fc0f95752b25157f27d878b25ba1` |
| Date | 2026-11-02, Europe/Zurich |
| Environment | Python 3.13.12; NumPy 2.5.4; Awkward 2.14.0; vector 1.9.0 |
| Hardware | Apple M1, 8 logical CPUs, 16 GiB; macOS 27.2 arm64 |
| Sampling | 3 fresh processes × 7 samples; 2 warmups; calibrated batches targeting 20 ms |
| Controls | Seed 20261101; numeric runtime threads 1; `PYTHONHASHSEED=0`; affinity/power not controlled |
| Library SHA256 | `bfca9a44d3a1a0422cf6b9df0a43730647e08211d728d88a765f02567d56c446` |
| Framework SHA256 | `ed6836eadd688560fe06fdc3136ea6750cebf9a7db22fd042847f7b219308268` |

Inputs, setup and validation are outside timing. Cache clearing is included
in cold cases; `timeit` disables cyclic garbage collection. Case order is
shuffled deterministically. Allocation measurements use a separate worker and
report incremental traced bytes including live outputs, not process RSS.

Fingerprints cover sorted filenames and contents of `lvec/*.py` and
`benchmarks/*.py`. Reports include revisions, dirty state, loaded source path,
versions, invocations, loop counts, and raw samples. The
[earlier optimization study](https://github.com/MohamedElashri/lvec/blob/afc384c75cc6e236d632926e6d9c03fadd5c90b5/benchmarks/README.md#kernel-optimization-update)
retains historical tables and allocation results. The current measurements also
describe development code, not a released wheel.

## Post-merge comparison

Three alternating before/after rounds compare merged revision `6af876f` with
`3cd0fdb`, using the same framework and 100,000-particle/event inputs.
Each cell is the median of three within-round ratios of call medians:
**final time / baseline time**, so smaller is faster. Each run uses one fresh
process, seven samples, and no allocation measurement.

| Workload | NumPy ratio | Jagged ratio |
| --- | ---: | ---: |
| Gamma boost, per-particle parameters | 0.654 | 0.417 |
| p4 boost, shared reference | 0.284 | 0.272 |
| p4 boost, per-particle references | 0.439 | 0.327 |
| Quaternion rotation, per-particle parameters | 0.700 | 0.266 |
| Diagonal matrix, per-particle coefficients | 1.021 | 0.881 |
| Add, boost, calculate mass | 1.077 | 0.967 |
| vector workflow control | 1.032 | 1.004 |

The large boost/quaternion gains repeat across rounds. Smaller differences in
unchanged workflows and controls vary between rounds and need a quieter host
before being treated as regressions or improvements. Numerical validation also
passes strided, missing, deep, heavy-tail, and extreme inputs.

## Reproduction

Use Python 3.13.12 and a clean checkout of the measured revision for the table
above. To evaluate the current source, use the same commands in that checkout:

```bash
python -m pip install -e ".[test,benchmark]" numpy==2.5.4 awkward==2.14.0 vector==1.9.0
python -m benchmarks --check --layouts numpy jagged missing deep --sizes 1 17
python -m benchmarks --layouts numpy jagged --sizes 1 100000 --compare-vector \
  --processes 3 --samples 7 --output /tmp/lvec-benchmarks.md
python -m benchmarks --layouts extreme --sizes 1000 10000 \
  --output /tmp/lvec-precision.md
```

The default NumPy suite runs offline without vector or Awkward. For a quick
local check, use `--processes 1 --samples 3 --min-time .005 --no-memory`.
For revision comparisons, invoke one fixed framework with `--source` selecting
each clean checkout; alternate their order and retain separate reports.
The alternating comparison used:

```bash
python -m benchmarks --source /path/to/revision --layouts numpy jagged \
  --sizes 100000 --compare-vector --processes 1 --samples 7 --no-memory \
  --cases construct_mass mass_uncached add boost boost_gamma boost_gamma_array \
  boost_p4 boost_p4_array matrix4d matrix4d_array matrix4d_dense_array rotate \
  rotate_euler rotate_euler_array rotate_quaternion_array workflow vector.workflow \
  --output /tmp/lvec-revision.md
```

## Workload selection

Use exact names with `--cases`. Selections must exist for every chosen layout.

| Cases | Timed boundary |
| --- | --- |
| `construct`, `construct_mass` | Public owned construction, optionally followed by mass; `construct_mass` is enabled with `--compare-vector` |
| `mass_cold`, `mass_hot`, `mass_uncached` | Clear then read; cached read; no retention |
| `properties`, `properties_intermediates` | Clear then read pt/eta/phi/mass; optional retained intermediates |
| `add`, `boost`, `rotate`, `sum`, `workflow` | Addition; beta=(.1,.05,.02); z rotation=.47; component reduction; add→boost→mass |
| `unit_cold`, `boost_gamma`, `matrix4d` | Clear then normalize; x boost with beta-equivalent .2; diag(2,3,4,5) matrix |
| `rotate_axis`, `rotate_euler` | Shared axis=(1,1,1), angle=2pi/3; zxz angles=(.2,.3,.4) |
| `rotate_axis_array`, `rotate_euler_array` | Angle=.2+.01*px; axis=(px+2,py+3,pz+4) or Euler theta=.3, psi=.4 |
| `rotate_axis_varying`, `rotate_euler_varying` | 257 pre-generated scalar orientations, exceeding the 128-entry coefficient caches |
| `boost_gamma_array`, `boost_p4`, `boost_p4_array` | Per-particle gamma=1+.1*abs(px); shared p4=(.1,.2,.3,2); or each input p4 as its own active boost reference |
| `rotate_quaternion_array`, `matrix4d_dense_array` | Unit z quaternion for angle=.1+.01*px; or diagonal 1+.01*px with off-diagonal .1 |
| `matrix4d_array` | Per-particle diagonal factor 1+.01*px |
| `record_mass`, `record_construct`, `record_construct_mass`, `record_rotate_euler`, `record_boost` | Native Awkward mass, Cartesian view construction, construction plus mass, Euler rotation, or beta boost |
| `cancellation_mass2`, `mixed_boost_0/1/100` | Exact-invariant cancellation or 0/1/100 exceptional boost entries; `extreme` layout only |

`vector.*` counterparts are available on NumPy/jagged layouts for construction,
mass, workflow, boost, normalization, gamma boost, matrices, and arbitrary/Euler
rotation variants. They recompute properties rather than use a retained cache.

Ordinary inputs have standard-normal momenta and mass 2. `numpy` and `strided`
sizes count particles; jagged, missing, deep, and heavy-tail sizes count events.
Jagged multiplicities are 0–8; heavy-tail uses capped Pareto multiplicities.
Strided inputs probe ingestion into owned snapshots. Flat sums group four
particles; jagged sums reduce the innermost axis. Size-one
jagged input is an **empty event**. `extreme` is capped at 10,000 particles and
checked against Fraction/400-digit Decimal references; exceptional counts are
capped by batch size. See
[cases.py](cases.py) for complete fixture definitions.

Reports are Markdown; no result JSON files are written. Keep generated reports
and source archives outside Git. Numerical correctness is tested independently
of performance; see [contributor checks](../CONTRIBUTING.md#development-checks).
