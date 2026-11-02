# Benchmarks

[Overview](../README.md) · [Physics guide](../user-guide.md)

The suite measures construction, observables, transformations, reductions,
and complete analysis workflows. Scikit-HEP vector is an optional comparison.
Each workload checks its result against an independent formula before and
after timing, including the positions of missing values.

## Representative results

**Milliseconds per call, median ± median absolute deviation (MAD).** NumPy
inputs contain 100,000 particles; jagged inputs contain 100,000 events and
401,402 particles. These are fresh calculations on existing collections.

| Workload | NumPy LVec | NumPy vector | Jagged LVec | Jagged vector |
| --- | ---: | ---: | ---: | ---: |
| Uncached mass | 0.315 ± 0.026 | 0.387 ± 0.017 | 3.049 ± 0.348 | 3.730 ± 0.087 |
| Four-vector normalization | 0.519 ± 0.012 | 1.878 ± 0.047 | 4.697 ± 0.573 | 9.583 ± 0.388 |
| Beta boost | 0.796 ± 0.010 | 1.620 ± 0.017 | 4.687 ± 0.280 | 7.381 ± 0.086 |
| Euler rotation, per-particle angles | 1.479 ± 0.012 | 2.159 ± 0.042 | 8.541 ± 0.706 | 9.846 ± 0.057 |
| Add, boost, calculate mass | 1.342 ± 0.035 | 2.782 ± 0.059 | 9.611 ± 0.415 | 13.188 ± 0.313 |

Native jagged record construction is at parity: 0.244 ± 0.001 ms for LVec
versus 0.245 ± 0.001 ms for vector. **Owned** LVec construction costs
1.505 ± 0.048 ms because it snapshots numerical buffers. A cached mass read
is another distinct boundary. Compare the path your analysis actually uses.

These fixtures favor LVec; others are close. The jagged varying-axis rotation
measured 4.699 ± 0.292 ms versus vector's 4.512 ± 0.339 ms. MAD is dispersion,
not a confidence interval. Results on one workstation do not establish a
universal ordering across operations or layouts.

## Measurement record

| Item | Value |
| --- | --- |
| Measured source | `a270b7897c3b60132dcccc7a7300f3d7ca85556f` |
| Date | 2026-11-02, Europe/Zurich |
| Environment | Python 3.13.12; NumPy 2.5.4; Awkward 2.14.0; vector 1.9.0 |
| Hardware | Apple M1, 8 logical CPUs, 16 GiB; macOS 27.2 arm64 |
| Sampling | 3 fresh processes × 7 samples; 2 warmups; calibrated batches targeting 20 ms |
| Controls | Seed 20261101; numeric runtime threads 1; `PYTHONHASHSEED=0`; affinity/power not controlled |
| Library SHA256 | `06d4f7045da86e31b765c3d174fb784ac200470e42f139555f2b7220582e3278` |
| Framework SHA256 | `c3333ef50d492ebf927ba62f25d7f2623280b070cbcc1f92f893f045605d844f` |

Inputs, setup and validation are outside timing. Cache clearing is included
in cold cases; `timeit` disables cyclic garbage collection. Case order is
shuffled deterministically. Allocation measurements use a separate worker and
report incremental traced bytes including live outputs, not process RSS.

Fingerprints cover sorted filenames and contents of `lvec/*.py` and
`benchmarks/*.py`. Reports include revisions, dirty state, loaded source path,
versions, invocations, loop counts, and raw samples. The
[complete recorded study](https://github.com/MohamedElashri/lvec/blob/afc384c75cc6e236d632926e6d9c03fadd5c90b5/benchmarks/README.md#kernel-optimization-update)
retains the full tables, allocation results, alternating revision checks, and
historical measurements. It describes development code, not a released wheel.

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
| `matrix4d_array` | Per-particle diagonal factor 1+.01*px |
| `record_mass`, `record_construct`, `record_construct_mass` | Native Awkward mass or Cartesian view construction, optionally followed by mass |
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
