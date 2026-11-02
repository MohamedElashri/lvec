# Benchmarks

[Overview](../README.md) · [Physics guide](../user-guide.md) · [Contributing](../CONTRIBUTING.md)

The suite measures construction, observables, transformations, reductions,
and analysis workflows. Each case checks its output against an independent
formula before and after timing. Scikit-HEP vector is an optional comparison.

## Results

Times are milliseconds per call, reported as median $\pm$ median absolute
deviation (MAD). NumPy inputs contain 100,000 particles; jagged inputs contain
100,000 events and 401,402 particles. Calculations use existing collections;
reference boosts reuse the LVec reference's cached mass.

| Workload | NumPy LVec | NumPy vector | Jagged LVec | Jagged vector |
| --- | ---: | ---: | ---: | ---: |
| Uncached mass | $0.316\pm0.003$ | $0.431\pm0.038$ | $3.269\pm0.434$ | $4.869\pm0.958$ |
| Four-vector normalization | $0.517\pm0.010$ | $1.809\pm0.072$ | $5.164\pm0.177$ | $10.061\pm1.063$ |
| $\beta$ boost | $0.795\pm0.056$ | $1.627\pm0.068$ | $4.523\pm0.341$ | $7.530\pm0.098$ |
| $\gamma$ boost, per-particle parameters | $0.889\pm0.028$ | $0.978\pm0.046$ | $6.754\pm0.443$ | $5.348\pm0.232$ |
| Boost from a shared reference | $1.336\pm0.013$ | $1.560\pm0.019$ | $6.371\pm0.295$ | $7.133\pm0.241$ |
| Boost from per-particle references | $1.768\pm0.123$ | $3.399\pm0.055$ | $10.110\pm0.315$ | $15.104\pm0.753$ |
| Euler rotation, per-particle angles | $1.466\pm0.052$ | $2.122\pm0.062$ | $9.438\pm0.439$ | $10.206\pm0.208$ |
| Quaternion rotation, per-particle parameters | $1.925\pm0.046$ | $2.091\pm0.041$ | $9.431\pm0.709$ | $9.044\pm0.202$ |
| Dense matrix, per-particle coefficients | $1.480\pm0.062$ | $1.609\pm0.002$ | $9.868\pm0.285$ | $8.233\pm0.605$ |
| Add, boost, calculate mass | $1.322\pm0.036$ | $2.674\pm0.048$ | $8.829\pm0.392$ | $14.535\pm0.478$ |

Native jagged construction takes $0.243\pm0.001$ ms for LVec and
$0.261\pm0.013$ ms for vector. Owned LVec construction takes
$1.668\pm0.165$ ms, including copies of numerical buffers.

Jagged $\gamma$ boosts and dense matrices remain slower than vector.
Quaternion rotations are close within the observed dispersion. MAD describes
sample dispersion; these measurements do not give confidence intervals.

## Measurement conditions

The run contains 198 timing rows. Source and framework fingerprints cover
sorted filenames and contents of `lvec/*.py` and `benchmarks/*.py`.

| Item | Value |
| --- | --- |
| Measured source | `3cd0fdb18aa2fc0f95752b25157f27d878b25ba1` |
| Date | 2026-11-02, Europe/Zurich |
| Environment | Python 3.13.12; NumPy 2.5.4; Awkward 2.14.0; vector 1.9.0 |
| Hardware | Apple M1, 8 logical CPUs, 16 GiB; macOS 27.2 arm64 |
| Sampling | 3 fresh processes, 7 samples each; 2 warmups; 20 ms target per sample |
| Controls | Seed 20261101; numeric runtime threads 1; `PYTHONHASHSEED=0`; affinity/power not controlled |
| Library SHA256 | `bfca9a44d3a1a0422cf6b9df0a43730647e08211d728d88a765f02567d56c446` |
| Framework SHA256 | `ed6836eadd688560fe06fdc3136ea6750cebf9a7db22fd042847f7b219308268` |

Setup and validation are outside timing. Cold cases include cache clearing;
`timeit` disables cyclic garbage collection. Case order is shuffled with a
fixed seed. A separate worker measures incremental traced allocations,
including live outputs. These are allocation measurements, not process RSS.

## Reproduction

Use Python 3.13.12 and a clean checkout of the measured revision:

```bash
python -m pip install -e ".[test,benchmark]" numpy==2.5.4 awkward==2.14.0 vector==1.9.0
python -m benchmarks --check --layouts numpy jagged missing deep --sizes 1 17
python -m benchmarks --layouts numpy jagged --sizes 1 100000 --compare-vector \
  --processes 3 --samples 7 --output /tmp/lvec-benchmarks.md
python -m benchmarks --check --layouts strided missing deep heavy-tail --sizes 1 17 1000
python -m benchmarks --check --layouts extreme --sizes 1 17 1000 10000
```

The default NumPy suite runs without vector or Awkward. For a quick timing run,
use `--processes 1 --samples 3 --min-time .005 --no-memory`.
For revision comparisons, keep the framework fixed and alternate clean source
checkouts with `--source`. The comparison above used:

```bash
python -m benchmarks --source /path/to/revision --layouts numpy jagged \
  --sizes 100000 --compare-vector --processes 1 --samples 7 --no-memory \
  --cases construct_mass mass_uncached add boost boost_gamma boost_gamma_array \
  boost_p4 boost_p4_array matrix4d matrix4d_array matrix4d_dense_array rotate \
  rotate_euler rotate_euler_array rotate_quaternion_array workflow vector.workflow \
  --output /tmp/lvec-revision.md
```

Reports record revisions, dirty state, versions, invocations, loop counts,
fingerprints, allocations, and raw samples. Keep generated Markdown reports
and source archives outside Git; no result JSON files are written.

## Fixtures and timing boundaries

Use `--cases` with names from [cases.py](cases.py); a selection must exist for
every chosen layout. `mass_cold` clears the cache, `mass_hot` reads a cached
value, and `mass_uncached` calculates without retention. Native `record_*`
cases use views; `construct` copies inputs. `vector.*` cases calculate
properties without a retained cache.

Ordinary inputs have standard-normal momenta and mass $m=2$ in consistent
units. NumPy and strided sizes count particles; other layouts count events.
Jagged multiplicities range from zero to eight; size one is an empty event.
Heavy-tail inputs use capped Pareto multiplicities. Flat sums group four
particles; jagged sums reduce the innermost axis.

| Case | Parameters |
| --- | --- |
| `boost`, `boost_gamma` | $\boldsymbol\beta=(0.1,0.05,0.02)$; axis boost equivalent to $\beta_x=0.2$ |
| `boost_gamma_array` | $\gamma=1+0.1\lvert p_x\rvert$ |
| `boost_p4`, `boost_p4_array` | Shared $(p_x,p_y,p_z,E)=(0.1,0.2,0.3,2)$; or each particle as its own active reference |
| `rotate_euler_array` | $(\phi,\theta,\psi)=(0.2+0.01p_x,0.3,0.4)$ in the documented Euler convention |
| `rotate_quaternion_array` | $q=(\cos a,0,0,\sin a)$ with $a=0.1+0.01p_x$; rotation angle $2a$ |
| `matrix4d_array`, `matrix4d_dense_array` | $M_{ii}=1+0.01p_x$; $M_{ij}=0$ or $0.1$ for $i\ne j$ |

Extreme cases use Fraction and 400-digit Decimal references, with at most
10,000 particles. Physics and ownership tests run separately from timing;
see [development checks](../CONTRIBUTING.md#development-checks).
