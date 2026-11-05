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
| Uncached mass | $0.321\pm0.024$ | $0.449\pm0.021$ | $3.360\pm0.130$ | $4.592\pm0.134$ |
| Four-vector normalization | $0.579\pm0.023$ | $2.163\pm0.102$ | $5.217\pm0.210$ | $11.571\pm0.996$ |
| $\beta$ boost | $0.948\pm0.066$ | $1.815\pm0.051$ | $4.935\pm0.307$ | $8.142\pm0.504$ |
| $\gamma$ boost, per-particle parameters | $1.011\pm0.083$ | $1.130\pm0.026$ | $7.702\pm1.423$ | $5.970\pm0.498$ |
| Boost from a shared reference | $1.396\pm0.043$ | $1.852\pm0.074$ | $6.765\pm0.923$ | $7.948\pm0.213$ |
| Boost from per-particle references | $2.078\pm0.078$ | $3.949\pm0.097$ | $11.467\pm0.862$ | $18.665\pm1.396$ |
| Euler rotation, per-particle angles | $1.600\pm0.031$ | $2.490\pm0.084$ | $9.711\pm0.668$ | $11.197\pm0.401$ |
| Quaternion rotation, per-particle parameters | $2.176\pm0.109$ | $2.552\pm0.062$ | $10.419\pm0.577$ | $10.259\pm0.828$ |
| Dense matrix, per-particle coefficients | $1.670\pm0.021$ | $1.860\pm0.038$ | $10.622\pm1.022$ | $8.471\pm0.379$ |
| Add, boost, calculate mass | $1.587\pm0.099$ | $3.224\pm0.184$ | $12.375\pm1.812$ | $16.210\pm1.099$ |

Native jagged construction takes $0.250\pm0.003$ ms for LVec and
$0.250\pm0.003$ ms for vector. Owned LVec construction takes
$1.758\pm0.110$ ms, including copies of numerical buffers.

Jagged $\gamma$ boosts and dense matrices remain slower than vector.
Quaternion rotations are close within the observed dispersion. MAD describes
sample dispersion; these measurements do not give confidence intervals.

## Measurement conditions

The run contains 198 timing rows. Source and framework fingerprints cover
sorted filenames and contents of `lvec/*.py` and `benchmarks/*.py`.

| Item | Value |
| --- | --- |
| Measured source | `c1866cfe6f64a9eb60afad76c9935a7ad5bfd43b` |
| Date | 2026-11-05, Europe/Zurich |
| Environment | Python 3.13.12; NumPy 2.5.4; Awkward 2.14.0; vector 1.9.0 |
| Hardware | Apple M1, 8 logical CPUs, 16 GiB; macOS 27.2 arm64 |
| Sampling | 3 fresh processes, 7 samples each; 2 warmups; 20 ms target per sample |
| Controls | Seed 20261101; numeric runtime threads 1; `PYTHONHASHSEED=0`; affinity/power not controlled |
| Library SHA256 | `fb9bd5d7bf386336aae87fad9d252bbad09670c727bfd3ef53d16c2538c2d251` |
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
