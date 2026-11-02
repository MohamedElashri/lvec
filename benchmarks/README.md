# LVec benchmark methods and results

This suite characterizes LVec construction, arithmetic, transformations,
reductions, cache behavior, traced allocations, and numerical recovery on
synthetic inputs. Scikit-HEP `vector` is an optional reference for selected
ordinary workloads. The default suite needs NumPy only and runs offline.

[Reproduction](#reproduction) · [Experimental setup](#experimental-setup)
· [Workloads](#workloads) · [Results](#results)
· [Secondary vector comparison](#secondary-vector-comparison)
· [Core parity update](#core-parity-update)
· [Interpretation and validation](#interpretation-and-validation)

## Reproduction

Run from the repository root. To reproduce the recorded snapshot, first check
out commit `a6a3236eb107a2916428c6af51bfefd82c7da6d0` in a dedicated checkout and
use Python 3.13.12 with these dependency versions:

```bash
python -m pip install -e ".[test]" numpy==2.5.4 awkward==2.14.0 vector==1.9.0
python -m benchmarks --check --layouts numpy strided jagged missing deep heavy-tail extreme --sizes 1 17
python -m benchmarks --check --layouts numpy jagged --sizes 1 17 --compare-vector
```

The recorded runs used the default timing settings and the following selections.
Add `--output /tmp/run-name.md` to each command to retain its full report.

```bash
python -m benchmarks --layouts numpy --sizes 1000 100000 1000000
python -m benchmarks --layouts jagged --sizes 1000 100000
python -m benchmarks --layouts strided --sizes 1000 100000
python -m benchmarks --layouts missing deep heavy-tail --sizes 10000
python -m benchmarks --layouts extreme --sizes 1000 10000
python -m benchmarks --layouts numpy jagged --sizes 100000 --compare-vector --cases construct_mass vector.construct_mass mass_uncached vector.mass workflow vector.workflow
```

For a current development run, install `.[benchmark,test]` and use
`python -m benchmarks`. A short check is available with `--check`; a quick timing
run can use `--processes 1 --samples 3 --min-time .005`. `--cases` takes exact
names that must apply to every selected layout. `--no-memory` omits allocation
measurement. Progress goes to stderr; the Markdown report goes to stdout or
`--output`. No JSON result files are written.

## Experimental setup

The tables below summarize **130 measured rows** from **2026-11-01**. Each timing
row pools 21 samples from three fresh interpreter processes. These are results
for a development checkout with version metadata 0.2.0; released wheels may
contain different code.

| Item | Recorded value |
| --- | --- |
| Source revision | `a6a3236eb107a2916428c6af51bfefd82c7da6d0` (clean tree) |
| Python | 3.13.12 |
| NumPy / Awkward / vector | 2.5.4 / 2.14.0 / 1.9.0 |
| CPU / logical CPUs | Apple M1 / 8 |
| Installed memory | 16 GiB |
| OS / architecture | macOS 27.2 / arm64 |
| Seed | 20261101 |
| Processes / samples / warmups | 3 per layout-size / 7 per case-process / 2 per case |
| Target sample duration | 20 ms, after adaptive calibration |
| Numeric runtime threads | 1 for OMP, OpenBLAS, MKL, Accelerate, and NumExpr |
| CPU affinity / power control | Not pinned / not controlled |

CPU model and memory were read with `sysctl -n machdep.cpu.brand_string hw.memsize`.
The library source SHA256 is
`d0c5dcb1f6c62a8bc2e0109f1b360f181eb66ba283c70fed780800416b93ad36`;
the framework SHA256 is
`630f43890e91367741d8a442f649adf81c2fdde34727230b2f5e86ce3ae905d6`.
The library fingerprint covers the top-level `lvec/*.py` files; the framework
fingerprint covers `benchmarks/*.py`. Each digest includes sorted filenames and
contents. Reports also record the executable, source path, Git dirty state,
thread settings, timestamp, loop counts, invocation, and raw samples.

### Measurement protocol

1. Generate seeded input outside timing. Validate an operation before measurement.
2. Allocate only the current case's collection; release it before the next case.
3. Warm up, then calibrate a batch toward 20 ms, capped at one million calls.
   Slow operations can use one call per sample. Record seven samples and validate
   the result again. `timeit` disables cyclic garbage collection during timing.
4. Repeat in three fresh processes per layout-size cell. Case order is shuffled
   deterministically using the seed and process index.
5. Measure allocations in an additional process after timing, with tracing
   started after input creation and warmup. Keep the operation result alive while
   reading retained and peak traced bytes.

Times are **per complete call**. Input generation, setup, reference calculations,
and validation are excluded. Clearing the cache is included where stated in the
case definition. Very short calls include Python-call overhead. Loop counts can
differ between processes; full reports preserve them.

Tables show the **median ± median absolute deviation (MAD)** of pooled samples.
MAD describes observed dispersion, not a confidence interval or standard error.
Some large NumPy allocation cases showed substantial variation on this interactive
workstation. Thread limits do not set CPU affinity, clock frequency, or background
load. For close comparisons, alternate revisions and repeat on target hardware.

## Workloads

Ordinary components are float64, with independent standard-normal three-momenta
and `E = sqrt(px² + py² + pz² + 4)`, giving mass 2 up to floating-point rounding.
Default LVec cache policies are used unless stated otherwise.

| Case | Timed boundary |
| --- | --- |
| `construct` | Public construction from existing components, including owned snapshots |
| `mass_cold` | Clear an existing object's cache, then calculate mass |
| `mass_hot` | Read an existing mass from a dedicated object's populated cache |
| `mass_uncached` | Calculate mass on an existing object with `max_cache_size=0` |
| `properties` | Clear cache; read pt, eta, phi, and mass |
| `properties_intermediates` | Same sequence with `cache_intermediates=True` |
| `add` | Add an existing collection to itself |
| `boost` | Boost by beta `(0.1, 0.05, 0.02)`, returning a new collection |
| `rotate` | Rotate about z by 0.47 radians, returning a new collection |
| `sum` | Reduce four-momenta over the groups described below |
| `workflow` | Add a collection to itself, boost, and calculate the result's mass |
| `unit_cold` | Clear the input cache; normalize the four-vector with the fused invariant kernel |
| `rotate_axis` | Rotate by 2pi/3 about (1,1,1), independently checked against the cyclic Cartesian permutation |
| `rotate_euler` | ROOT/vector zxz Euler rotation by (0.2,0.3,0.4), checked with independently composed matrices |
| `boost_gamma` | X-axis boost with gamma=1/sqrt(1-0.2²), checked against an independent beta formula |
| `matrix4d` | General transformation API with diagonal matrix diag(2,3,4,5); this is not a norm-preserving matrix |
| `record_mass` | Uncached mass calculation through native Awkward record behavior; Awkward layouts only |

The final six cases were added with core numerical parity. Their `vector.*`
counterparts, where applicable, are opt-in through `--compare-vector` on NumPy
and jagged layouts. Native `record_mass` compares naturally with `vector.mass`;
owned `mass_hot` has a populated cache and represents a different boundary.

| Layout | Size means | Input definition |
| --- | --- | --- |
| `numpy` | Particles | Flat component arrays; reductions use groups of four |
| `strided` | Particles | Noncontiguous views for all four constructor inputs |
| `jagged` | Events | Multiplicities 0–8; first event is empty |
| `missing` | Events | Jagged input with matching particle/event option masks, approximately 10% missing at each level |
| `deep` | Events | Jagged input with two additional singleton list levels |
| `heavy-tail` | Events | Integer Pareto(1.5) multiplicities, capped at 128, including empty events |
| `extreme` | Particles | Separate cancellation and exceptional-boost inputs; capped at 10,000 particles |

Flat reductions pad a final short group with zeros outside timing. Awkward
reductions use the innermost axis, skipping missing particles and preserving
missing events. Generated and visible particle counts are both reported.
Different event layouts therefore do not represent equal amounts of work.
LVec snapshots public inputs: strided arithmetic operates on owned components,
so that layout primarily tests ingestion of noncontiguous data.

The `extreme` suite contains:

- `cancellation_mass2`: clear cache and evaluate `(1e200, 1, 0, 1e200)`, whose
  invariant is exactly -1 using rational products of the supplied binary floats.
- `mixed_boost_0`, `mixed_boost_1`, `mixed_boost_100`: boost ordinary batches with
  0, 1, or up to 100 exceptional entries `(1e308, 1, 0, 1e308)` by beta
  `(-0.9, 0, 0)`. A 400-digit Decimal reference checks exceptional outputs.
  Counts shrink and deduplicate for small batches.

These adversarial numerical inputs exercise recovery paths; they are not a
representative particle-energy distribution. Ordinary checks use independent
NumPy/Awkward formulas for components and observables and verify the full visible
list and option structure. RuntimeWarnings, failed checks, worker errors, and
timeouts fail the command rather than producing a completed report.

## Results

### Dense-array scaling

**Microseconds per call, median ± MAD.** Sizes count particles. Values are
rounded for readability; full local reports retain samples and finer precision.

| Case | 1,000 | 100,000 | 1,000,000 |
| --- | ---: | ---: | ---: |
| `construct` | 18.7 ± 0.2 | 222 ± 19 | 2,973 ± 340 |
| `mass_cold` | 14.4 ± 0.2 | 314 ± 10 | 6,103 ± 299 |
| `mass_hot` | 0.384 ± 0.006 | 0.388 ± 0.004 | 0.384 ± 0.003 |
| `mass_uncached` | 14.3 ± 0.3 | 318 ± 18 | 7,224 ± 1,143 |
| `properties` | 59.7 ± 1.4 | 6,030 ± 291 | 62,780 ± 1,629 |
| `properties_intermediates` | 66.1 ± 3.4 | 5,497 ± 133 | 62,744 ± 2,496 |
| `add` | 19.4 ± 0.4 | 204 ± 13 | 3,867 ± 196 |
| `boost` | 86.5 ± 2.4 | 1,452 ± 222 | 22,653 ± 3,847 |
| `rotate` | 43.2 ± 0.8 | 370 ± 19 | 7,627 ± 438 |
| `sum` | 60.9 ± 0.9 | 885 ± 13 | 14,941 ± 483 |
| `workflow` | 131 ± 2 | 2,241 ± 277 | 26,762 ± 1,134 |

Hot reads reuse a previously allocated array. Their nearly constant latency
measures cache access rather than arithmetic throughput.

### Jagged and irregular layouts

**Milliseconds per call, median ± MAD.** Sizes count events except for strided
input, where they count particles. Each row is a distinct population and shape.

| Layout | Size | Particles generated / visible | Cold mass ms | Boost ms | Sum ms | Workflow ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jagged | 1,000 | 3973 / 3973 | 0.201 ± 0.001 | 0.562 ± 0.010 | 1.872 ± 0.013 | 1.353 ± 0.007 |
| jagged | 100,000 | 401402 / 401402 | 3.110 ± 0.174 | 7.708 ± 0.361 | 10.769 ± 0.378 | 11.960 ± 0.215 |
| strided | 100,000 | 100000 / 100000 | 0.309 ± 0.011 | 1.212 ± 0.011 | 0.884 ± 0.009 | 1.807 ± 0.021 |
| missing | 10,000 | 39936 / 32395 | 2.566 ± 0.045 | 3.994 ± 0.021 | 10.066 ± 0.059 | 9.328 ± 0.080 |
| deep | 10,000 | 39936 / 39936 | 0.389 ± 0.006 | 1.161 ± 0.025 | 3.555 ± 0.043 | 2.494 ± 0.064 |
| heavy-tail | 10,000 | 13593 / 13593 | 0.242 ± 0.002 | 0.685 ± 0.008 | 2.279 ± 0.068 | 1.590 ± 0.046 |

### Incremental allocations

These are separate **single observations**, in MiB, for one million NumPy
particles. Retained allocation includes the live return value and any new cache
buffers. Peak traced allocation includes temporary traced storage.

| Case | Retained MiB | Peak MiB |
| --- | ---: | ---: |
| `construct` | 30.52 | 30.52 |
| `mass_cold` | 7.63 | 22.89 |
| `properties` | 30.52 | 55.32 |
| `properties_intermediates` | 68.67 | 78.21 |
| `workflow` | 7.63 | 91.56 |

Tracing excludes input storage, prior caches, interpreter baseline, and untraced
native allocations. These values are **not RSS or total memory use**. A cached
read can allocate very little during tracing while still retaining a large cache
created beforehand. Intermediate retention increased allocated storage for the
property sequence without a clear timing benefit in this run.

### Numerical recovery cost

**Milliseconds per call, median ± MAD.** Exceptional counts are fixed per batch,
so 100 exceptional entries represent different fractions at the two sizes.

| Case | 1,000 particles | 10,000 particles |
| --- | ---: | ---: |
| `cancellation_mass2` | 4.177 ± 0.018 | 43.774 ± 1.565 |
| `mixed_boost_0` | 0.087 ± 0.002 | 0.179 ± 0.011 |
| `mixed_boost_1` | 0.353 ± 0.018 | 0.608 ± 0.040 |
| `mixed_boost_100` | 20.796 ± 0.486 | 21.922 ± 1.424 |

The checks passed for every measured batch. Recovery has a substantial cost when
many entries need it; these timings describe successful recovery for the stated
inputs, not a guarantee of correct rounding for every possible calculation.

## Secondary vector comparison

This separate matched run uses **100,000 NumPy particles** or **100,000 jagged
events containing 401,402 particles**. Values are **milliseconds per call,
median ± MAD**. Both implementations receive the same input components and use
the same boost convention. Construction plus mass uses a new object each call;
recalculation uses existing objects with LVec storage disabled.

| Layout | Workload | LVec ms | vector ms |
| --- | --- | ---: | ---: |
| numpy | Construct + mass | 0.494 ± 0.011 | 0.823 ± 0.036 |
| numpy | Recalculate mass | 0.302 ± 0.019 | 0.428 ± 0.011 |
| numpy | Add + boost + mass | 1.963 ± 0.087 | 2.823 ± 0.078 |
| jagged | Construct + mass | 4.002 ± 0.371 | 4.128 ± 0.591 |
| jagged | Recalculate mass | 2.497 ± 0.240 | 3.170 ± 0.107 |
| jagged | Add + boost + mass | 11.749 ± 0.671 | 12.305 ± 0.248 |

The NumPy rows had lower LVec medians in this sample. Jagged construction plus
mass and the complete workflow were close; small differences are comparable to
observed dispersion. Recalculated jagged mass had a lower LVec median. These
observations do not support an aggregate package ranking.

Construction reflects each public API and representation: LVec owns read-only
component snapshots, while the comparison uses `vector.array` for NumPy and
`vector.zip` for Awkward. Copying and ownership costs are not normalized. No hot
LVec cache read is compared with `vector` arithmetic. Numerical stress cases are
excluded because the packages have different numerical-recovery policies.
The adapters follow the documented [NumPy constructor](https://vector.readthedocs.io/en/stable/src/make_numpy.html),
[Awkward constructor](https://vector.readthedocs.io/en/stable/src/make_awkward.html),
and [`boost_beta3`](https://vector.readthedocs.io/en/stable/src/vector4d.html) APIs.

## Interpretation and validation

- Compare matching case names, shapes, cache settings, source revisions, and
  dependency versions. Filtering does not change seeded inputs, but can change
  measurement order and allocation history. Use the same selections across
  revisions; the separate comparison run need not match the core timings exactly.
- Pooled samples can be correlated within each process; median/MAD values are descriptive.
  No confidence interval, statistical significance test, or portable speedup
  guarantee is claimed. Allocation measurements are not repeated timing samples.
- This is a synthetic CPU suite on one workstation. It does not measure I/O,
  ROOT ingestion, GPUs, application-scale analyses, or general physics accuracy.
  Physics regressions are checked separately with independent oracles.
- The integrated validation passed **474 tests each** on Python 3.13.12 with
  current dependencies, Python 3.13.12 with NumPy 2.2.4/Awkward 2.7.4, and Python
  3.15.2 with current dependencies. NumPy-only validation passed **232 tests**
  and skipped 214 optional checks. RuntimeWarnings were treated as errors.
- Framework checks run through the normal pytest configuration. Timing thresholds
  are not CI gates. Run `python -m pytest benchmarks/tests -W error::RuntimeWarning`
  to check the harness separately.

### Revision comparisons and research records

Use one framework and interpreter for both checkouts:

```bash
python -m benchmarks --source /path/to/baseline --layouts numpy jagged --sizes 100000 --output /tmp/before.md
python -m benchmarks --source /path/to/candidate --layouts numpy jagged --sizes 100000 --output /tmp/after.md
```

The source checkouts must support the selected LVec APIs. The runner verifies
which source was imported and detects source/environment changes between
workers. Use a quiet machine, consistent power settings, and repeated alternating
runs when a difference is small. Record CPU model, memory, OS, interpreter,
dependency versions, source/framework fingerprints, and complete commands.

The source tree stores code, tests, and this curated summary. Generated reports
belong under ignored `benchmarks/results/` or outside the checkout; generated
plots and JSON/JSONL/CSV under `benchmarks/` are also ignored. The CLI writes
Markdown and uses JSON only for in-memory worker communication. No external
samples or downloads are needed for benchmark execution.

For publication, retain the full Markdown reports (including raw samples) and
an environment export with the analysis or supplementary materials. A versioned
summary alone does not contain the raw data needed for reanalysis. See the
[root citation guidance](../README.md#citation) for software attribution.

## Core parity update

The expanded numerical API was measured on 1 November 2026 at clean revision
`1a178b86f3ead707004ca3eb0cd87bcb0ad687b7`. This update supplements the earlier
130-row scaling study; those earlier tables retain their original source and
framework versions. The expanded run contains **106 rows**, and the matched
revision study contains **108 rows** across six additional reports.

The environment and hardware are the same as described above: Python 3.13.12,
NumPy 2.5.4, Awkward 2.14.0, vector 1.9.0, Apple M1, 16 GiB, macOS 27.2 arm64,
and one numerical runtime thread. Samples target 20 ms with two warmups, seed
20261101. The expanded run uses three fresh processes and seven samples per
case-process. Allocation tracing was disabled for this focused timing update.
The large NumPy case has 100,000 particles; the large jagged case has 100,000
events and 401,402 particles, so their absolute times are not directly comparable.

- Library SHA256: `df2313a4bc40154e98382feb993a6445eb125298dc59c906cd1c817a8c6aaebc`.
- Framework SHA256: `946220ae829fd055f47e850ea794ea9b4657fa11261c4622ec3f4323c3218460`.
- Baseline revision: `195c5dedd309d5507890ef62ed4cb53d708c2f98`, with library
  SHA256 `d0c5dcb1f6c62a8bc2e0109f1b360f181eb66ba283c70fed780800416b93ad36`.
  Its source was extracted with `git archive`; its reports therefore say that
  Git revision metadata is unavailable, and the archive's reported dirty flag
  is not meaningful. The verified source fingerprint identifies it.

### New operations

Times below are **milliseconds per call, median ± MAD**, pooled from 21 samples.
Each timed call returns a new vector collection. Normalization clears the input
cache first; the fused normalization kernel computes the invariant itself.
Matrices are the specified diagonal fixture, rather than an arbitrary dense
matrix workload. Axis and Euler rotations use the fixtures in the workload table.

| Operation | NumPy LVec | NumPy vector | Jagged LVec | Jagged vector |
| --- | ---: | ---: | ---: | ---: |
| Four-vector normalization | 0.616 ± 0.015 | 2.286 ± 0.150 | 6.103 ± 1.916 | 11.863 ± 0.851 |
| Gamma-parameterized x boost | 0.436 ± 0.018 | 0.939 ± 0.068 | 5.015 ± 0.401 | 7.807 ± 0.651 |
| Diagonal 4x4 transform | 0.924 ± 0.014 | 1.916 ± 0.033 | 8.247 ± 2.234 | 7.481 ± 0.290 |
| Arbitrary-axis rotation | 1.101 ± 0.037 | 1.361 ± 0.038 | 11.332 ± 0.918 | 6.167 ± 0.940 |
| Euler rotation | 1.059 ± 0.042 | 1.457 ± 0.089 | 9.761 ± 1.310 | 4.791 ± 0.326 |

The NumPy normalization, gamma-boost, and diagonal-transform fixtures favor LVec.
The jagged gamma and normalization fixtures also favor LVec; the jagged matrix
comparison is noisy. Jagged arbitrary-axis and Euler rotations remain slower,
by approximately 1.8x and 2.0x in this run. Numerical risk detection and separate
owned components have costs; these results do not establish universal parity.

Small arrays also retain Python overhead. For one-element NumPy arrays, the
matrix fixture took 150.875 ± 2.303 µs in LVec versus 29.732 ± 0.291 µs in vector;
Euler rotation took 145.049 ± 2.119 µs versus 26.713 ± 0.333 µs. These are array
measurements, not Python-float scalar measurements. The size-one jagged input
is an empty event and must not be interpreted as a single-particle calculation.

Native Awkward record mass took 3.680 ± 0.659 ms versus 3.841 ± 0.283 ms
for vector's recomputed record mass. Native records have ordinary Awkward view
semantics and no retained property cache. Owned `mass_hot` represents a cached
read, so it is not a comparable calculation boundary.

### Matched revision checks

The first full-run comparison showed a 35% jagged workflow increase while the
unchanged vector workflow also increased by 39%. That observation prompted a
matched study rather than a no-regression claim. Both revisions then ran with
identical case selections and inputs, alternating in the order before/after,
after/before, before/after. Each invocation used one fresh process and seven
samples, giving 21 samples per revision-case across three rounds. Case order
was the same within each pair; the numerical seed did not change.

The table reports pooled **milliseconds per call, median ± MAD**. Ratios quoted
below are medians of the three within-round median ratios, which can differ
from ratios of these pooled medians.

| Layout | Case | Original main | Core parity revision |
| --- | --- | ---: | ---: |
| numpy | `construct_mass` | 0.618 ± 0.029 | 0.628 ± 0.042 |
| numpy | `mass_uncached` | 0.334 ± 0.020 | 0.319 ± 0.016 |
| numpy | `add` | 0.230 ± 0.009 | 0.207 ± 0.008 |
| numpy | `boost` | 1.523 ± 0.042 | 1.622 ± 0.100 |
| numpy | `rotate` | 0.404 ± 0.010 | 0.398 ± 0.006 |
| numpy | `workflow` | 2.351 ± 0.049 | 2.406 ± 0.117 |
| jagged | `construct_mass` | 5.227 ± 0.149 | 5.245 ± 0.236 |
| jagged | `mass_uncached` | 3.250 ± 0.222 | 3.209 ± 0.184 |
| jagged | `add` | 2.748 ± 0.106 | 2.800 ± 0.126 |
| jagged | `boost` | 8.775 ± 0.221 | 9.295 ± 0.287 |
| jagged | `rotate` | 8.728 ± 0.168 | 8.762 ± 0.161 |
| jagged | `workflow` | 14.940 ± 0.678 | 14.847 ± 0.758 |

The median paired workflow ratios were 1.000 for NumPy and 1.020 for jagged data;
the large apparent workflow penalty did not persist. Beta boosts had ratios
1.103 and 1.085, respectively, and remain a performance follow-up rather than
evidence of zero overhead. The numerical kernels for those established boosts
were retained; the measurements cannot attribute the observed differences to
one cause. The unchanged vector jagged workflow itself varied substantially
between paired invocations. Repeat on controlled target hardware before making
claims about differences of this size.

### Repeating the focused update

Use the dependency versions above and a clean checkout of the measured revision.
The expanded suite command is:

```bash
python -m benchmarks --layouts numpy jagged --sizes 1 100000 --compare-vector \
  --processes 3 --samples 7 --no-memory --output /tmp/lvec-core-expanded.md
```

For the matched study, prepare a separate checkout of the baseline revision.
Run the following with `--source` pointing to each revision, in the alternating
order above, retaining six separate Markdown reports:

```bash
python -m benchmarks --source /path/to/revision --layouts numpy jagged \
  --sizes 100000 --cases construct_mass mass_uncached add boost rotate workflow \
  vector.construct_mass vector.mass vector.workflow --compare-vector \
  --processes 1 --samples 7 --no-memory --output /tmp/lvec-core-paired-run.md
```

The current framework loads workload definitions from the invoking checkout
while `--source` selects the LVec implementation. Use the same framework for
both revisions. Source and framework fingerprints, exact invocations, timestamps,
loop counts, and raw samples are retained in the local reports. The source
archive and reports are kept outside Git; only this summary is tracked.

Validation for the implementation includes 544 passing tests with current
and minimum dependencies and on Python 3.15, plus 270 passing NumPy-only tests
with optional-backend tests skipped. See the [coverage matrix](../docs/vector-feature-parity.md)
and [migration guide](../docs/vector-migration.md) for numerical policies and
ownership contracts. Timing agreement does not replace independent physics checks.
