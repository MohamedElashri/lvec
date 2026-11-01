# LVec benchmarks

This framework measures LVec's scaling, cache tradeoffs, transformations, event
reductions, complete workflows, and numerical recovery. An optional comparison
with Scikit-HEP `vector` provides context for a few equivalent ordinary workloads.
The default run needs only NumPy; no datasets, downloads, plotting packages, or
benchmark services are used.

Only framework source, regression tests, and this curated summary belong in Git.
The old runners, generated plots, and JSON reports have been retired. Reports
are printed as Markdown, with optional local output; no JSON files are written.

## Run from a checkout

Use Python 3.10 or newer and run commands from the repository root:

```bash
python -m pip install -e ".[test]"
python -m benchmarks --check --sizes 1 17
python -m benchmarks
```

Install the optional workloads and reference implementation with:

```bash
python -m pip install -e ".[benchmark]"
python -m benchmarks --layouts numpy strided --sizes 1000 100000 1000000
python -m benchmarks --layouts jagged --sizes 1000 100000
python -m benchmarks --layouts missing deep heavy-tail --sizes 1000 10000
python -m benchmarks --layouts extreme --sizes 1000
python -m benchmarks --layouts numpy jagged --sizes 1000 100000 --compare-vector
```

`--check` validates outputs without performance measurement. A fast development
run can use `--processes 1 --samples 3 --min-time .005`; use the default three
processes, seven samples, and 20 ms target samples when recording results.
`--cases mass_cold workflow` selects exact case names; every requested name must
apply to each selected layout. `--no-memory` omits allocation measurement.
Progress goes to stderr, leaving stdout suitable for redirection.

```bash
python -m benchmarks --output benchmarks/results/local.md
python -m benchmarks --layouts jagged --sizes 1000 > /tmp/lvec-jagged.md
python -m pytest benchmarks/tests -W error::RuntimeWarning
```

`benchmarks/results/`, `benchmarks/plots/`, and JSON/JSONL/CSV output under
`benchmarks/` are ignored. Keep detailed local runs there or outside the checkout;
update this README with selected results and their exact run configuration.

## Workloads and measurement boundaries

All components are float64. Ordinary inputs have independent standard-normal
three-momenta and energy `sqrt(px² + py² + pz² + 4)`, giving mass 2. LVec's
default cache settings are used unless the case name states otherwise.

| Case | Timed work |
| --- | --- |
| `construct` | Public construction from existing components, including LVec's owned snapshots |
| `mass_cold` | Clear an existing object's cache and compute mass; includes cache clearing |
| `mass_hot` | Read a precomputed mass from a dedicated object's cache |
| `mass_uncached` | Compute mass with `max_cache_size=0`; no explicit clear |
| `properties` | Clear cache, then read pt, eta, phi, mass |
| `properties_intermediates` | Same property sequence with intermediate caching enabled |
| `add` | Add an existing vector collection to itself |
| `boost` | Boost by beta `(0.1, 0.05, 0.02)`, returning a fresh collection |
| `rotate` | Rotate about z by 0.47 radians, returning a fresh collection |
| `sum` | Sum four-momenta over groups described below |
| `workflow` | Add, boost, then compute the new collection's mass |

Flat `numpy` and `strided` sizes count particles. Strided construction receives
noncontiguous views for all four components. Flat reductions use groups of four,
padding the final group with zeros when necessary. Other flat operations keep
the original shape. Jagged sizes count **events**, so particle counts differ;
reports give both generated and visible particle counts.

`jagged` uses multiplicities 0–8 and guarantees an empty first event. `missing`
adds matching particle/event option masks with approximately 10% missing entries
at each level. `deep` adds two singleton list levels to the jagged layout.
`heavy-tail` uses integer Pareto(1.5) multiplicities capped at 128, including empty
events. Awkward reductions use the innermost axis: in the deep layout that
reduces particle lists and preserves the outer nesting. Missing particles are
skipped and missing events remain missing.

The separate `extreme` layout is capped at 10,000 particles:

- `cancellation_mass2` clears cache and computes mass squared for
  `(1e200, 1, 0, 1e200)`, checking exactly against -1 using rational arithmetic.
- `mixed_boost_0`, `mixed_boost_1`, and `mixed_boost_100` boost a normal batch
  containing 0, 1, or up to 100 exceptional entries with `E=px=1e308`, `py=1`,
  `pz=0` by beta `(-0.9, 0, 0)`. A 400-digit Decimal calculation from the original
  binary floats checks the exceptional output. Small sizes reduce/deduplicate
  these counts. Construction and reference calculations are outside timing.

For ordinary workloads, independent NumPy/Awkward formulas check component sums,
boosts, rotations, properties, and workflow masses. Checks include complete list
and option structure, not just flattened values. Every case is checked before
and after measurement. A mismatch, RuntimeWarning, worker failure, or timeout
fails the command; there is no success-shaped partial report.

## Timing, memory, and reproducibility

Each layout/size runs sequentially in three fresh interpreter processes. Each
case allocates its own collection during setup and releases it before the next
case; reference arrays are calculated only during validation. Each
case gets two explicit warmups, adaptive loop calibration, then seven samples
of at least approximately 20 ms each (or one slow call). Calibration is capped
at one million calls/sample. Case order is deterministically shuffled per
process. The report includes pooled median, median absolute deviation (MAD),
loops per sample, and the raw samples grouped by process. MAD describes observed
dispersion; it is not a confidence interval. Very small calls include Python
call overhead. `timeit` disables cyclic garbage collection during timing.

Allocation measurements run in an additional fresh process per cell after timing
has finished. Tracing starts after input creation, validation, and warmup. The
operation's result remains alive while retained/peak traced bytes are read;
these include traced NumPy buffers and Python bookkeeping allocated during that
call. Cached reads reuse buffers allocated before tracing. Values exclude input
storage, prior caches, interpreter overhead, untraced native allocations, and
total process memory; they are **not RSS** or a claim about total memory use.
Allocation results are single observations, separate from the timing samples.

The seed defaults to `20261101`, with a fresh local RNG per layout/size. Filtering
cases or changing run order does not change inputs. Workers set
`PYTHONHASHSEED=0` and OMP/OpenBLAS/MKL/Accelerate/NumExpr thread limits to one
before importing numeric libraries. Reports record the timestamp, executable,
Python/platform/CPU count, package versions, source path, Git revision and dirty
state, source/framework SHA256 fingerprints, thread settings, and invocation.
Input generation and validation are excluded from both measurements.

For revision comparisons, keep the framework and interpreter fixed and change
only the source checkout. Both checkouts must support the selected LVec APIs:

```bash
python -m benchmarks --source /path/to/baseline --sizes 1000 100000 --output /tmp/lvec-before.md
python -m benchmarks --source . --sizes 1000 100000 --output /tmp/lvec-after.md
```

The runner verifies the imported LVec comes from the requested checkout and
rejects source/environment changes between workers. Compare matching rows in the
reports, alternating before/after runs if differences are small. Use an idle
machine on consistent power/thermal settings and record hardware details with
published results. Dependency versions are recorded, not automatically pinned;
recreate the recorded versions explicitly for a comparison. Different library
versions, OS builds, allocators, CPUs, or background load can change results.
These benchmarks do not impose timing thresholds in CI. CI runs correctness
checks for the framework through the repository's normal pytest configuration.

## Secondary `vector` comparison

`--compare-vector` adds only ordinary `numpy`/`jagged` comparisons:

| LVec case | `vector` case | Shared boundary |
| --- | --- | --- |
| `construct` | `vector.construct` | Construct from the same existing components |
| `construct_mass` | `vector.construct_mass` | Fresh construction and mass calculation |
| `mass_uncached` | `vector.mass` | Recalculate mass on existing collections |
| `workflow` | `vector.workflow` | Add, boost with the same beta, compute mass |

LVec owns read-only component snapshots; `vector` has different representations,
ownership, and numerical recovery policies. Construction costs reflect those
public APIs. No artificial copying is added to equalize semantics. The hot LVec
cache is never used for a cross-package throughput claim. The comparison follows
the documented [`vector` NumPy constructor](https://vector.readthedocs.io/en/stable/src/make_numpy.html)
and [`boost_beta3` API](https://vector.readthedocs.io/en/latest/src/vector4d.html).
Awkward comparisons use `vector.zip`. Extreme numerical cases are LVec-specific
and excluded from cross-package comparisons. There is no aggregate winner score.

## Recorded results

Measurements on 2026-11-01 used Python **3.13.12**, NumPy **2.5.4**, Awkward
**2.14.0**, and `vector` **1.9.0**, on arm64 macOS 27.2 with eight logical CPUs
(CPU model not captured). All runs used the defaults: seed 20261101, three fresh
processes per cell, seven samples per process, two warmups, 20 ms target samples,
and one thread for each numeric runtime listed above. These are observations on
one machine; small differences require repeated runs before drawing conclusions.

LVec was version 0.2.0 at revision `d0221e4e66d4093cb62863fb87b8d726617ec3d3`,
with a dirty working tree from this benchmark rebuild. The measured library
source SHA256 was
`9e957359b2ae9ed4c01c667d68a7d728e1ca2d6cacde3184254f7953b05de7f5`;
the framework SHA256 was
`630f43890e91367741d8a442f649adf81c2fdde34727230b2f5e86ce3ae905d6`.

To recreate the dependency versions and workload selections:

```bash
python -m pip install numpy==2.5.4 awkward==2.14.0 vector==1.9.0
python -m pip install -e . --no-deps
python -m benchmarks --layouts numpy --sizes 1000 100000 1000000
python -m benchmarks --layouts jagged --sizes 1000 100000
python -m benchmarks --layouts strided missing deep heavy-tail --sizes 10000
python -m benchmarks --layouts extreme --sizes 1000 10000
python -m benchmarks --layouts numpy jagged --sizes 100000 --compare-vector --cases construct_mass vector.construct_mass mass_uncached vector.mass workflow vector.workflow
```

### LVec scaling

NumPy times in **µs/call**, shown as **median ± MAD**:

| Case | 1,000 particles | 100,000 particles | 1,000,000 particles |
| --- | ---: | ---: | ---: |
| `construct` | 18.52 ± 0.10 | 178.85 ± 2.74 | 2,210.21 ± 60.28 |
| `mass_cold` | 14.29 ± 0.11 | 305.33 ± 2.04 | 5,354.94 ± 309.07 |
| `mass_hot` | 0.38 ± 0.00 | 0.38 ± 0.01 | 0.38 ± 0.00 |
| `mass_uncached` | 13.89 ± 0.07 | 270.12 ± 20.05 | 4,993.91 ± 105.68 |
| `properties` | 58.70 ± 0.30 | 5,348.00 ± 71.38 | 60,173.79 ± 1,229.42 |
| `properties_intermediates` | 65.66 ± 1.01 | 5,556.74 ± 58.98 | 59,378.00 ± 551.67 |
| `add` | 18.95 ± 0.08 | 186.34 ± 6.24 | 3,282.32 ± 77.66 |
| `boost` | 84.31 ± 0.61 | 1,255.76 ± 28.82 | 17,745.60 ± 414.52 |
| `rotate` | 42.69 ± 0.07 | 349.90 ± 5.03 | 7,099.26 ± 433.17 |
| `sum` | 58.45 ± 0.42 | 878.39 ± 11.04 | 13,163.06 ± 1,659.17 |
| `workflow` | 123.71 ± 0.52 | 1,882.35 ± 45.31 | 26,200.04 ± 580.46 |

Cached reads stay near 0.38 µs as the buffer grows; they reuse existing arrays.
Fresh calculations and the workflow scale with input size. Intermediate caching
had similar property-sequence times at one million particles while retaining
more traced allocation, as the next table shows.

Separate allocation observations for **one million NumPy particles**, in MiB:

| Case | Retained traced MiB | Peak traced MiB |
| --- | ---: | ---: |
| `construct` | 30.52 | 30.52 |
| `mass_cold` | 7.63 | 22.89 |
| `mass_hot` | 0.00 | 0.00 |
| `properties` | 30.52 | 55.32 |
| `properties_intermediates` | 68.67 | 78.21 |
| `workflow` | 7.63 | 91.56 |

Hot-read allocation rounds to 0.00 MiB because the cached buffer predates tracing;
it does not mean the collection or cache has zero memory footprint.

Jagged scaling and irregular-layout checks below use **ms/call, median ± MAD**.
Sizes count events except for strided input, where they count particles. Particle
counts include both generated and visible entries. These layouts have different
shapes and populations, so their rows are not equal-work backend comparisons.

| Layout | Size | Particles generated / visible | Cold mass ms | Boost ms | Sum ms | Workflow ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jagged | 1,000 | 3973 / 3973 | 1.134 ± 0.011 | 2.696 ± 0.012 | 2.441 ± 0.009 | 5.941 ± 0.019 |
| jagged | 100,000 | 401402 / 401402 | 5.294 ± 0.135 | 26.170 ± 0.246 | 12.830 ± 0.542 | 44.867 ± 2.770 |
| strided | 10,000 | 10000 / 10000 | 0.035 ± 0.000 | 0.166 ± 0.001 | 0.134 ± 0.000 | 0.242 ± 0.011 |
| missing | 10,000 | 39936 / 32395 | 7.506 ± 0.117 | 12.469 ± 0.080 | 11.108 ± 0.070 | 27.890 ± 0.122 |
| deep | 10,000 | 39936 / 39936 | 1.690 ± 0.011 | 6.341 ± 0.055 | 5.941 ± 0.023 | 12.975 ± 0.089 |
| heavy-tail | 10,000 | 13593 / 13593 | 1.192 ± 0.010 | 3.450 ± 0.036 | 3.008 ± 0.058 | 7.336 ± 0.167 |

### Numerical recovery cost

Times in **ms/call, median ± MAD**. Exceptional counts are fixed per batch,
so the 100-entry case has a larger exceptional fraction in the smaller batch.

| Case | 1,000 particles | 10,000 particles |
| --- | ---: | ---: |
| `cancellation_mass2` | 4.260 ± 0.018 | 42.281 ± 0.194 |
| `mixed_boost_0` | 0.084 ± 0.001 | 0.165 ± 0.001 |
| `mixed_boost_1` | 0.334 ± 0.002 | 0.499 ± 0.004 |
| `mixed_boost_100` | 20.233 ± 0.039 | 20.373 ± 0.039 |

One exceptional boost adds roughly 0.25–0.33 ms here; 100 add about 20 ms.
Exact cancellation recovery also has a substantial per-entry cost. Every measured
case passed its independent output check; this is the cost of successful recovery
for these inputs, not a claim of universal exact rounding.

### Context from `vector`

This smaller comparison uses 100,000 NumPy particles or 100,000 jagged events
(401,402 particles), with **ms/call, median ± MAD**. Values come from a separate
matched run, so they need not exactly match the core table above.

| Layout | Workload | LVec ms | `vector` ms |
| --- | --- | ---: | ---: |
| numpy | Construct + mass | 0.488 ± 0.016 | 0.797 ± 0.026 |
| numpy | Recalculate mass | 0.291 ± 0.005 | 0.420 ± 0.005 |
| numpy | Add + boost + mass | 1.881 ± 0.041 | 2.844 ± 0.057 |
| jagged | Construct + mass | 17.305 ± 0.906 | 3.632 ± 0.079 |
| jagged | Recalculate mass | 4.728 ± 0.079 | 3.784 ± 0.099 |
| jagged | Add + boost + mass | 44.489 ± 0.840 | 12.218 ± 0.817 |

LVec had lower NumPy timings in this sample; `vector` had lower jagged timings,
particularly for construction plus mass and the complete workflow. Ownership,
layout, validation, caching, and numerical policies differ, so these results
support workload-specific decisions rather than a general package ranking.
