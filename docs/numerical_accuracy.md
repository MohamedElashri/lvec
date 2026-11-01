# Numerical accuracy and stress testing

LVec computes with float64 components in common units and the metric `(+,-,-,-)`.
The kernels select additional arithmetic for numerical risk; there is no global
high-precision mode or additional runtime dependency. See the
[migration guide](migration.md) for changes to signed mass, frame conventions,
and immutable component arrays.

| Calculation | Ordinary path | Recovery when selected |
| --- | --- | --- |
| Spatial magnitudes and coordinate energy | Vectorized squares or hypot | Scaled/hypot evaluation outside the safe range |
| Mass, mass squared, Minkowski dot | Vectorized metric sum | Compensated products/sums, then exact binary accumulation for unresolved or extreme entries |
| Boosts and frame gamma | Vectorized boost and range check | Accurate `1-beta²` near the boundary; local Decimal recomputation for selected exceptional boosts |
| Rotations | Vectorized linear combinations | Compensated/exact products for selected cancelled or nonfinite entries |
| Event sums | Native axis reduction and cancellation bound | Compensated short groups, fsum for long/extreme groups, exact binary recovery from intermediate overflow |

All public outputs remain float64. Accuracy is relative to the supplied binary
inputs; input rounding cannot be undone. The selective recovery and its detection
cost are both covered below.

## Reproducing the checks

Run the deterministic physics and cache-lifecycle stress cases with:

```bash
python -m pytest lvec/tests/test_stress_physics.py -W error
```

The suite checks 60,000 random boost/rotation/frame cases, 20,000 two-body
decays, 10,000 massless vectors, decimal boost references near light speed,
scales from 1e-200 to 1e200, collimated angles, nested missing values, and
bounded-cache churn. Comparisons normalize by component scales and use
condition-appropriate absolute tolerances for nearly null momenta.

For process-isolated runtime and memory scaling, run:

```bash
PYTHONPATH=. python benchmarks/stress_lvec.py \
    --sizes 1000 10000 100000 1000000 --repeats 5 > stress-results.jsonl
```

This covers contiguous and strided NumPy arrays, photons, jagged collections,
missing events/particles, deeply nested lists, and heavy-tailed multiplicities.
Missing/deep/heavy-tail cases stop at 100,000 events to bound memory usage.
JSON rows distinguish construction, cold/hot properties, complete workflows,
and reductions; cache buffers, traced property allocations, and process peak
RSS are separate measurements. Run the same script with `PYTHONPATH` pointing
at another checkout to compare versions. Timings are measurements, not CI
pass/fail thresholds. A failed worker makes the script exit unsuccessfully.
The [recorded stress report](../benchmarks/results/stress-2026-11-01.json)
contains the original environments, baseline comparisons, and test counts.
Its previously expected invariant failure is resolved by the follow-up
[invariant accuracy report](../benchmarks/results/invariant-accuracy-2026-11-01.json).

Construction checks component structure once; immutable components let
reductions reuse that guarantee. Awkward conditional calculations restore
missing-list masks only at optional list levels, avoiding artificial option
buffers for ordinary arrays. Numerical fast paths retain square/sum kernels
for normal NumPy magnitudes and use scaled/hypot fallbacks outside their range.

Magnitudes and physical classification avoid overflowing or underflowing
intermediate squares. The coordinate constructor computes energy using hypot,
and pseudorapidity has a logarithmic fallback for overflowing pz/pt. Opening and
helicity angles use atan2 of the cross-product magnitude and scalar product,
retaining small collinear angles that arccos would round to zero.
Ill-conditioned invariants and dot products use compensated products and sums.
For extreme component ranges or cancellation beyond that precision, an exact
binary integer accumulator preserves all input products until the final float64
rounding. Signed mass is rounded directly from the exact squared value, so a
squared invariant that overflows to infinity or underflows to signed zero can
still yield a finite, nonzero mass. NumPy batches select only affected entries;
Awkward layouts retain their lists and missing-value masks. These fallback paths
need more work than the ordinary vectorized calculation, but add no runtime
dependency. The compensated transformations follow
[Ogita, Rump and Oishi (2005)](https://doi.org/10.1137/030601818).

The explicit cancellation benchmark is opt-in and capped at 10,000 entries:

```bash
PYTHONPATH=. python benchmarks/stress_lvec.py --kinds extreme --sizes 1 1000 10000
```

It reports both the expected and observed squared mass for
`(px, py, pz, E)=(1e200, 1, 0, 1e200)`, alongside timing and memory. This
adversarial workload exercises exact integer accumulation for every entry;
near-null photon workloads also require more compensated arithmetic than
ordinary massive vectors.

Float64 still limits squared observables: an unrepresentable `mass2`, `pt2`,
or `p2` can be infinite or zero even when its square root is representable.
Near-lightlike invariant masses suffer cancellation; boosts and inverse boosts
become ill-conditioned as beta approaches one. Precision lost in supplied
energy/momentum components cannot be recovered by these methods. Public
components and results remain float64. The former expected-failure example
`LVec(1e200, 1, 0, 1e200)` now has `mass2=-1`, `mass=-1`, and `dot(v, v)=-1`.
It is a normal passing regression test, also checked under axis permutations,
mixed array batches, and jagged missing values. Strict particle validation
retains the sign of negative squared invariants even when they round to -0.


## Adaptive transforms and event sums

Boosts, rotations, and event sums also select precision recovery by numerical
risk. Ordinary well-scaled boosts stay in the vectorized kernel after a buffer
range check. Near the speed-of-light boundary, velocity validation and frame
`gamma` recover `1-beta²` before rounding can turn a valid velocity into an
invalid one. Exceptional boosts recompute only affected entries from the
original float64 components using a local Decimal context; no decimal objects
are constructed for ordinary entries. This also recovers finite boost results
whose intermediate expressions overflow. Rotations recover cancelled linear
combinations using the binary product-sum kernel, and evaluate float32 angle
arrays at float64 precision.

Event reductions detect severe cancellation and intermediate range failures.
Only affected groups receive compensated summation (short groups) or
[`math.fsum`](https://docs.python.org/3/library/math.html#math.fsum) (long or
extreme groups). An exact binary accumulator handles intermediate overflow in
fsum. Groups with at most two entries retain native addition. The recovery
preserves NumPy axes/keepdims and Awkward ragged lists, missing particles, and
missing events; ordinary groups do not enter a Python loop over events.

## Measured performance cost

Detection itself has a cost. The
[adaptive precision report](../benchmarks/results/adaptive-precision-2026-11-01.json)
compares the same seeded workloads with commit `6ab83bf`. At one million
entries, NumPy boost/rotation/sum times were approximately 17.0/6.6/12.7 ms,
versus 10.9/2.6/6.4 ms before these checks. Jagged-array boosts and rotations
were within about 7% of baseline; jagged sums took about twice as long.
These measurements include guards even when no high-precision fallback runs.
Mixed-batch probes verify that 0, 1, or 1,000 extreme vectors in 100,000 entries
recover exactly that many entries. The 1,000-entry fallback costs about 0.2 s;
its extra arithmetic is deliberately reserved for affected entries.

To reproduce the ordinary and mixed-batch measurements:

```bash
PYTHONPATH=. python benchmarks/adaptive_precision.py --before /path/to/baseline --after .
PYTHONPATH=. python benchmarks/adaptive_precision.py --probe --repeats 3
```

The risk thresholds identify severe cancellation (roughly `sqrt(eps)` of the
sum bound), extreme component hierarchies, and near-light-speed boosts. They
preserve ordinary float64 rounding behavior within that scale; they do not
promise exact rounding for every possible calculation. Float64 inputs and
outputs still limit recoverable precision. Regression tests compare exceptional
cases with Fraction, integer, and high-precision Decimal oracles, and verify that
ordinary inputs do not call precision fallbacks. CI treats runtime warnings as
errors during tests.
