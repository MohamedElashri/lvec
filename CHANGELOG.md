# Changelog

## Unreleased

This entry describes the pending implementation stack; no new package version
has been published. See the [migration guide](docs/migration.md) before updating
existing analyses.

### Dependencies and correctness

- Require Python 3.10+ and NumPy 2.2.4+; make Awkward 2.7.4+ optional.
- Move test, example, plotting, and benchmark tools into extras.
- Preserve signed invariant mass and support general four-vector algebra with
  explicit physical-particle validation.
- Correct active/passive boost and reference-frame sign conventions.
- Snapshot real components into owned, read-only float64 arrays; reject
  incompatible shapes, missing-value masks, complex values, and nonnumeric data.
- Check minimum/current dependencies, NumPy-only installs, and distributions in CI.

### Analysis features

- Add squared momenta, Minkowski products, rapidity, beta/gamma, wrapped
  delta-phi, and eta/rapidity delta-R.
- Add axis-aware event four-momentum reductions and Awkward record round trips.
- Add rest-frame boosts, opening angles, and explicit helicity-angle conventions.
- Preserve empty collections and nested missing-value behavior.

### Performance and caches

- Cache final observables by default and make intermediate retention opt-in.
- Vectorize jagged shape/mask checks and reuse safe internal NumPy result buffers.
- Support zero-retention caches, monotonic TTLs, persistent property policies,
  optional hit/miss counters, and retained-byte reporting.
- Add seeded cold/hot timing, workflow, memory, and scaling benchmarks.

### Numerical stress fixes

- Recover extreme invariant residuals with compensated products/sums and exact
  binary accumulation; round signed mass directly when mass squared is outside
  float64's representable range.
- Use range-safe magnitudes, coordinate energy, pseudorapidity, and collinear
  angle calculations; fix nested optional-list masking overhead.
- Select additional arithmetic for risky boosts, rotations, and reduction
  groups, including intermediate overflow and near-boundary velocity validation.
- Evaluate float32 rotation angles at float64 component precision.
- Add independent Fraction, integer, and high-precision Decimal regression
  oracles, deterministic physics stress cases, and runtime-warning CI checks.
- Record the adaptive detection cost explicitly: large NumPy workloads measured
  about 1.6–2.5 times slower than the pre-adaptive baseline; jagged boosts and
  rotations were within about 7%, while jagged sums were about twice as costly.

### Documentation

- Refresh the API/usage guide, runnable quick starts, installation extras,
  migration guidance, examples index, numerical policy, and benchmark reproduction.
- Remove stale API examples and outputs that implied invariant mass changes
  under a Lorentz boost.
