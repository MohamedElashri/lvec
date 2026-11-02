# Contributing

Numerical counterexamples, physics examples, and improvements to the public API
are welcome through [issues](https://github.com/MohamedElashri/lvec/issues) and
pull requests. Include the input, expected result and independent reference,
Python/dependency versions, and array layout when reporting a numerical problem.

## Development checks

From a checkout:

```bash
python -m pip install -e ".[awkward,test,benchmark]"
python -m pytest -W error::RuntimeWarning
python -m benchmarks --check --layouts numpy jagged missing deep --sizes 1 17
```

Tests include independent Fraction/Decimal references, Lorentz invariants,
decay conservation, extreme inputs, missing data, and ownership/cache behavior.
Add an independent physics or numerical reference when changing a calculation.
Timings are measured separately; machine-dependent speed thresholds are not CI gates.
See [benchmark reproduction](benchmarks/README.md#reproduction).

The [CI workflow](.github/workflows/Build_Package.yml) covers Python 3.10–3.15,
minimum/current dependencies, NumPy-only installations, and distributions.

Use separate virtual environments for compatibility checks. Install minimum
versions with `python -m pip install -e ".[awkward,test]" -c requirements.txt`;
for NumPy-only checks install `.[test]` without the Awkward or benchmark extras.
Run the same test command in each environment; optional tests skip when their
backend is absent.

## Finding and changing numerical code

| Area | Files |
| --- | --- |
| Public vectors, observables, and named frames | `lvec/lvec.py`, `lvec/vectors.py`, `lvec/frame.py` |
| Shared mathematical API and NumPy protocols | `lvec/_protocol.py` |
| Coordinate factories and projections | `lvec/_coordinates.py` |
| Array validation, ownership, and optional backend | `lvec/backends.py`, `lvec/utils.py` |
| Native Awkward record behavior | `lvec/awkward.py` |
| Adaptive arithmetic and exact accumulation | `lvec/_adaptive.py`, `lvec/_exact.py` |
| Boost, matrix, and rotation kernels | `lvec/_boost.py`, `lvec/_transform.py`, `lvec/_rotation.py` |
| Independent checks and measured workloads | `lvec/tests/`, `benchmarks/cases.py` |

When changing a calculation:

1. State the coordinate convention, units, valid domain, and expected physics
   identity before implementing it. Keep signed invariants and the active/passive
   boost distinction consistent with the [physics guide](user-guide.md).
2. Keep an ordinary NumPy path and test Awkward masks, empty events, broadcasting,
   and scalar/array parameters. Numerical kernels should operate on leaves;
   repeated high-level Awkward expressions often repeat layout traversal.
3. Test against an independent analytic, Fraction, Decimal, or matrix reference.
   Cover cancellation, subnormal values, range limits, and numerical block
   boundaries when applicable. A comparison against vector alone is insufficient.
4. Measure complete calls with the same framework, seeded inputs, and timing
   boundaries before and after. Run timings separately from tests and builds,
   using fixed clean checkouts. Retain recovery checks when optimizing.

Public collections must snapshot numeric **and structural** buffers. Private
owned-result paths require proof that every retained buffer is fresh or already
owned; array-valued parameters can supply mutable offsets, indices, or masks.
Native Awkward records instead have view semantics: temporary numerical views
must not retain property caches, freeze external buffers, or escape as public
owned collections. Cache only immutable scalar transformation parameters.
Call `_new_matching` only after a kernel has established matching float64 fields;
external structure still needs snapshots unless ownership is also proved.
Use `_new` for results that still need shape validation.

For a targeted physics check, run:

```bash
python -m pytest lvec/tests/test_physics_regressions.py \
  lvec/tests/test_stress_physics.py lvec/tests/test_parameter_kernels.py \
  -W error::RuntimeWarning
```

## Examples

```bash
python -m pip install -e ".[examples]"
cd examples
python 07_jagged_arrays.py
python 10_reference_frames.py
```

For ROOT I/O scripts, run `mkdir -p samples` and `python create_test_data.py`
from `examples` first. That fixture demonstrates I/O and is not an
energy-conserving decay model. `08_lhcb_data.py` may download data and write
plots; check the input units before using it for analysis.

## Documentation

Use Python 3.11 or later to build the manual:

```bash
python -m pip install -e ".[docs,awkward]"
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
python -m sphinx -b doctest -W --keep-going docs docs/_build/doctest
python docs/check.py docs/_build/html
```

Open `docs/_build/html/index.html` to review the result. The HTML build checks
cross-references; doctests verify tutorial outputs, and `docs/check.py` checks
links, executable Python examples, and documented API signatures. Write
mathematics in LaTeX using `$...$` or `$$...$$`; avoid `\operatorname`.

Tutorials teach complete physics tasks. Reference pages give signatures,
parameters, returns, and domains. `docs/conf.py` includes `user-guide.md`,
`benchmarks/README.md`, and this file as manual pages without duplicating them.
The documentation workflow validates pull requests and deploys `main` to
GitHub Pages after its build and example checks pass.

## Workflow dependencies and releases

[Sanad](https://github.com/MohamedElashri/sanad) manages
`.github/workflows/actions.lock` under `.sanad.toml`. After changing workflow
actions, review with `sanad sync`, reconcile with `sanad sync --write`, then
run `sanad check`. Commit the lockfile with the workflow; do not edit it by hand.

Build and inspect release artifacts from a clean checkout:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

`pyproject.toml`, `lvec.__version__`, and tag `v<version>` must agree.
Publishing a GitHub release triggers the [release workflow](.github/workflows/python-publish.yml),
which tests, builds, checks, and publishes through PyPI Trusted Publishing.
Draft releases do not publish. Keep [CITATION.cff](CITATION.cff) and the
user-facing documentation consistent with the released behavior.

`hatch_build.py` converts relative README links to canonical GitHub URLs in
package metadata, so they work on PyPI while repository links follow the branch
being read. Check the built description and install the wheel outside the
checkout before a release; an editable install can hide missing packaged files.
