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
