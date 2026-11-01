"""Check workload validity and isolation rather than machine-dependent speed."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.cases import assert_values, build_cases, make_data
from benchmarks.runner import ROOT, allocations, measure, run_worker


def test_specs_allocate_only_the_current_collection(monkeypatch):
    import weakref
    import benchmarks.cases as module
    constructor = module.LVec
    created = []

    def tracked(*args, **kwargs):
        result = constructor(*args, **kwargs)
        created.append(weakref.ref(result))
        return result

    monkeypatch.setattr(module, "LVec", tracked)
    specs, _ = build_cases("numpy", 17, 42)
    assert not created
    for spec in specs:
        case = spec.prepare()
        assert sum(ref() is not None for ref in created) <= 1
        case.check(case.operation())
        del case
        assert all(ref() is None for ref in created)


@pytest.mark.parametrize("layout", ["numpy", "strided", "jagged", "missing", "deep", "heavy-tail", "extreme"])
@pytest.mark.parametrize("size", [1, 17])
def test_workloads_check_repeated_calls_and_edge_layouts(layout, size):
    if layout in {"jagged", "missing", "deep", "heavy-tail"}:
        pytest.importorskip("awkward")
    cases, _ = build_cases(layout, size, 42)
    for spec in cases:
        case = spec.prepare()
        for _ in range(2):
            case.check(case.operation())
        del case


def test_seeded_strided_inputs_are_reproducible_without_global_rng_changes():
    np.random.seed(32)
    state = np.random.get_state()
    left, metadata = make_data("strided", 17, 42)
    make_data("numpy", 1000, 99)
    right, repeated = make_data("strided", 17, 42)
    assert repeated == metadata
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)
        assert not a.flags.c_contiguous
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_structure_check_catches_missing_positions_even_if_flat_values_match():
    ak = pytest.importorskip("awkward")
    with pytest.raises(AssertionError, match="missing positions"):
        assert_values(ak.Array([[None, 1.], []]), ak.Array([[1., None], []]))


@pytest.mark.parametrize("layout", ["numpy", "jagged"])
def test_vector_comparison_checks_equivalent_outputs(layout):
    pytest.importorskip("vector")
    if layout == "jagged":
        pytest.importorskip("awkward")
    cases, _ = build_cases(layout, 17, 42, True)
    assert any(case.name == "vector.workflow" for case in cases)
    for spec in cases:
        case = spec.prepare()
        case.check(case.operation())
        del case


def test_memory_keeps_result_alive_and_does_not_instrument_timing():
    import tracemalloc
    result = allocations(lambda: np.ones(100_000))
    assert result["retained_bytes"] >= 800_000
    assert result["peak_bytes"] >= result["retained_bytes"]
    assert not tracemalloc.is_tracing()
    timing = measure(lambda: np.ones(10), samples=3, min_time=.001, warmups=1)
    assert len(timing["seconds"]) == 3
    assert all(t > 0 for t in timing["seconds"])
    assert not tracemalloc.is_tracing()


def config(source=ROOT):
    return dict(source=str(source), layout="numpy", size=17, seed=42,
                compare_vector=False, cases=["mass_cold", "workflow"], samples=1,
                min_time=.001, warmups=1, process=0, mode="check")


def test_worker_uses_requested_source_in_fresh_process(tmp_path):
    # A sentinel implementation proves the requested checkout wins over cwd.
    source = tmp_path / "checkout"
    package = source / "lvec"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise RuntimeError("requested checkout loaded")\n')
    with pytest.raises(RuntimeError, match="requested checkout loaded"):
        run_worker(config(source), timeout=30)
    result = run_worker(config(), timeout=30)
    assert set(result["rows"]) == {"mass_cold", "workflow"}
    assert Path(result["environment"]["loaded_module"]).resolve().parent == ROOT / "lvec"
    assert set(result["environment"]["threads"].values()) == {"1"}


def test_worker_propagates_invalid_workloads():
    invalid = dict(config(), cases=["typo"])
    with pytest.raises(RuntimeError, match="unknown cases"):
        run_worker(invalid, timeout=30)


@pytest.mark.parametrize("arguments", [["--sizes", "0"], ["--samples", "0"],
                                      ["--min-time", "nan"], ["--timeout", "inf"],
                                      ["--output", "results.json"],
                                      ["--source", "/nonexistent-lvec-checkout"],
                                      ["--layouts", "extreme", "--compare-vector"]])
def test_cli_rejects_invalid_configuration(arguments):
    result = subprocess.run([sys.executable, "-m", "benchmarks", *arguments],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert not result.stdout


def test_cli_outputs_markdown_without_result_files(tmp_path):
    output = tmp_path / "run.md"
    result = subprocess.run([sys.executable, "-m", "benchmarks", "--check", "--sizes", "17",
                             "--output", str(output)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = output.read_text()
    assert "Validation only" in report and "| numpy | 17 |" in report
    assert "source SHA256" in report and "Framework SHA256" in report
    assert list(tmp_path.iterdir()) == [output]


def test_cli_timing_and_memory_report(tmp_path):
    output = tmp_path / "timing.md"
    result = subprocess.run([sys.executable, "-m", "benchmarks", "--sizes", "17",
                             "--processes", "1", "--samples", "2", "--min-time", ".001",
                             "--cases", "mass_cold", "mass_hot", "--output", str(output)],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = output.read_text()
    assert "Median µs" in report and "Peak KiB" in report
    assert "numpy/17/mass_cold:" in report and "numpy/17/mass_hot:" in report
