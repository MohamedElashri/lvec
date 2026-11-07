"""Invariant evaluation policy, blocked recovery, and coordinate warnings."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from fractions import Fraction
import warnings

import numpy as np
import pytest

import lvec
from lvec import LVec, CoordinateWarning
from lvec._exact import _blocked_compensated_sum, _compensated_sum


def light_particles(size=200_000, mass=0.1056583755, seed=7, momentum=30.):
    rng = np.random.default_rng(seed)
    p = rng.normal(0., momentum, (3, size))
    energy = np.sqrt((p*p).sum(axis=0) + mass*mass)
    return p[0], p[1], p[2], energy


def exact_mass(row):
    squares = [Fraction.from_float(float(value))**2 for value in row]
    invariant = squares[3] - sum(squares[:3])
    with localcontext() as context:
        context.prec = 100
        root = (Decimal(abs(invariant.numerator))/Decimal(invariant.denominator)).sqrt()
    return float(root.copy_sign(Decimal(invariant.numerator)))


def within_one_ulp(result, expected):
    result, expected = np.asarray(result), np.asarray(expected)
    return bool(np.all(abs(result - expected) <= np.spacing(abs(expected))))


def test_default_policy_is_accurate():
    assert lvec.get_invariant_policy() == 'accurate'


def test_fast_policy_returns_plain_binary64():
    px, py, pz, energy = light_particles()
    plain = energy*energy - (px*px + py*py + pz*pz)
    with lvec.invariant_policy('fast'):
        vector = LVec(px, py, pz, energy)
        assert np.array_equal(vector.mass2, plain)
        assert np.array_equal(vector.mass, np.sign(plain)*np.sqrt(abs(plain)))
    assert lvec.get_invariant_policy() == 'accurate'


def test_accurate_policy_recovers_light_masses():
    # Every entry lies inside the signed-mass recovery region, m/E < 1/64.
    rows = np.stack(light_particles(size=2000, momentum=1000.)).T
    assert np.all(rows[:, 3] > 64*0.1056583755)
    vector = LVec(*rows.T)
    expected = np.array([exact_mass(row) for row in rows])
    assert within_one_ulp(vector.mass, expected)
    with lvec.invariant_policy('fast'):
        assert not within_one_ulp(LVec(*rows.T).mass, expected)


def test_cache_separates_policies():
    px, py, pz, energy = light_particles(size=5000)
    vector = LVec(px, py, pz, energy)
    accurate = vector.mass.copy()
    with lvec.invariant_policy('fast'):
        fast = vector.mass
    assert not np.array_equal(accurate, fast)
    assert np.array_equal(vector.mass, accurate)
    with lvec.invariant_policy('fast'):
        assert np.array_equal(vector.mass, fast)


def test_scalar_dot_and_gamma_follow_policy():
    a = LVec(0., 0., 2.**60, 2.**60 + 2.**8)
    exact = Fraction(2**60 + 2**8)**2 - Fraction(2**60)**2
    assert a.mass2 == float(exact)
    assert a.dot(a) == float(exact)
    with lvec.invariant_policy('fast'):
        assert a.mass2 == (2.**60 + 2.**8)**2 - (2.**60)**2
        assert LVec(0., 0., 2.**60, 2.**60 + 2.**8).dot(a) == (2.**60 + 2.**8)**2 - (2.**60)**2


def test_awkward_records_follow_policy():
    ak = pytest.importorskip('awkward')
    import lvec.awkward as lawk
    px, py, pz, energy = (ak.Array([list(c[:3]), [], list(c[3:6])])
                          for c in light_particles(size=6, momentum=1000.))
    records = lawk.zip({'px': px, 'py': py, 'pz': pz, 'E': energy})
    flat = [np.asarray(ak.flatten(c)) for c in (px, py, pz, energy)]
    plain = flat[3]**2 - (flat[0]**2 + flat[1]**2 + flat[2]**2)
    with lvec.invariant_policy('fast'):
        assert np.array_equal(np.asarray(ak.flatten(records.mass2)), plain)
        assert np.array_equal(np.asarray(ak.flatten(records.mass)), np.sqrt(plain))
    expected = [exact_mass(row) for row in zip(*flat)]
    assert within_one_ulp(np.asarray(ak.flatten(records.mass)), expected)


def test_policy_is_context_local():
    def worker():
        return lvec.get_invariant_policy()
    with lvec.invariant_policy('fast'):
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(worker).result() == 'accurate'


def test_set_policy_returns_previous_and_validates():
    previous = lvec.set_invariant_policy('fast')
    try:
        assert previous == 'accurate'
        assert lvec.get_invariant_policy() == 'fast'
    finally:
        lvec.set_invariant_policy(previous)
    with pytest.raises(ValueError):
        lvec.set_invariant_policy('approximate')
    with pytest.raises(ValueError):
        with lvec.invariant_policy('exact'):
            pass


@pytest.mark.parametrize('dot', [False, True])
def test_blocked_compensated_sum_is_bit_identical(dot):
    rng = np.random.default_rng(11)
    size = 3*65536 + 17
    values = tuple(rng.normal(0., 10., size) for _ in range(8 if dot else 4))
    total, flags = _compensated_sum(values, dot=dot)
    blocked, blocked_flags = _blocked_compensated_sum(values, dot=dot)
    assert np.array_equal(total, blocked)
    assert np.array_equal(flags, blocked_flags)


def test_unrepresentable_signed_mass_warns():
    with pytest.warns(CoordinateWarning):
        vector = lvec.obj(px=3., py=0., pz=0., mass=-4.)
    assert vector.E == 0.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert lvec.obj(px=3., py=0., pz=0., mass=-2.).E == np.sqrt(5.)
        lvec.obj(px=3., py=0., pz=0., mass=2.)


def test_construction_ignores_fast_policy():
    with lvec.invariant_policy('fast'):
        fast = lvec.obj(px=2.**30, py=0., pz=0., mass=-1.)
    accurate = lvec.obj(px=2.**30, py=0., pz=0., mass=-1.)
    assert fast.E == accurate.E
