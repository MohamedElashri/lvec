"""Evaluation policy for Minkowski invariants and scalar products.

The fast policy (default) returns the ordinary binary64 result. The
accurate policy selects ill-conditioned or range-extreme entries after the
vectorized binary64 path and re-evaluates them from the original components.
Boost, sum, and rotation recovery are unaffected. The policy is held in a
context variable, so a new thread starts with the default policy rather
than inheriting the caller's.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_POLICIES = ('accurate', 'fast')
_policy = ContextVar('lvec_invariant_policy', default='fast')


def _checked(policy):
    if policy not in _POLICIES:
        raise ValueError(f"Invariant policy must be one of {_POLICIES}, not {policy!r}")
    return policy


def get_invariant_policy():
    """Return the active invariant policy, 'accurate' or 'fast'."""
    return _policy.get()


def set_invariant_policy(policy):
    """Set the invariant policy for the current context and return the previous one."""
    previous = _policy.get()
    _policy.set(_checked(policy))
    return previous


@contextmanager
def invariant_policy(policy):
    """Temporarily select 'accurate' or 'fast' invariant evaluation."""
    token = _policy.set(_checked(policy))
    try:
        yield
    finally:
        _policy.reset(token)


def recover_invariants():
    return _policy.get() == 'accurate'
