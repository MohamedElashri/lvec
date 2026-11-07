import pytest

import lvec


@pytest.fixture
def accurate_policy():
    """Evaluate invariants with recovery, which is opt-in from release 0.3.5."""
    with lvec.invariant_policy('accurate'):
        yield
