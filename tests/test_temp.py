import numpy as np
from hypothesis import given, strategies


@given(size=strategies.integers(min_value=2, max_value=10))
def test_zeros(size):
    assert np.all(np.zeros(size) == 0)


def test_import():
    from QUnfold import QUnfoldQUBO

    assert isinstance(QUnfoldQUBO, type)
