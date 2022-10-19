import numpy as np
import random as rand
import pytest

@pytest.fixture
def seed_random():
    rand.seed(123)
    np.random.seed(123)


@pytest.fixture
def np_rng():
    return np.random.default_rng(123)
