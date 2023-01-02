import pytest

from sklearn.datasets import make_regression

@pytest.fixture
def one_dimensional_data():
    return make_regression(n_samples=1000, random_state=123)

@pytest.fixture
def two_dimensional_data():
    return make_regression(n_samples=1000, n_targets=20, random_state=123)
