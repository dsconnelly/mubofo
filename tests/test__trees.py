import numpy as np

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from mubofo import MultioutputDecisionTree

def test_feature_importances__sign():
    X, Y = make_regression(n_samples=1000, n_features=40, n_targets=10)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    model = MultioutputDecisionTree(max_depth=5).fit(X, Y)

    assert np.all(model.feature_importances_ >= 0)

def test_is_estimator():
    check_estimator(MultioutputDecisionTree())

def test_predictions():
    X, Y = make_regression(n_samples=200, n_features=40, n_targets=5)
    base = DecisionTreeRegressor(max_depth=3, random_state=123).fit(X, Y)
    model = MultioutputDecisionTree(max_depth=3, random_state=123).fit(X, Y)

    assert np.allclose(base.predict(X), model.predict(X))

def test_1d_importances():
    X, Y = make_regression(n_samples=200, n_features=40, n_targets=1)
    base = DecisionTreeRegressor(max_depth=3, random_state=123).fit(X, Y)
    model = MultioutputDecisionTree(max_depth=3, random_state=123).fit(X, Y)

    assert np.allclose(base.feature_importances_, model.feature_importances_)
